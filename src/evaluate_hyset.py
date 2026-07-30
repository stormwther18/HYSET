from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from hyset_model import HYSETModel, func_name_of


SPLITS = {
    "I1-Inst": ("G1_query.json", "G1_instruction_test_query_ids.json"),
    "I1-Tool": ("G1_query.json", "G1_tool_test_query_ids.json"),
    "I1-Cate": ("G1_query.json", "G1_category_test_query_ids.json"),
    "I2-Inst": ("G2_query.json", "G2_instruction_test_query_ids.json"),
    "I2-Cate": ("G2_query.json", "G2_category_test_query_ids.json"),
    "I3-Inst": ("G3_query.json", "G3_instruction_test_query_ids.json"),
}


def query_text(item: dict) -> str:
    value = item.get("query", item.get("instruction", ""))
    if isinstance(value, list):
        return " ".join(str(part) for part in value).strip()
    return str(value).strip()


def ground_truth_names(item: dict) -> List[str]:
    names: List[str] = []
    relevant = item.get("relevant APIs", item.get("relevant_apis", []))
    for entry in relevant:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            names.append(func_name_of(str(entry[0]), str(entry[1])))
        elif isinstance(entry, str):
            names.append(entry)
    if not names:
        for entry in item.get("api_list", []):
            if isinstance(entry, dict) and entry.get("tool_name") and entry.get("api_name"):
                names.append(func_name_of(str(entry["tool_name"]), str(entry["api_name"])))
    return list(dict.fromkeys(names))


def load_split(query_path: Path, id_path: Path) -> List[dict]:
    with query_path.open(encoding="utf-8") as stream:
        records = json.load(stream)
    with id_path.open(encoding="utf-8") as stream:
        selected_ids = {str(value) for value in json.load(stream)}
    matched = [item for item in records if str(item.get("query_id", item.get("id", ""))) in selected_ids]
    matched_ids = {str(item.get("query_id", item.get("id", ""))) for item in matched}
    missing_ids = sorted(selected_ids - matched_ids)
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(
            f"{query_path} is missing {len(missing_ids)} requested test IDs "
            f"(first: {preview})"
        )
    return matched


def recall_at_k(ranking: Sequence[str], ground_truth: Sequence[str], k: int) -> float:
    return len(set(ranking[:k]) & set(ground_truth)) / len(set(ground_truth))


def ndcg_at_k(ranking: Sequence[str], ground_truth: Sequence[str], k: int) -> float:
    relevant = set(ground_truth)
    dcg = sum(1.0 / math.log2(position + 2) for position, tool in enumerate(ranking[:k]) if tool in relevant)
    idcg = sum(1.0 / math.log2(position + 2) for position in range(min(k, len(relevant))))
    return dcg / idcg if idcg else 0.0


def comp_at_k(ranking: Sequence[str], ground_truth: Sequence[str], k: int) -> float:
    return float(set(ground_truth).issubset(set(ranking[:k])))


def exact_set_match(predicted: Sequence[str], ground_truth: Sequence[str]) -> float:
    return float(set(predicted) == set(ground_truth))


def average_metrics(records: Sequence[dict]) -> dict:
    metrics: Dict[str, float] = {}
    for k in (3, 5):
        metrics[f"Recall@{k}"] = sum(record[f"Recall@{k}"] for record in records) / len(records)
        metrics[f"NDCG@{k}"] = sum(record[f"NDCG@{k}"] for record in records) / len(records)
        metrics[f"COMP@{k}"] = sum(record[f"COMP@{k}"] for record in records) / len(records)
    metrics["PredictedSetExactMatch"] = sum(record["PredictedSetExactMatch"] for record in records) / len(records)
    metrics["MeanPredictedCardinality"] = sum(len(record["predicted_set"]) for record in records) / len(records)
    return metrics


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate paper-aligned HYSET")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default=os.getenv("HYSET_ENCODER_PATH", ""))
    parser.add_argument("--instruction_dir", default=str(root / "data" / "instruction"))
    parser.add_argument("--test_id_dir", default=str(root / "data" / "test_query_ids"))
    parser.add_argument("--output_dir", default=str(root / "results" / "hyset"))
    parser.add_argument("--k1", type=int, default=15)
    parser.add_argument("--k_pool", type=int, default=20)
    parser.add_argument("--ranking_length", type=int, default=5)
    parser.add_argument("--encode_batch_size", type=int, default=32)
    parser.add_argument("--candidate_batch_size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.encoder:
        parser.error("--encoder or HYSET_ENCODER_PATH is required")
    return args


def main() -> None:
    args = parse_args()
    model = HYSETModel.load(args.checkpoint, args.encoder, device=args.device)
    instruction_dir = Path(args.instruction_dir)
    test_id_dir = Path(args.test_id_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[dict] = []
    split_summaries: Dict[str, dict] = {}
    for split_name, (query_file, id_file) in SPLITS.items():
        queries = load_split(instruction_dir / query_file, test_id_dir / id_file)
        split_records: List[dict] = []
        for start in range(0, len(queries), args.encode_batch_size):
            batch = queries[start : start + args.encode_batch_size]
            texts = [query_text(item) for item in batch]
            embeddings = model.encode_texts(texts, batch_size=args.encode_batch_size)
            for item, text, embedding in zip(batch, texts, embeddings):
                query_id = str(item.get("query_id", item.get("id", "")))
                ground_truth = ground_truth_names(item)
                if not ground_truth:
                    raise ValueError(
                        f"Query {query_id!r} in split {split_name!r} has no ground-truth tools."
                    )
                missing_tools = [name for name in ground_truth if name not in model.fn_to_idx]
                if missing_tools:
                    raise ValueError(
                        f"Query {query_id!r} references tools absent from the checkpoint corpus: "
                        f"{missing_tools}"
                    )
                result = model.scorer.retrieve_from_embedding(
                    embedding,
                    k1=args.k1,
                    k_pool=args.k_pool,
                    ranking_length=args.ranking_length,
                    candidate_batch_size=args.candidate_batch_size,
                )
                ranking = [model.func_names[index] for index in result.ranking]
                predicted_set = [model.func_names[index] for index in result.predicted_set]
                record = {
                    "query_id": query_id,
                    "query": text,
                    "ground_truth": ground_truth,
                    "predicted_set": predicted_set,
                    "ranking": ranking,
                    "predicted_score": result.predicted_score,
                    "Recall@3": recall_at_k(ranking, ground_truth, 3),
                    "Recall@5": recall_at_k(ranking, ground_truth, 5),
                    "NDCG@3": ndcg_at_k(ranking, ground_truth, 3),
                    "NDCG@5": ndcg_at_k(ranking, ground_truth, 5),
                    "COMP@3": comp_at_k(ranking, ground_truth, 3),
                    "COMP@5": comp_at_k(ranking, ground_truth, 5),
                    "PredictedSetExactMatch": exact_set_match(predicted_set, ground_truth),
                }
                split_records.append(record)

        if not split_records:
            raise RuntimeError(f"no evaluable records found for {split_name}")
        summary = {"split": split_name, "queries": len(split_records), **average_metrics(split_records)}
        split_summaries[split_name] = summary
        all_records.extend(split_records)
        with (output_dir / f"{split_name}_predictions.json").open("w", encoding="utf-8") as stream:
            json.dump(split_records, stream, indent=2, ensure_ascii=False)
        print(split_name + " " + " ".join(f"{key}={value:.4f}" for key, value in summary.items() if isinstance(value, float)))

    overall = {"queries": len(all_records), **average_metrics(all_records), "splits": split_summaries}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(overall, stream, indent=2, ensure_ascii=False)
    print("Overall " + " ".join(f"{key}={value:.4f}" for key, value in overall.items() if isinstance(value, float)))


if __name__ == "__main__":
    main()
