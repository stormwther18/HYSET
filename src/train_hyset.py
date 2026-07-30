from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from hyset_model import HYSETModel, func_name_of


@dataclass(frozen=True)
class QueryExample:
    query_id: str
    query: str
    ground_truth: Tuple[int, ...]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def canonical_tool_set(names: Iterable[str]) -> str:
    return "|".join(sorted(set(names)))


def _query_text(item: dict) -> str:
    value = item.get("query", item.get("instruction", ""))
    if isinstance(value, list):
        return " ".join(str(part) for part in value).strip()
    return str(value).strip()


def _ground_truth_names(item: dict) -> List[str]:
    names: List[str] = []
    relevant = item.get("relevant APIs", item.get("relevant_apis", []))
    for entry in relevant:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            names.append(func_name_of(str(entry[0]), str(entry[1])))
        elif isinstance(entry, str):
            names.append(entry)

    if not names:
        for entry in item.get("api_list", []):
            if not isinstance(entry, dict):
                continue
            tool = entry.get("tool_name", "")
            api = entry.get("api_name", "")
            if tool and api:
                names.append(func_name_of(str(tool), str(api)))
    return list(dict.fromkeys(names))


def load_test_ids(directory: Path) -> set[str]:
    ids: set[str] = set()
    if not directory.exists():
        return ids
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            ids.update(str(value) for value in json.load(stream))
    return ids


def load_examples(
    paths: Sequence[Path],
    fn_to_idx: Dict[str, int],
    m_max: int,
    excluded_ids: set[str] | None = None,
) -> List[QueryExample]:
    excluded_ids = excluded_ids or set()
    examples: List[QueryExample] = []
    skipped = 0
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            records = json.load(stream)
        for item in records:
            query_id = str(item.get("query_id", item.get("id", "")))
            query = _query_text(item)
            if not query_id:
                query_id = query_hash(query)
            if query_id in excluded_ids or not query:
                continue
            indices = tuple(
                sorted({fn_to_idx[name] for name in _ground_truth_names(item) if name in fn_to_idx})
            )
            if not 1 <= len(indices) <= m_max:
                skipped += 1
                continue
            examples.append(QueryExample(query_id, query, indices))
    if skipped:
        print(f"Skipped {skipped} records with missing tools or cardinality outside 1..{m_max}.")
    return examples


class HardNeighborIndex:
    def __init__(
        self,
        model: HYSETModel,
        required_tools: Iterable[int],
        neighbor_count: int = 50,
        chunk_size: int = 256,
    ):
        z = F.normalize(model.scorer.Z.weight.detach(), dim=-1)
        required = sorted(set(int(index) for index in required_tools))
        count = min(neighbor_count + 1, model.num_tools)
        self.neighbors: Dict[int, Tuple[int, ...]] = {}
        for start in range(0, len(required), chunk_size):
            tool_ids = required[start : start + chunk_size]
            rows = torch.tensor(tool_ids, device=z.device)
            similarities = z[rows] @ z.T
            top = torch.topk(similarities, k=count, dim=1).indices.cpu().tolist()
            for tool_id, candidates in zip(tool_ids, top):
                filtered = tuple(candidate for candidate in candidates if candidate != tool_id)
                self.neighbors[tool_id] = filtered[:neighbor_count]

    def get(self, tool_id: int) -> Tuple[int, ...]:
        return self.neighbors.get(tool_id, ())


def _mixture_quotas(total: int) -> Tuple[int, int, int]:
    weights = (0.5, 0.3, 0.2)
    raw = [total * weight for weight in weights]
    quotas = [math.floor(value) for value in raw]
    remainder = total - sum(quotas)
    order = sorted(range(3), key=lambda index: raw[index] - quotas[index], reverse=True)
    for index in order[:remainder]:
        quotas[index] += 1
    return tuple(quotas)


class NegativeSampler:
    def __init__(
        self,
        num_tools: int,
        k_neg: int,
        rng: random.Random,
        neighbors: HardNeighborIndex,
    ):
        if k_neg < 2:
            raise ValueError("k_neg includes the positive and must be at least 2")
        self.num_tools = num_tools
        self.k_neg = k_neg
        self.rng = rng
        self.neighbors = neighbors

    def _uniform(self, cardinality: int) -> Tuple[int, ...]:
        return tuple(sorted(self.rng.sample(range(self.num_tools), cardinality)))

    def _hard(self, ground_truth: Tuple[int, ...]) -> Tuple[int, ...] | None:
        replacement_count = 1 if len(ground_truth) == 1 else self.rng.choice((1, 2))
        positions = self.rng.sample(range(len(ground_truth)), replacement_count)
        candidate = list(ground_truth)
        for position in positions:
            options = list(self.neighbors.get(ground_truth[position]))
            self.rng.shuffle(options)
            replacement = next((tool for tool in options if tool not in candidate), None)
            if replacement is None:
                return None
            candidate[position] = replacement
        result = tuple(sorted(candidate))
        return result if result != ground_truth and len(set(result)) == len(result) else None

    def build_pool(self, example: QueryExample, batch: Sequence[QueryExample]) -> List[Tuple[int, ...]]:
        target = self.k_neg - 1
        uniform_quota, in_batch_quota, hard_quota = _mixture_quotas(target)
        positive = example.ground_truth
        negatives: List[Tuple[int, ...]] = []
        seen = {positive}

        def add(candidate: Tuple[int, ...] | None) -> bool:
            if candidate is None or candidate in seen:
                return False
            seen.add(candidate)
            negatives.append(candidate)
            return True

        attempts = 0
        while len(negatives) < uniform_quota and attempts < uniform_quota * 100:
            add(self._uniform(len(positive)))
            attempts += 1

        in_batch = [other.ground_truth for other in batch if other.query_id != example.query_id]
        self.rng.shuffle(in_batch)
        in_batch_added = 0
        for candidate in in_batch:
            if add(candidate):
                in_batch_added += 1
            if in_batch_added >= in_batch_quota:
                break

        hard_added = 0
        attempts = 0
        while hard_added < hard_quota and attempts < hard_quota * 100:
            if add(self._hard(positive)):
                hard_added += 1
            attempts += 1

        attempts = 0
        while len(negatives) < target and attempts < target * 1000:
            add(self._uniform(len(positive)))
            attempts += 1
        if len(negatives) != target:
            raise RuntimeError("could not construct K_neg distinct candidate sets")
        return [positive] + negatives


class RewardCache:
    def __init__(self, path: str, rewarded_query_ids: set[str]):
        self.path = path
        self.rewarded_query_ids = rewarded_query_ids
        self.records: dict = {}
        self.load()

    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            self.records = {}
            return
        with open(self.path, encoding="utf-8") as stream:
            payload = json.load(stream)
        self.records = payload.get("records", payload)
        print(f"Loaded execution rewards for {len(self.records)} query keys.")

    def get(self, example: QueryExample, candidate: Sequence[int], func_names: Sequence[str]) -> float:
        if example.query_id not in self.rewarded_query_ids:
            return 0.0
        set_key = canonical_tool_set(func_names[index] for index in candidate)
        for query_key in (example.query_id, query_hash(example.query)):
            query_records = self.records.get(query_key, {})
            record = query_records.get(set_key) if isinstance(query_records, dict) else None
            if isinstance(record, dict):
                return float(max(0.0, min(1.0, record.get("score", 0.0))))
            if isinstance(record, (float, int)):
                return float(max(0.0, min(1.0, record)))
        return 0.0


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


@torch.no_grad()
def validation_recall_at_5(
    model: HYSETModel,
    examples: Sequence[QueryExample],
    k1: int,
    k_pool: int,
    encode_batch_size: int,
    limit: int | None,
) -> float:
    model.eval()
    selected = list(examples[:limit] if limit else examples)
    if not selected:
        return 0.0
    values: List[float] = []
    for start in range(0, len(selected), encode_batch_size):
        batch = selected[start : start + encode_batch_size]
        embeddings = model.encode_texts([example.query for example in batch], encode_batch_size)
        for example, query_embedding in zip(batch, embeddings):
            output = model.scorer.retrieve_from_embedding(
                query_embedding,
                k1=k1,
                k_pool=k_pool,
                ranking_length=5,
            )
            values.append(len(set(output.ranking[:5]) & set(example.ground_truth)) / len(example.ground_truth))
    model.train()
    return float(sum(values) / len(values))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train paper-aligned HYSET")
    parser.add_argument("--encoder", default=os.getenv("HYSET_ENCODER_PATH", ""))
    parser.add_argument("--encoder_type", choices=("bert", "qwen"), default="bert")
    parser.add_argument("--d_z", type=int, default=768)
    parser.add_argument("--m_max", type=int, default=5)  # maximum set cardinality
    parser.add_argument("--corpus", default=str(root / "data" / "hyset_corpus.json"))
    parser.add_argument("--instruction_dir", default=str(root / "data" / "instruction"))
    parser.add_argument("--test_id_dir", default=str(root / "data" / "test_query_ids"))
    parser.add_argument("--val_file", default="")
    parser.add_argument("--val_fraction", type=float, default=0.02)
    parser.add_argument("--val_limit", type=int, default=None)
    parser.add_argument("--checkpoint_dir", default=str(root / "checkpoints" / "hyset"))
    parser.add_argument("--reward_cache", default="")
    parser.add_argument("--reward_subset", type=int, default=5000)  # reward-annotated queries
    parser.add_argument("--reward_refresh_steps", type=int, default=20000)  # T_ref
    parser.add_argument("--reward_refreshes", type=int, default=4)
    parser.add_argument("--k_neg", type=int, default=64)  # candidate pool size
    parser.add_argument("--hard_neighbor_k", type=int, default=50)
    parser.add_argument("--k1", type=int, default=15)  # singleton shortlist size
    parser.add_argument("--k_pool", type=int, default=20)  # reranking shortlist size
    parser.add_argument("--eta", type=float, default=0.3)  # self-training weight
    parser.add_argument("--lambda_interaction", type=float, default=0.01)  # interaction regularization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--encode_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.encoder:
        parser.error("--encoder or HYSET_ENCODER_PATH is required")
    if not 0.0 <= args.val_fraction < 1.0:
        parser.error("--val_fraction must be in [0, 1)")
    if args.eta < 0 or args.lambda_interaction < 0:
        parser.error("--eta and --lambda_interaction must be non-negative")
    if args.eta > 0 and not args.reward_cache:
        parser.error("--reward_cache is required when --eta is positive; use --eta 0 for A-only")
    if args.reward_cache and not Path(args.reward_cache).is_file():
        parser.error(f"reward cache does not exist: {args.reward_cache}")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup_ratio must be in [0, 1)")
    if not 1 <= args.k1 < args.k_pool:
        parser.error("shortlist sizes must satisfy 1 <= k1 < k_pool")
    if args.m_max < 1 or args.k_pool < args.m_max:
        parser.error("m_max must be positive and no larger than k_pool")
    for name in (
        "d_z",
        "k_neg",
        "hard_neighbor_k",
        "batch_size",
        "encode_batch_size",
        "epochs",
        "patience",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.reward_subset < 0 or args.reward_refresh_steps < 0 or args.reward_refreshes < 0:
        parser.error("--reward_subset, --reward_refresh_steps, and --reward_refreshes must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    with open(args.corpus, encoding="utf-8") as stream:
        corpus = json.load(stream)
    func_names = list(corpus)
    fn_to_idx = {name: index for index, name in enumerate(func_names)}
    api_texts = [corpus[name]["text"] for name in func_names]

    instruction_dir = Path(args.instruction_dir)
    train_paths = [instruction_dir / name for name in ("G1_query.json", "G2_query.json", "G3_query.json")]
    missing = [str(path) for path in train_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing ToolBench instruction files: " + ", ".join(missing))
    excluded = load_test_ids(Path(args.test_id_dir))
    all_examples = load_examples(train_paths, fn_to_idx, args.m_max, excluded)
    if not all_examples:
        raise RuntimeError("no training examples were loaded")

    split_rng = random.Random(args.seed)
    split_rng.shuffle(all_examples)
    if args.val_file:
        validation = load_examples([Path(args.val_file)], fn_to_idx, args.m_max)
        training = all_examples
    else:
        val_count = max(1, int(len(all_examples) * args.val_fraction))
        validation = all_examples[:val_count]
        training = all_examples[val_count:]
    print(f"Training examples: {len(training)}; validation examples: {len(validation)}")

    model = HYSETModel(
        encoder_path=args.encoder,
        func_names=func_names,
        d_z=args.d_z,
        m_max=args.m_max,
        encoder_type=args.encoder_type,
    ).to(device)
    model.init_tool_embeddings(api_texts, batch_size=args.encode_batch_size)
    print(f"Trainable parameters: {model.trainable_parameter_count():,}")

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(training) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler = cosine_warmup_scheduler(optimizer, total_steps, args.warmup_ratio)

    subset_rng = random.Random(args.seed)
    subset_size = min(args.reward_subset, len(training))
    rewarded_ids = {example.query_id for example in subset_rng.sample(training, subset_size)}
    rewards = RewardCache(args.reward_cache, rewarded_ids)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_recall = -1.0
    stale_epochs = 0
    global_step = 0
    reward_refresh_count = 0

    safe_config = {
        "encoder_type": args.encoder_type,
        "d_z": args.d_z,
        "m_max": args.m_max,
        "k_neg": args.k_neg,
        "k1": args.k1,
        "k_pool": args.k_pool,
        "eta": args.eta,
        "lambda_interaction": args.lambda_interaction,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "reward_subset": args.reward_subset,
        "reward_refresh_steps": args.reward_refresh_steps,
        "reward_refreshes": args.reward_refreshes,
        "seed": args.seed,
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_rng = random.Random(args.seed + epoch)
        epoch_examples = list(training)
        epoch_rng.shuffle(epoch_examples)
        required_tools = (index for example in training for index in example.ground_truth)
        neighbor_index = HardNeighborIndex(model, required_tools, neighbor_count=args.hard_neighbor_k)
        sampler = NegativeSampler(model.num_tools, args.k_neg, epoch_rng, neighbor_index)

        running_ret = 0.0
        running_self = 0.0
        for start in range(0, len(epoch_examples), args.batch_size):
            batch = epoch_examples[start : start + args.batch_size]
            query_embeddings = model.encode_texts([example.query for example in batch], args.encode_batch_size)
            retrieval_losses: List[torch.Tensor] = []
            self_losses: List[torch.Tensor] = []
            for example, query_embedding in zip(batch, query_embeddings):
                candidates = sampler.build_pool(example, batch)
                logits = model.scorer.score_candidate_sets(query_embedding, candidates)
                log_probabilities = torch.log_softmax(logits, dim=0)
                retrieval_losses.append(-log_probabilities[0])

                selected_position = int(logits.detach().argmax().item())
                reward = rewards.get(example, candidates[selected_position], model.func_names)
                self_losses.append(-reward * log_probabilities[selected_position])

            retrieval_loss = torch.stack(retrieval_losses).sum()
            self_loss = torch.stack(self_losses).sum()
            regularisation = model.scorer.regularisation(args.lambda_interaction)
            loss = retrieval_loss + args.eta * self_loss + regularisation

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.scorer.project_constraints_()
            scheduler.step()

            global_step += 1
            running_ret += float(retrieval_loss.item()) / len(batch)
            running_self += float(self_loss.item()) / len(batch)
            if (
                args.reward_cache
                and args.reward_refresh_steps > 0
                and reward_refresh_count < args.reward_refreshes
                and global_step % args.reward_refresh_steps == 0
            ):
                rewards.load()
                reward_refresh_count += 1
            if global_step % 200 == 0:
                print(
                    f"epoch={epoch} step={global_step}/{total_steps} "
                    f"Lret/record={retrieval_loss.item() / len(batch):.4f} "
                    f"Lself/record={self_loss.item() / len(batch):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        recall = validation_recall_at_5(
            model,
            validation,
            k1=args.k1,
            k_pool=args.k_pool,
            encode_batch_size=args.encode_batch_size,
            limit=args.val_limit,
        )
        mean_ret = running_ret / steps_per_epoch
        mean_self = running_self / steps_per_epoch
        print(
            f"epoch={epoch} mean_Lret={mean_ret:.4f} mean_Lself={mean_self:.4f} "
            f"validation_Recall@5={recall:.6f}"
        )

        model.save(
            str(checkpoint_dir / "last.pt"),
            extra={"epoch": epoch, "validation_recall_at_5": recall, "train_config": safe_config},
        )
        if recall > best_recall:
            best_recall = recall
            stale_epochs = 0
            model.save(
                str(checkpoint_dir / "best.pt"),
                extra={"epoch": epoch, "validation_recall_at_5": recall, "train_config": safe_config},
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs (patience={args.patience}).")
                break

    print(f"Best validation Recall@5: {best_recall:.6f}")
    print(f"Checkpoint: {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
