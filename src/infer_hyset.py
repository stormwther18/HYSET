from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable

import torch

from hyset_model import HYSETModel


def input_records(query: str, input_path: str) -> Iterable[dict]:
    if query:
        yield {"query_id": "query-0", "query": query}
        return
    stream = open(input_path, encoding="utf-8") if input_path else sys.stdin
    try:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("query"), str) or not record["query"].strip():
                raise ValueError(f"line {line_number} must contain a non-empty string query")
            record.setdefault("query_id", f"query-{line_number}")
            yield record
    finally:
        if input_path:
            stream.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HYSET inference and reranking")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default=os.getenv("HYSET_ENCODER_PATH", ""))
    parser.add_argument("--query", default="")
    parser.add_argument("--input", default="", help="JSONL input; stdin is used when omitted")
    parser.add_argument("--output", default="", help="JSONL output; stdout is used when omitted")
    parser.add_argument("--k1", type=int, default=15)
    parser.add_argument("--k_pool", type=int, default=20)
    parser.add_argument("--ranking_length", type=int, default=5)
    parser.add_argument("--candidate_batch_size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.encoder:
        parser.error("--encoder or HYSET_ENCODER_PATH is required")
    if args.query and args.input:
        parser.error("use either --query or --input, not both")
    return args


def main() -> None:
    args = parse_args()
    model = HYSETModel.load(args.checkpoint, args.encoder, device=args.device)
    output = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for record in input_records(args.query, args.input):
            result = model.retrieve(
                record["query"],
                k1=args.k1,
                k_pool=args.k_pool,
                ranking_length=args.ranking_length,
                candidate_batch_size=args.candidate_batch_size,
            )
            payload = {
                "query_id": str(record["query_id"]),
                "query": record["query"],
                **result.as_dict(),
            }
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")
    finally:
        if args.output:
            output.close()


if __name__ == "__main__":
    main()
