from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METRICS = (
    "Recall@3",
    "Recall@5",
    "NDCG@3",
    "NDCG@5",
    "COMP@3",
    "COMP@5",
    "PredictedSetExactMatch",
    "MeanPredictedCardinality",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize HYSET seed runs")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs = []
    for filename in args.inputs:
        with open(filename, encoding="utf-8") as stream:
            runs.append(json.load(stream))
    if not runs:
        raise RuntimeError("no result summaries were provided")

    summary = {"runs": len(runs), "metrics": {}}
    for metric in METRICS:
        values = [float(run[metric]) for run in runs]
        summary["metrics"][metric] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    for metric, values in summary["metrics"].items():
        print(f"{metric}: {values['mean']:.6f} +/- {values['sample_std']:.6f}")


if __name__ == "__main__":
    main()
