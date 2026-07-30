from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyset_model import func_name_of


def api_text(api: dict) -> str:
    fields = (
        api.get("category_name", ""),
        api.get("tool_name", ""),
        api.get("api_name", ""),
        api.get("api_description", ""),
    )
    return " ".join(str(value) for value in fields if value).strip()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build HYSET's ToolBench corpus")
    parser.add_argument("--instruction_dir", default=str(root / "data" / "instruction"))
    parser.add_argument("--output", default=str(root / "data" / "hyset_corpus.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instruction_dir = Path(args.instruction_dir)
    corpus: dict = {}
    for filename in ("G1_query.json", "G2_query.json", "G3_query.json"):
        path = instruction_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing official instruction file: {path}")
        with path.open(encoding="utf-8") as stream:
            queries = json.load(stream)
        for item in queries:
            for api in item.get("api_list", []):
                tool_name = api.get("tool_name", "")
                api_name = api.get("api_name", "")
                if not tool_name or not api_name:
                    continue
                function_name = func_name_of(str(tool_name), str(api_name))
                if function_name not in corpus:
                    corpus[function_name] = {
                        "idx": len(corpus),
                        "category": api.get("category_name", ""),
                        "tool_name": tool_name,
                        "api_name": api_name,
                        "text": api_text(api),
                        "full_api_json": api,
                    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(corpus, stream, indent=2, ensure_ascii=False)
    print(f"Saved {len(corpus)} API endpoints to {output}")


if __name__ == "__main__":
    main()
