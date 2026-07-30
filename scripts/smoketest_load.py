from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hyset_model import HYSETModel


DEFAULT_QUERY = (
    "Find round-trip flights, a hotel, the weather forecast, and a currency "
    "conversion for a five-day trip."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default=os.getenv("HYSET_ENCODER_PATH", ""))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    args = parser.parse_args()
    if not args.encoder:
        parser.error("--encoder or HYSET_ENCODER_PATH is required")

    model = HYSETModel.load(args.checkpoint, args.encoder, device=args.device)
    result = model.retrieve(args.query)
    print(
        json.dumps(
            {
                "num_tools": model.num_tools,
                "d_z": model.d_z,
                "m_max": model.m_max,
                **result.as_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
