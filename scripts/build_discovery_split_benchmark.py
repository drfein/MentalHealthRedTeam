from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a response-generation input restricted to one discovery split."
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("data/releases/wilddelusion_combined/train.parquet"),
    )
    parser.add_argument("--discovery-split", default="legacy_probe_gpt52")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/discovery_route_benchmark/historical.parquet"),
    )
    args = parser.parse_args()

    release = pd.read_parquet(args.release)
    frame = release[release["discovery_split"] == args.discovery_split].copy()
    if frame.empty:
        raise ValueError(f"No rows found for discovery split {args.discovery_split!r}.")
    if frame["message_hash"].duplicated().any():
        raise ValueError("Discovery-split message hashes must be unique.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    manifest = {
        "release": str(args.release),
        "output": str(args.output),
        "discovery_split": args.discovery_split,
        "rows": len(frame),
        "conversations": int(frame["conversation_id"].nunique()),
        "sources": frame["source"].value_counts().sort_index().to_dict(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
