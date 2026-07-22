from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build matched target-only inputs for the response context ablation."
    )
    parser.add_argument(
        "--release", type=Path, default=Path("data/releases/WildDelusionVerified/train.parquet")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/context_ablation/target_only.parquet")
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.release).copy()
    required = {"messages", "target_message_index", "target_text", "message_hash"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Release is missing required columns: {sorted(missing)}")
    if frame["message_hash"].duplicated().any():
        raise ValueError("Release message_hash values must be unique.")

    frame["context_arm"] = "target_only"
    frame["full_target_message_index"] = frame["target_message_index"].astype(int)
    frame["full_context_message_count"] = frame.apply(
        lambda row: int(row["target_message_index"]), axis=1
    )
    frame["messages"] = frame["target_text"].map(
        lambda text: [{"role": "user", "content": str(text)}]
    )
    frame["target_message_index"] = 0
    frame["message_hash"] = frame["message_hash"].astype(str) + ":target_only"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    manifest = {
        "release": str(args.release),
        "output": str(args.output),
        "rows": len(frame),
        "arm": "target_only",
        "intervention": (
            "Replace the full conversation prefix with the verified target user turn; "
            "generation system prompt and model parameters are held fixed."
        ),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
