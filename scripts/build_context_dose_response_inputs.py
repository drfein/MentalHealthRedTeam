from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from wild_delusion_miner.context_dose import (
    DEFAULT_PRIOR_COUNTS,
    build_context_arms,
    normalize_messages,
)

PRIOR_COUNTS = DEFAULT_PRIOR_COUNTS
build_arms = build_context_arms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a matched 0/1/2/4/8/all source-message context sweep."
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("data/releases/WildDelusionVerified/train.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/context_dose_response/inputs.parquet"),
    )
    parser.add_argument("--minimum-prior-messages", type=int, default=8)
    args = parser.parse_args()

    release = pd.read_parquet(args.release).copy()
    release = release[
        release["target_message_index"].astype(int) >= args.minimum_prior_messages
    ].copy()
    rows = [
        arm
        for row in release.to_dict(orient="records")
        for arm in build_arms(row)
    ]
    output = pd.DataFrame(rows)
    counts = output.groupby("message_hash")["context_arm"].nunique()
    if not counts.eq(1).all():
        raise ValueError("Dose-response message hashes must be unique.")
    arm_counts = output.groupby("context_arm").size()
    if arm_counts.nunique() != 1:
        raise ValueError("Every context arm must contain the same targets.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    manifest = {
        "release": str(args.release),
        "output": str(args.output),
        "targets": len(release),
        "conversations": int(release["conversation_id"].nunique()),
        "rows": len(output),
        "arms": [f"prior_{count}" for count in PRIOR_COUNTS] + ["prior_all"],
        "minimum_prior_messages": args.minimum_prior_messages,
        "counting_rule": "Both user and assistant source messages count as prior messages.",
        "system_prompt": None,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
