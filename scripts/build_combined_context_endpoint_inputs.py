from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from wild_delusion_miner.context_dose import build_context_arms


ENDPOINT_ARMS = {"prior_0", "prior_all"}


def build_endpoint_rows(release: pd.DataFrame) -> pd.DataFrame:
    eligible = release[release["target_message_index"].astype(int) >= 1].copy()
    rows = [
        arm
        for row in eligible.to_dict(orient="records")
        for arm in build_context_arms(row, prior_counts=(0,))
        if arm["context_arm"] in ENDPOINT_ARMS
    ]
    output = pd.DataFrame(rows)
    key = ["message_hash"]
    if output.duplicated(key).any():
        raise ValueError("Endpoint message hashes must be unique.")
    counts = output.groupby("context_arm").size()
    if set(counts.index) != ENDPOINT_ARMS or counts.nunique() != 1:
        raise ValueError("Each eligible target must have both endpoint arms.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build target-only and full-context inputs for the combined release."
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("data/releases/WildDelusionCombined/train.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/combined_context_endpoints/inputs.parquet"),
    )
    args = parser.parse_args()

    release = pd.read_parquet(args.release)
    required = {
        "source",
        "conversation_id",
        "messages",
        "target_message_index",
        "message_hash",
        "discovery_split",
    }
    missing = required - set(release)
    if missing:
        raise ValueError(f"Combined release is missing columns: {sorted(missing)}")

    output = build_endpoint_rows(release)
    eligible = output[output["context_arm"] == "prior_0"]
    route_counts = {}
    for route, group in eligible.groupby("discovery_split"):
        route_counts[str(route)] = {
            "targets": int(len(group)),
            "conversations": int(
                group[["source", "conversation_id"]].drop_duplicates().shape[0]
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    manifest: dict[str, Any] = {
        "release": str(args.release),
        "output": str(args.output),
        "release_targets": int(len(release)),
        "eligible_targets": int(len(eligible)),
        "ineligible_zero_prior_targets": int(
            (release["target_message_index"].astype(int) == 0).sum()
        ),
        "source_conversations": int(
            eligible[["source", "conversation_id"]].drop_duplicates().shape[0]
        ),
        "rows": int(len(output)),
        "arms": ["prior_0", "prior_all"],
        "by_discovery_split": route_counts,
        "counting_rule": "Both user and assistant source messages count.",
        "system_prompt": None,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
