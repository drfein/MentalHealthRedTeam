#!/usr/bin/env python3
"""Build the redistributable WildDelusion release from authoritative rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EXCLUDED_SOURCES = ("lmsys_chat_1m",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/verification/whitened_top3k_verified_gpt54mini.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/releases/WildDelusionVerified"),
    )
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=list(DEFAULT_EXCLUDED_SOURCES),
        help="Source whose content license prohibits republication. Repeat as needed.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def export_projection(row: dict[str, Any]) -> dict[str, Any]:
    projected = {
        field: value
        for field, value in row.items()
        if field not in {"stable_key", "raw_keys"}
    }
    target_index = int(projected["target_message_index"])
    projected.setdefault("target_text", projected["messages"][target_index]["content"])
    return projected


def conversation_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["source"]), str(row["conversation_id"])


def main() -> None:
    args = parse_args()
    excluded = set(args.exclude_source)
    positives = [row for row in read_jsonl(args.input) if row.get("judge_label") == "positive"]
    release = [export_projection(row) for row in positives if row["source"] not in excluded]
    omitted = [row for row in positives if row["source"] in excluded]
    if not release or not omitted:
        raise ValueError("Expected both redistributable and license-excluded positive rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "train.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in release:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(release).to_parquet(args.output_dir / "train.parquet", index=False)

    manifest = {
        "input": str(args.input),
        "release_rule": "judge_label == positive and source not license-excluded",
        "rows": len(release),
        "distinct_source_conversations": len({conversation_key(row) for row in release}),
        "source_target_turn_counts": dict(sorted(Counter(row["source"] for row in release).items())),
        "excluded_sources": sorted(excluded),
        "excluded_positive_rows": len(omitted),
        "exclusion_reason": {
            "lmsys_chat_1m": "LMSYS-Chat-1M license prohibits transfer or hosting of conversation data"
        },
        "excluded_rows_remain_in_private_pipeline_artifacts": True,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
