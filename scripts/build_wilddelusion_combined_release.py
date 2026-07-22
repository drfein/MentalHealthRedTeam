#!/usr/bin/env python3
"""Combine the primary and independently retrieved legacy verified splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from wild_delusion_miner.legacy_expansion import conversation_fingerprint, target_fingerprint


EXCLUDED_SOURCES = {"lmsys_chat_1m"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary",
        type=Path,
        default=Path("data/releases/WildDelusionVerified/train.jsonl"),
    )
    parser.add_argument(
        "--legacy-verified",
        type=Path,
        default=Path("data/legacy_expansion/context_verified.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/releases/WildDelusionCombined"),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalize_primary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "discovery_split": "openai_embedding_whitened",
        "discovery_model": "text-embedding-3-small",
        "discovery_score": row.get("retrieval_score"),
        "annotation_model": row.get("annotation_model") or "gpt-5.5",
    }


def normalize_legacy(row: dict[str, Any]) -> dict[str, Any]:
    projected = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "text",
            "is_positive",
            "estimated_cost_usd",
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "conversation_fingerprint",
            "target_fingerprint",
        }
    }
    route = str(row.get("legacy_source_route", ""))
    discovery_model = (
        "historical ShareChat probe; GPT-5.2 candidate confirmation"
        if route == "sharechat_probe_gpt52"
        else "meta-llama/Llama-3.1-8B, mean-pooled layer 13 logistic probe"
    )
    return {
        **projected,
        "discovery_split": "legacy_probe_gpt52",
        "discovery_model": discovery_model,
        "discovery_score": row.get("legacy_probe_score"),
        "retrieval_score": row.get("legacy_probe_score"),
    }


def validate(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        if row.get("source") in EXCLUDED_SOURCES or "lmsys" in str(row.get("source", "")).lower():
            raise ValueError(f"LMSYS content at row {index}")
        target_index = int(row["target_message_index"])
        messages = row["messages"]
        if not 0 <= target_index < len(messages):
            raise ValueError(f"Invalid target index at row {index}")
        target = messages[target_index]
        if str(target.get("role", "")).lower() != "user":
            raise ValueError(f"Non-user target at row {index}")
        if str(target.get("content", "")) != str(row["target_text"]):
            raise ValueError(f"Target text mismatch at row {index}")
        if row.get("judge_label") != "positive":
            raise ValueError(f"Non-positive contextual-verifier row at row {index}")


def digest(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        sorted(rows, key=lambda row: (str(row["source"]), str(row["message_hash"]))),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    primary = [normalize_primary(row) for row in read_jsonl(args.primary)]
    legacy = [
        normalize_legacy(row)
        for row in read_jsonl(args.legacy_verified)
        if row.get("judge_label") == "positive"
    ]
    rows = primary + legacy
    validate(rows)

    conversation_keys = {
        (str(row["source"]), str(row["conversation_id"]))
        for row in rows
    }
    conversation_fps = [conversation_fingerprint(row["messages"]) for row in rows]
    target_fps = [target_fingerprint(row["target_text"]) for row in rows]
    if len(conversation_fps) != len(set(zip(conversation_fps, target_fps, strict=True))):
        raise ValueError("Combined release contains an exact conversation/target duplicate")

    rows.sort(
        key=lambda row: (
            str(row["discovery_split"]),
            str(row["source"]),
            str(row["conversation_id"]),
            int(row["target_message_index"]),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "train.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(rows).to_parquet(args.output_dir / "train.parquet", index=False)

    manifest = {
        "dataset": "WildDelusionCombined",
        "unit": "contextualized target turn",
        "release_rule": (
            "message-level package score >= 7 and contextual verifier label == positive; "
            "primary split retains its previously published verified rows"
        ),
        "rows": len(rows),
        "distinct_source_conversations": len(conversation_keys),
        "discovery_split_target_counts": dict(
            sorted(Counter(row["discovery_split"] for row in rows).items())
        ),
        "discovery_split_conversation_counts": {
            split: len(
                {
                    (str(row["source"]), str(row["conversation_id"]))
                    for row in rows
                    if row["discovery_split"] == split
                }
            )
            for split in sorted({row["discovery_split"] for row in rows})
        },
        "source_target_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
        "excluded_sources": sorted(EXCLUDED_SOURCES),
        "lmsys_rows": 0,
        "canonical_sha256": digest(rows),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
