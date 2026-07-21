#!/usr/bin/env python3
"""Verify a published WildDelusion export against the authoritative local rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


KEY_FIELDS = ("source", "message_hash")
FLOAT_FIELDS = ("retrieval_score",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authoritative",
        type=Path,
        default=Path("data/verification/whitened_top3k_verified_gpt54mini.jsonl"),
    )
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument(
        "--release-label",
        default=None,
        help="Stable identifier recorded in the result instead of the local download path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/wilddelusion_release_integrity/verification.json"),
    )
    parser.add_argument("--float-atol", type=float, default=1e-12)
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="Exclude this source from the authoritative comparison. Repeat as needed.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict[str, Any]) -> tuple[str, str]:
    return tuple(str(row[field]) for field in KEY_FIELDS)  # type: ignore[return-value]


def validate_rows(name: str, rows: list[dict[str, Any]]) -> None:
    keys = [key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} contains duplicate source/message_hash keys")
    for index, row in enumerate(rows):
        target_index = int(row["target_message_index"])
        messages = row["messages"]
        if not 0 <= target_index < len(messages):
            raise ValueError(f"{name} row {index} has an invalid target index")
        target = messages[target_index]
        if str(target.get("role", "")).lower() != "user":
            raise ValueError(f"{name} row {index} target is not a user message")
        target_text = row.get("target_text", target.get("content", ""))
        if str(target.get("content", "")) != str(target_text):
            raise ValueError(f"{name} row {index} target text does not match messages")


def export_projection(row: dict[str, Any]) -> dict[str, Any]:
    projected = {key: value for key, value in row.items() if key not in {"stable_key", "raw_keys"}}
    target_index = int(projected["target_message_index"])
    projected.setdefault("target_text", projected["messages"][target_index]["content"])
    return projected


def canonical_digest(rows: list[dict[str, Any]]) -> str:
    canonical = []
    for row in sorted(rows, key=key):
        normalized = dict(row)
        for field in FLOAT_FIELDS:
            normalized[field] = round(float(normalized[field]), 12)
        canonical.append(normalized)
    payload = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    authoritative = [
        row
        for row in read_jsonl(args.authoritative)
        if row.get("judge_label") == "positive"
        and row.get("source") not in set(args.exclude_source)
    ]
    release = read_jsonl(args.release)
    validate_rows("authoritative", authoritative)
    validate_rows("release", release)

    authoritative_map = {key(row): row for row in authoritative}
    release_map = {key(row): row for row in release}
    if set(authoritative_map) != set(release_map):
        raise ValueError(
            "Release key set differs: "
            f"{len(set(authoritative_map) - set(release_map))} local-only and "
            f"{len(set(release_map) - set(authoritative_map))} release-only"
        )

    exact_mismatches: Counter[str] = Counter()
    max_float_error = {field: 0.0 for field in FLOAT_FIELDS}
    for row_key in authoritative_map:
        expected = export_projection(authoritative_map[row_key])
        observed = release_map[row_key]
        if set(expected) != set(observed):
            raise ValueError(f"Schema mismatch for {row_key}")
        for field in observed:
            if field in FLOAT_FIELDS:
                error = abs(float(expected[field]) - float(observed[field]))
                max_float_error[field] = max(max_float_error[field], error)
                if not np.isclose(
                    expected[field], observed[field], rtol=0.0, atol=args.float_atol
                ):
                    raise ValueError(f"Float mismatch in {field} for {row_key}: {error}")
            elif expected[field] != observed[field]:
                exact_mismatches[field] += 1
    if exact_mismatches:
        raise ValueError(f"Exact field mismatches: {dict(exact_mismatches)}")

    result = {
        "passed": True,
        "authoritative": str(args.authoritative),
        "release": args.release_label or str(args.release),
        "rows": len(release),
        "unique_keys": len(release_map),
        "key_fields": list(KEY_FIELDS),
        "source_counts": dict(sorted(Counter(row["source"] for row in release).items())),
        "excluded_authoritative_sources": sorted(set(args.exclude_source)),
        "target_index_errors": 0,
        "target_text_errors": 0,
        "non_user_targets": 0,
        "message_mismatches": 0,
        "exact_field_mismatches": {},
        "float_atol": args.float_atol,
        "max_float_absolute_error": max_float_error,
        "canonical_sha256": canonical_digest(release),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
