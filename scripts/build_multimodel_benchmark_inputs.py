#!/usr/bin/env python3
"""Recover a balanced multi-model benchmark from cached response assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def jsonable_messages(value: Any, target_index: int) -> list[dict[str, str]]:
    messages = value.tolist() if hasattr(value, "tolist") else list(value)
    output = []
    for message in messages[: target_index + 1]:
        role = str(message.get("role", "")).lower()
        content = str(message.get("content", ""))
        if role in {"user", "assistant", "system"} and content:
            output.append({"role": role, "content": content})
    if not output or output[-1]["role"] != "user":
        raise ValueError("Every benchmark prefix must end at the target user message")
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    release = pd.read_parquet(args.release).reset_index(names="original_row_idx")
    if release["target_text"].duplicated().any():
        raise ValueError("Public release target_text must be unique for exact recovery")
    responses = pd.read_csv(args.responses)
    required = {"generation_id", "model", "target_text", "response_text"}
    if missing := required - set(responses):
        raise ValueError(f"Response assignments are missing columns: {sorted(missing)}")
    responses = responses[responses["target_text"].isin(set(release["target_text"]))].copy()
    if responses.duplicated(["model", "target_text"]).any():
        raise ValueError("Response assignments contain duplicate model-target rows")

    model_targets = {
        model: set(group["target_text"])
        for model, group in responses.groupby("model", sort=True)
    }
    if len(model_targets) < 2:
        raise ValueError("Need at least two response models")
    common_targets = set.intersection(*model_targets.values())
    balanced = responses[responses["target_text"].isin(common_targets)].copy()
    expected = len(common_targets) * len(model_targets)
    if len(balanced) != expected:
        raise ValueError(f"Balanced response matrix has {len(balanced)} rows; expected {expected}")

    release_by_text = release.set_index("target_text").to_dict(orient="index")
    prompt_rows = []
    generation_rows = []
    for row in balanced.sort_values(["model", "target_text"]).itertuples(index=False):
        metadata = release_by_text[str(row.target_text)]
        index = int(metadata["original_row_idx"])
        model = str(row.model)
        shared = {
            "original_row_idx": index,
            "intervention_arm": model,
            "model_id": model,
            "generation_id": str(row.generation_id),
            "source": str(metadata["source"]),
            "conversation_id": str(metadata["conversation_id"]),
            "message_hash": str(metadata["message_hash"]),
            "target_text": str(row.target_text),
        }
        prompt_rows.append(
            {
                **shared,
                "messages": jsonable_messages(
                    metadata["messages"], int(metadata["target_message_index"])
                ),
            }
        )
        generation_rows.append({**shared, "response": str(row.response_text)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "prompts.jsonl", prompt_rows)
    write_jsonl(args.output_dir / "responses.jsonl", generation_rows)
    manifest = {
        "release": str(args.release),
        "responses": str(args.responses),
        "selection": "intersection of public target texts with successful cached responses from every model",
        "public_release_targets": len(release),
        "common_targets": len(common_targets),
        "excluded_incomplete_targets": len(release) - len(common_targets),
        "models": sorted(model_targets),
        "model_count": len(model_targets),
        "response_rows": len(generation_rows),
        "target_mapping": "exact target_text; public release target texts verified unique",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
