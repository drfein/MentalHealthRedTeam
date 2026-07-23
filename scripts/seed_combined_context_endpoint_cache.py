from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from wild_delusion_miner.assistant_responses import generation_id
from wild_delusion_miner.text import stable_json


ENDPOINT_PATTERN = re.compile(r":prior_(?:0|all)$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_messages(messages: list[dict[str, Any]]) -> str:
    return stable_json(
        [
            {
                "role": str(message["role"]),
                "content": str(message["content"]),
            }
            for message in messages
        ]
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed combined endpoint caches from the completed six-arm run."
    )
    parser.add_argument(
        "--source-responses",
        type=Path,
        default=Path("data/context_dose_response/responses.jsonl"),
    )
    parser.add_argument(
        "--source-judgments",
        type=Path,
        default=Path("results/context_dose_response/package_judgments.jsonl"),
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path("data/combined_context_endpoints/inputs.parquet"),
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("data/releases/WildDelusionCombined/train.parquet"),
    )
    parser.add_argument(
        "--output-responses",
        type=Path,
        default=Path("data/combined_context_endpoints/responses.jsonl"),
    )
    parser.add_argument(
        "--output-judgments",
        type=Path,
        default=Path("results/combined_context_endpoints/package_judgments.jsonl"),
    )
    args = parser.parse_args()

    inputs = pd.read_parquet(args.inputs)
    input_by_hash = {
        str(row["message_hash"]): row for row in inputs.to_dict(orient="records")
    }
    release = pd.read_parquet(args.release).reset_index(names="original_row_idx")
    release_by_hash = {
        str(row["message_hash"]): row for row in release.to_dict(orient="records")
    }

    source_responses = read_jsonl(args.source_responses)
    response_key_by_old_id: dict[str, tuple[str, str]] = {}
    responses_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in source_responses:
        message_hash = str(row.get("message_hash", ""))
        if not ENDPOINT_PATTERN.search(message_hash):
            continue
        if row.get("error_type") or not str(row.get("response_text") or "").strip():
            continue
        model = str(row["model"])
        prompt = input_by_hash.get(message_hash)
        if prompt is None:
            continue
        if canonical_messages(row["input_messages"]) != canonical_messages(prompt["messages"]):
            raise ValueError(f"Cached prompt does not match combined input: {message_hash}")
        response_key_by_old_id[str(row["generation_id"])] = (message_hash, model)
        remapped = dict(row)
        remapped["generation_id"] = generation_id(
            prompt, model, str(row["prompt_version"])
        )
        responses_by_key[(message_hash, model)] = remapped
    responses = list(responses_by_key.values())

    judgments_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(args.source_judgments):
        if row.get("judge_error"):
            continue
        key = response_key_by_old_id.get(str(row.get("generation_id", "")))
        if key is None:
            continue
        message_hash, model = key
        prompt = input_by_hash[message_hash]
        arm_match = ENDPOINT_PATTERN.search(message_hash)
        assert arm_match is not None
        original_hash = message_hash[: arm_match.start()]
        source = release_by_hash[original_hash]
        remapped = dict(row)
        remapped.update(
            {
                "generation_id": generation_id(
                    prompt, model, str(responses_by_key[key]["prompt_version"])
                ),
                "original_row_idx": int(source["original_row_idx"]),
                "intervention_arm": str(prompt["context_arm"]),
                "model_id": model,
                "source": str(source["source"]),
                "conversation_id": str(source["conversation_id"]),
                "discovery_split": str(source["discovery_split"]),
                "message_hash": original_hash,
            }
        )
        judgments_by_key[key] = remapped
    judgments = list(judgments_by_key.values())

    write_jsonl(args.output_responses, responses)
    write_jsonl(args.output_judgments, judgments)
    summary = {
        "seeded_responses": len(responses),
        "seeded_judgments": len(judgments),
        "source_responses": str(args.source_responses),
        "source_judgments": str(args.source_judgments),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
