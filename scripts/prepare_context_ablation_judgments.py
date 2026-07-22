from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map target-only generations to release rows for package judging."
    )
    parser.add_argument(
        "--release", type=Path, default=Path("data/releases/WildDelusionVerified/train.parquet")
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("data/context_ablation/target_only_responses.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/context_ablation/target_only_judge_inputs.jsonl"),
    )
    args = parser.parse_args()

    release = pd.read_parquet(args.release).reset_index(names="original_row_idx")
    release_index = {
        str(row.message_hash): row for row in release.itertuples(index=False)
    }
    response_attempts = read_jsonl(args.responses)
    by_generation: dict[str, list[dict[str, Any]]] = {}
    for response in response_attempts:
        by_generation.setdefault(str(response["generation_id"]), []).append(response)
    responses = []
    for attempts in by_generation.values():
        successes = [
            row
            for row in attempts
            if not row.get("error_type") and str(row.get("response_text") or "").strip()
        ]
        responses.append(successes[-1] if successes else attempts[-1])
    output_rows = []
    errors = 0
    missing = 0
    for response in responses:
        if response.get("error_type") or not str(response.get("response_text") or "").strip():
            errors += 1
            continue
        suffixed_hash = str(response["message_hash"])
        if not suffixed_hash.endswith(":target_only"):
            raise ValueError(f"Unexpected target-only message hash: {suffixed_hash}")
        message_hash = suffixed_hash.removesuffix(":target_only")
        release_row = release_index.get(message_hash)
        if release_row is None:
            missing += 1
            continue
        output_rows.append(
            {
                "original_row_idx": int(release_row.original_row_idx),
                "intervention_arm": "target_only",
                "model_id": response["model"],
                "generation_id": response["generation_id"],
                "source": release_row.source,
                "conversation_id": release_row.conversation_id,
                "message_hash": message_hash,
                "target_message_index": int(release_row.target_message_index),
                "preceding_messages": int(release_row.target_message_index),
                "target_text": release_row.target_text,
                "response": response["response_text"],
            }
        )

    keys = [(row["original_row_idx"], row["model_id"]) for row in output_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Target-only generations are not unique by release row and model.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in sorted(output_rows, key=lambda item: (item["original_row_idx"], item["model_id"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "release_rows": len(release),
        "response_attempts": len(response_attempts),
        "unique_generation_ids": len(responses),
        "judge_inputs": len(output_rows),
        "generation_errors": errors,
        "missing_release_rows": missing,
        "models": sorted({row["model_id"] for row in output_rows}),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
