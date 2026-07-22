from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_attempts(path: Path) -> dict[str, dict[str, Any]]:
    attempts = {}
    for row in read_jsonl(path):
        attempts[str(row["generation_id"])] = row
    return attempts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map assistant-history intervention generations to SPIRALS judge inputs."
    )
    parser.add_argument(
        "--interventions",
        type=Path,
        default=Path("data/assistant_history_substitution/intervention_arms.parquet"),
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("data/assistant_history_substitution/downstream_responses_k32.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/assistant_history_substitution/judge_inputs.jsonl"),
    )
    args = parser.parse_args()

    interventions = pd.read_parquet(args.interventions)
    by_hash = {
        str(row.message_hash): row for row in interventions.itertuples(index=False)
    }
    responses = latest_attempts(args.responses)
    output_rows = []
    errors = 0
    for response in responses.values():
        if response.get("error_type") or not str(response.get("response_text") or "").strip():
            errors += 1
            continue
        source = by_hash.get(str(response["message_hash"]))
        if source is None:
            raise ValueError(f"Unknown intervention message hash: {response['message_hash']}")
        output_rows.append(
            {
                "original_row_idx": int(source.original_row_idx),
                "intervention_arm": source.context_arm,
                "model_id": response["model"],
                "generation_id": response["generation_id"],
                "source": source.source,
                "conversation_id": source.conversation_id,
                "target_text": source.target_text,
                "response": response["response_text"],
            }
        )
    keys = [
        (row["original_row_idx"], row["intervention_arm"], row["model_id"])
        for row in output_rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Judge inputs are not unique by target, arm, and model.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in sorted(
            output_rows,
            key=lambda item: (
                item["original_row_idx"],
                item["intervention_arm"],
                item["model_id"],
            ),
        ):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "interventions": str(args.interventions),
        "responses": str(args.responses),
        "output": str(args.output),
        "judge_inputs": len(output_rows),
        "generation_errors": errors,
        "targets": len({row["original_row_idx"] for row in output_rows}),
        "models": sorted({row["model_id"] for row in output_rows}),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
