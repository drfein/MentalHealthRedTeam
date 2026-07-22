from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ARM_PATTERN = re.compile(r":(last_\d+)$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map context-truncation generations to release rows for package judging."
    )
    parser.add_argument(
        "--release", type=Path, default=Path("data/releases/WildDelusionVerified/train.parquet")
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("data/context_ablation/truncation_sweep_responses.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/context_ablation/truncation_sweep_judge_inputs.jsonl"),
    )
    args = parser.parse_args()

    release = pd.read_parquet(args.release).reset_index(names="original_row_idx")
    release_index = {str(row.message_hash): row for row in release.itertuples(index=False)}
    attempts = read_jsonl(args.responses)
    by_generation: dict[str, list[dict[str, Any]]] = {}
    for response in attempts:
        by_generation.setdefault(str(response["generation_id"]), []).append(response)

    rows = []
    errors = 0
    for generation_attempts in by_generation.values():
        successes = [
            row
            for row in generation_attempts
            if not row.get("error_type") and str(row.get("response_text") or "").strip()
        ]
        response = successes[-1] if successes else generation_attempts[-1]
        if not successes:
            errors += 1
            continue
        match = ARM_PATTERN.search(str(response["message_hash"]))
        if match is None:
            raise ValueError(f"Unexpected truncation hash: {response['message_hash']}")
        arm = match.group(1)
        message_hash = str(response["message_hash"])[: match.start()]
        release_row = release_index.get(message_hash)
        if release_row is None:
            raise ValueError(f"Release row not found for {message_hash}")
        input_messages = [
            message
            for message in list(response["input_messages"])
            if str(message.get("role", "")).lower() != "system"
        ]
        rows.append(
            {
                "original_row_idx": int(release_row.original_row_idx),
                "intervention_arm": arm,
                "model_id": response["model"],
                "generation_id": response["generation_id"],
                "source": release_row.source,
                "conversation_id": release_row.conversation_id,
                "message_hash": message_hash,
                "target_message_index": int(release_row.target_message_index),
                "preceding_messages": int(release_row.target_message_index),
                "visible_prior_messages": len(input_messages) - 1,
                "target_text": release_row.target_text,
                "response": response["response_text"],
                "judge_messages": input_messages,
            }
        )

    keys = [(row["original_row_idx"], row["model_id"], row["intervention_arm"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Truncation generations are not unique by row, model, and arm.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompt_path = args.output.with_name(args.output.stem + "_prompts.jsonl")
    with args.output.open("w", encoding="utf-8") as output, prompt_path.open(
        "w", encoding="utf-8"
    ) as prompts:
        for row in sorted(
            rows,
            key=lambda item: (
                item["original_row_idx"],
                item["model_id"],
                item["intervention_arm"],
            ),
        ):
            output.write(
                json.dumps({key: value for key, value in row.items() if key != "judge_messages"})
                + "\n"
            )
            prompts.write(
                json.dumps(
                    {
                        **{key: value for key, value in row.items() if key != "judge_messages"},
                        "messages": row["judge_messages"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = {
        "response_attempts": len(attempts),
        "unique_generation_ids": len(by_generation),
        "judge_inputs": len(rows),
        "generation_errors": errors,
        "models": sorted({row["model_id"] for row in rows}),
        "arms": sorted({row["intervention_arm"] for row in rows}),
        "prompts": str(prompt_path),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
