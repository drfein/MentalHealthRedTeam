from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROMPT_VERSION = "observed_all_user_turns_last10_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_id(conversation_id: str, user_index: int) -> str:
    value = f"{PROMPT_VERSION}\0{conversation_id}\0{user_index}"
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build user-endorses-delusion inputs aligned to all scored assistant turns."
    )
    parser.add_argument(
        "--assistant-responses",
        type=Path,
        default=Path("results/observed_all_turn_trajectories/responses.jsonl"),
    )
    parser.add_argument(
        "--assistant-prompts",
        type=Path,
        default=Path("results/observed_all_turn_trajectories/prompts.jsonl"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/observed_all_turn_user_trajectories")
    )
    args = parser.parse_args()

    responses = read_jsonl(args.assistant_responses)
    prompts = {row["generation_id"]: row for row in read_jsonl(args.assistant_prompts)}
    inputs: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    for response in responses:
        assistant_prompt = prompts[response["generation_id"]]
        messages = assistant_prompt["messages"]
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("Assistant prompt does not end with its preceding user turn.")
        user_index = int(response["assistant_message_index"]) - 1
        generation_id = stable_id(str(response["conversation_id"]), user_index)
        base = {
            "generation_id": generation_id,
            "conversation_id": response["conversation_id"],
            "canonical_row_idx": int(response["canonical_row_idx"]),
            "user_message_index": user_index,
            "assistant_message_index": int(response["assistant_message_index"]),
            "assistant_turn_index": int(response["assistant_turn_index"]),
            "assistant_turn_count": int(response["assistant_turn_count"]),
            "conversation_progress": float(response["conversation_progress"]),
            "is_retained_target": bool(response["follows_retained_target"]),
            "retained_target_ordinal": response["retained_target_ordinal"],
            "user_text": messages[-1]["content"],
            "prompt_version": PROMPT_VERSION,
        }
        inputs.append(base)
        context_rows.append({**base, "messages": messages[:-1]})

    if len(inputs) != len({row["generation_id"] for row in inputs}):
        raise ValueError("Duplicate user-turn generation IDs.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("inputs.jsonl", inputs), ("prompts.jsonl", context_rows)]:
        with (args.out_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "user_turns": len(inputs),
        "conversations": len({row["conversation_id"] for row in inputs}),
        "retained_targets": sum(row["is_retained_target"] for row in inputs),
        "annotation_id": "user-endorses-delusion",
        "context": "Up to nine messages preceding the scored user turn; the aligned assistant analysis sees these messages plus the user turn.",
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
