from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


USER_ROLES = {"human", "user"}
ASSISTANT_ROLES = {"assistant", "llm"}
PROMPT_VERSION = "observed_all_turns_last10_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_message(message: dict[str, Any]) -> dict[str, str] | None:
    role = str(message.get("role", "")).lower()
    if role in USER_ROLES:
        role = "user"
    elif role in ASSISTANT_ROLES:
        role = "assistant"
    else:
        return None
    content = str(message.get("content", "") or "").strip()
    return {"role": role, "content": content} if content else None


def stable_id(conversation_id: str, assistant_index: int) -> str:
    value = f"{PROMPT_VERSION}\0{conversation_id}\0{assistant_index}"
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build package-judge inputs for every assistant turn in recoverable repeated conversations."
    )
    parser.add_argument(
        "--canonical-conversations",
        type=Path,
        default=Path("data/verification/wilddelusion_combined_conversations.jsonl"),
    )
    parser.add_argument(
        "--observed-replies",
        type=Path,
        default=Path("results/observed_assistant_responses/responses.jsonl"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/observed_all_turn_trajectories")
    )
    parser.add_argument("--context-window", type=int, default=10)
    args = parser.parse_args()

    canonical = read_jsonl(args.canonical_conversations)
    observed = pd.DataFrame(read_jsonl(args.observed_replies))
    counts = observed.groupby("conversation_id").size()
    repeated_ids = set(counts[counts >= 2].index)
    repeated = observed[observed["conversation_id"].isin(repeated_ids)].copy()

    responses: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    for conversation_id, group in repeated.groupby("conversation_id"):
        canonical_indices = set(group["canonical_row_idx"].astype(int))
        if len(canonical_indices) != 1:
            raise ValueError(f"Conversation {conversation_id} maps to multiple canonical rows.")
        canonical_index = canonical_indices.pop()
        messages = list(canonical[canonical_index]["messages"])
        retained_user_indices = set(group["canonical_target_message_index"].astype(int))
        assistant_indices = [
            index
            for index, message in enumerate(messages)
            if str(message.get("role", "")).lower() in ASSISTANT_ROLES
        ]
        eligible: list[tuple[int, list[dict[str, str]]]] = []
        for assistant_index in assistant_indices:
            context = [
                normalized
                for message in messages[max(0, assistant_index - args.context_window) : assistant_index]
                if (normalized := normalized_message(message)) is not None
            ]
            if not context or context[-1]["role"] != "user":
                skip_reasons["assistant_not_immediately_after_user"] += 1
                continue
            response = normalized_message(messages[assistant_index])
            if response is None:
                skip_reasons["empty_assistant_response"] += 1
                continue
            eligible.append((assistant_index, context))

        total_turns = len(eligible)
        for turn_index, (assistant_index, context) in enumerate(eligible):
            preceding_user_index = assistant_index - 1
            follows_retained = preceding_user_index in retained_user_indices
            retained_ordinal = (
                sorted(retained_user_indices).index(preceding_user_index) + 1
                if follows_retained
                else None
            )
            base = {
                "generation_id": stable_id(str(conversation_id), assistant_index),
                "original_row_idx": canonical_index,
                "intervention_arm": "observed_all_turn",
                "conversation_id": str(conversation_id),
                "canonical_row_idx": canonical_index,
                "assistant_message_index": assistant_index,
                "assistant_turn_index": turn_index,
                "assistant_turn_count": total_turns,
                "conversation_progress": turn_index / max(total_turns - 1, 1),
                "follows_retained_target": follows_retained,
                "retained_target_ordinal": retained_ordinal,
                "retained_target_count": len(retained_user_indices),
                "target_text": context[-1]["content"],
                "prompt_version": PROMPT_VERSION,
            }
            responses.append(
                {**base, "response": str(messages[assistant_index].get("content", "")).strip()}
            )
            prompts.append({**base, "messages": context})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("responses.jsonl", responses), ("prompts.jsonl", prompts)]:
        with (args.out_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "canonical_conversations": str(args.canonical_conversations),
        "observed_replies": str(args.observed_replies),
        "recoverable_repeated_target_conversations": int(len(repeated_ids)),
        "scorable_conversations": int(len(set(row["conversation_id"] for row in responses))),
        "assistant_turns": len(responses),
        "retained_target_adjacent_turns": sum(
            bool(row["follows_retained_target"]) for row in responses
        ),
        "context_window_messages": args.context_window,
        "skip_reasons": dict(skip_reasons),
        "unit": (
            "One source assistant reply immediately following a user message. The exact package judge "
            "receives up to the preceding 10 normalized source messages."
        ),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
