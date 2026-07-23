from __future__ import annotations

from typing import Any, Iterable


DEFAULT_PRIOR_COUNTS = (0, 1, 2, 4, 8)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        role = str(message.get("role", "")).lower()
        if role in {"human", "user"}:
            role = "user"
        elif role in {"assistant", "llm"}:
            role = "assistant"
        else:
            continue
        content = str(message.get("content", "") or "").strip()
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def build_context_arms(
    row: dict[str, Any],
    *,
    prior_counts: Iterable[int] = DEFAULT_PRIOR_COUNTS,
    include_all: bool = True,
) -> list[dict[str, Any]]:
    messages = normalize_messages(list(row["messages"]))
    target_index = int(row["target_message_index"])
    if target_index >= len(messages):
        raise ValueError("Role normalization changed target indexing.")
    if messages[target_index]["role"] != "user":
        raise ValueError("The retained target must be a user message.")

    arms = []
    for prior_count in prior_counts:
        start = target_index - prior_count
        arms.append((f"prior_{prior_count}", messages[start : target_index + 1]))
    if include_all:
        arms.append(("prior_all", messages[: target_index + 1]))

    output = []
    for arm, visible in arms:
        arm_row = dict(row)
        arm_row.update(
            {
                "context_arm": arm,
                "full_target_message_index": target_index,
                "full_prior_message_count": target_index,
                "visible_prior_message_count": len(visible) - 1,
                "messages": visible,
                "target_message_index": len(visible) - 1,
                "message_hash": f"{row['message_hash']}:{arm}",
            }
        )
        output.append(arm_row)
    return output
