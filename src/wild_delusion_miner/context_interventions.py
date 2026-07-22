from __future__ import annotations

from typing import Any


USER_ROLES = {"human", "user"}
ASSISTANT_ROLES = {"assistant", "llm"}
HISTORY_ARMS = ("original", "assistant_removed", "gpt52_substituted")


def normalize_role(role: Any) -> str | None:
    value = str(role or "").lower()
    if value in USER_ROLES:
        return "user"
    if value in ASSISTANT_ROLES:
        return "assistant"
    return None


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        role = normalize_role(message.get("role"))
        content = str(message.get("content") or "").strip()
        if role and content:
            normalized.append({"role": role, "content": content})
    return normalized


def direct_preceding_assistant_index(row: dict[str, Any]) -> int | None:
    messages = normalize_messages(list(row["messages"]))
    target_index = int(row["target_message_index"])
    if target_index >= len(messages):
        raise ValueError("Role normalization changed target indexing.")
    if messages[target_index]["role"] != "user":
        raise ValueError("The retained target must be a user message.")
    candidate = target_index - 1
    if candidate < 1 or messages[candidate]["role"] != "assistant":
        return None
    if messages[candidate - 1]["role"] != "user":
        return None
    return candidate


def build_replacement_input(
    row: dict[str, Any],
    original_row_idx: int,
    *,
    max_prior_messages: int | None = None,
) -> dict[str, Any] | None:
    assistant_index = direct_preceding_assistant_index(row)
    if assistant_index is None:
        return None
    messages = normalize_messages(list(row["messages"]))
    prior_user_index = assistant_index - 1
    replacement_messages = messages[: prior_user_index + 1]
    if max_prior_messages is not None:
        replacement_messages = replacement_messages[-max_prior_messages:]
    suffix = (
        "gpt52_replacement"
        if max_prior_messages is None
        else f"gpt52_replacement_k{max_prior_messages}"
    )
    output = dict(row)
    output.update(
        {
            "original_row_idx": original_row_idx,
            "messages": replacement_messages,
            "target_message_index": len(replacement_messages) - 1,
            "target_text": messages[prior_user_index]["content"],
            "original_target_text": messages[int(row["target_message_index"])]["content"],
            "original_assistant_index": assistant_index,
            "original_assistant_text": messages[assistant_index]["content"],
            "message_hash": f"{row['message_hash']}:{suffix}",
        }
    )
    return output


def build_history_arms(
    row: dict[str, Any],
    original_row_idx: int,
    replacement_text: str,
    replacement_generation_id: str,
    *,
    max_prior_messages: int | None = None,
) -> list[dict[str, Any]]:
    assistant_index = direct_preceding_assistant_index(row)
    if assistant_index is None:
        raise ValueError("Target does not have a directly preceding assistant reply.")
    messages = normalize_messages(list(row["messages"]))
    target_index = int(row["target_message_index"])
    original_assistant = messages[assistant_index]["content"]
    base = {
        **dict(row),
        "original_row_idx": original_row_idx,
        "original_message_hash": str(row["message_hash"]),
        "original_assistant_index": assistant_index,
        "original_assistant_text": original_assistant,
        "replacement_assistant_text": replacement_text,
        "replacement_generation_id": replacement_generation_id,
    }

    window_start = 0
    if max_prior_messages is not None:
        window_start = max(0, target_index - max_prior_messages)
    prior_window = messages[window_start:target_index]
    if prior_window[-1]["role"] != "assistant":
        raise ValueError("The context window must end with the replaced assistant reply.")
    arm_messages = {
        "original": prior_window + [messages[target_index]],
        "assistant_removed": prior_window[:-1] + [messages[target_index]],
        "gpt52_substituted": (
            prior_window[:-1]
            + [{"role": "assistant", "content": replacement_text}]
            + [messages[target_index]]
        ),
    }
    window_suffix = "full" if max_prior_messages is None else f"k{max_prior_messages}"
    rows = []
    for arm in HISTORY_ARMS:
        arm_row = dict(base)
        arm_row.update(
            {
                "context_arm": arm,
                "messages": arm_messages[arm],
                "target_message_index": len(arm_messages[arm]) - 1,
                "target_text": messages[target_index]["content"],
                "visible_prior_messages": len(arm_messages[arm]) - 1,
                "message_hash": f"{row['message_hash']}:{arm}:{window_suffix}",
            }
        )
        rows.append(arm_row)
    return rows
