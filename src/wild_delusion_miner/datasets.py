from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from datasets import load_dataset

from wild_delusion_miner.config import DatasetSpec
from wild_delusion_miner.records import ChatMessage, ConversationRecord, UserMessageRecord
from wild_delusion_miner.text import dedupe_key, normalize_text, stable_json, stringify

ROLE_ALIASES = {
    "human": "user",
    "prompter": "user",
    "customer": "user",
    "client": "user",
    "questioner": "user",
    "user": "user",
    "assistant": "assistant",
    "bot": "assistant",
    "chatbot": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "ai": "assistant",
}

ROLE_KEYS = ("role", "from", "speaker", "author", "sender")
CONTENT_KEYS = ("content", "value", "text", "message", "body", "utterance")
CONVERSATION_KEYS = (
    "conversation",
    "conversations",
    "messages",
    "turns",
    "chat",
    "dialogue",
    "dialog",
)
ID_KEYS = (
    "conversation_id",
    "conversation_hash",
    "conv_id",
    "chat_id",
    "id",
    "share_id",
    "url",
)


def normalize_role(value: Any) -> str | None:
    role = stringify(value).strip().lower()
    return ROLE_ALIASES.get(role)


def _content_from_mapping(item: Mapping[str, Any]) -> str:
    for key in CONTENT_KEYS:
        if key in item:
            text = normalize_text(stringify(item[key]))
            if text:
                return text
    return ""


def _message_from_mapping(item: Mapping[str, Any]) -> ChatMessage | None:
    role = None
    for key in ROLE_KEYS:
        if key in item:
            role = normalize_role(item[key])
            if role:
                break
    if role is None:
        return None
    content = _content_from_mapping(item)
    if not content:
        return None
    return ChatMessage(role=role, content=content)


def _parse_json_if_needed(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _messages_from_sequence(value: Sequence[Any]) -> tuple[ChatMessage, ...]:
    messages: list[ChatMessage] = []
    for item in value:
        item = _parse_json_if_needed(item)
        if isinstance(item, Mapping):
            message = _message_from_mapping(item)
            if message:
                messages.append(message)
    return tuple(messages)


def _find_message_sequence(value: Any, *, depth: int = 0) -> tuple[ChatMessage, ...]:
    if depth > 4:
        return ()
    value = _parse_json_if_needed(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        direct = _messages_from_sequence(value)
        if direct:
            return direct
        for item in value:
            nested = _find_message_sequence(item, depth=depth + 1)
            if nested:
                return nested
    if isinstance(value, Mapping):
        for key in CONVERSATION_KEYS:
            if key in value:
                nested = _find_message_sequence(value[key], depth=depth + 1)
                if nested:
                    return nested
        for nested_value in value.values():
            nested = _find_message_sequence(nested_value, depth=depth + 1)
            if nested:
                return nested
    return ()


def _conversation_id(row: Mapping[str, Any], source: str, split: str, row_offset: int) -> str:
    for key in ID_KEYS:
        value = row.get(key)
        text = normalize_text(stringify(value))
        if text:
            return text
    return f"{source}:{split}:{row_offset}"


def row_to_conversation(
    row: Mapping[str, Any],
    *,
    source: str,
    split: str,
    row_offset: int,
) -> ConversationRecord | None:
    messages = _find_message_sequence(row)
    if not messages:
        return None
    return ConversationRecord(
        source=source,
        split=split,
        row_offset=row_offset,
        conversation_id=_conversation_id(row, source, split, row_offset),
        messages=messages,
        raw_keys=tuple(sorted(str(key) for key in row.keys())),
    )


def iter_user_messages(conversation: ConversationRecord) -> Iterable[UserMessageRecord]:
    for index, message in enumerate(conversation.messages):
        if message.role != "user":
            continue
        text = normalize_text(message.content)
        if not text:
            continue
        yield UserMessageRecord(
            message_hash=dedupe_key(text),
            text=text,
            source=conversation.source,
            split=conversation.split,
            row_offset=conversation.row_offset,
            conversation_id=conversation.conversation_id,
            message_index=index,
        )


def stream_conversations(
    spec: DatasetSpec,
    *,
    limit_rows: int | None = None,
    start_offset: int = 0,
) -> Iterable[ConversationRecord]:
    token = os.environ.get("HF_TOKEN")
    kwargs: dict[str, Any] = {
        "path": spec.name,
        "split": spec.split,
        "streaming": True,
    }
    if spec.config:
        kwargs["name"] = spec.config
    if token:
        kwargs["token"] = token

    dataset = load_dataset(**kwargs)
    if start_offset > 0:
        dataset = dataset.skip(start_offset)
    for row_offset, row in enumerate(dataset, start=start_offset):
        if limit_rows is not None and row_offset >= limit_rows:
            break
        if not isinstance(row, Mapping):
            continue
        conversation = row_to_conversation(
            row,
            source=spec.source,
            split=spec.split,
            row_offset=row_offset,
        )
        if conversation:
            yield conversation


def conversation_to_json(conversation: ConversationRecord) -> dict[str, Any]:
    return {
        "source": conversation.source,
        "split": conversation.split,
        "row_offset": conversation.row_offset,
        "conversation_id": conversation.conversation_id,
        "raw_keys": list(conversation.raw_keys),
        "messages": [
            {"role": message.role, "content": message.content}
            for message in conversation.messages
        ],
        "stable_key": stable_json(
            [conversation.source, conversation.split, conversation.conversation_id]
        ),
    }
