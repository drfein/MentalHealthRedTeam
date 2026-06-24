from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ConversationRecord:
    source: str
    split: str
    row_offset: int
    conversation_id: str
    messages: tuple[ChatMessage, ...]
    raw_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class UserMessageRecord:
    message_hash: str
    text: str
    source: str
    split: str
    row_offset: int
    conversation_id: str
    message_index: int


JsonDict = dict[str, Any]
