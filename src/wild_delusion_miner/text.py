from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def dedupe_key(text: str) -> str:
    normalized = normalize_text(text).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
