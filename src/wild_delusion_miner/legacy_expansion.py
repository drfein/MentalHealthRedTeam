from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any


LEGACY_VERIFIED_REPO = "danielfein/WildDelusionVerified"
LEGACY_VERIFIED_REVISION = "59a9eacfdeeeae1e5b02bd2e0bcd1c51776d1a23"
LEGACY_COMBINED_REPO = "danielfein/WildDelusion"
LEGACY_COMBINED_REVISION = "8e8a2f74d3aea6b406a9b2c56c26ca26af40856b"


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def canonical_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "role": str(message.get("role", "")).strip().lower(),
            "content": re.sub(r"\s+", " ", str(message.get("content", ""))).strip(),
        }
        for message in messages
    ]


def conversation_fingerprint(messages: Iterable[dict[str, Any]]) -> str:
    payload = json.dumps(
        canonical_messages(messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def target_fingerprint(text: Any) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def message_hash(source: str, conversation_id: str, target_index: int, text: str) -> str:
    payload = f"{source}\0{conversation_id}\0{target_index}\0{normalize_text(text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_provenance(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def full_conversation(combined: dict[str, Any], verified: dict[str, Any]) -> tuple[list[dict[str, str]], int]:
    raw_messages = combined.get("full_conversation")
    if isinstance(raw_messages, str) and raw_messages.strip():
        raw_messages = json.loads(raw_messages)
    if isinstance(raw_messages, (list, tuple)):
        messages = canonical_messages(raw_messages)
        target = normalize_text(combined.get("flagged_text") or verified.get("target_text"))
        matches = [
            index
            for index, message in enumerate(messages)
            if message["role"] == "user"
            and (
                normalize_text(message["content"]) == target
                or normalize_text(message["content"]).startswith(target)
                or target.startswith(normalize_text(message["content"]))
            )
        ]
        if matches:
            return messages, min(
                matches,
                key=lambda index: abs(len(normalize_text(messages[index]["content"])) - len(target)),
            )
    return canonical_messages(verified["messages"]), int(verified["target_message_index"])


def release_source(route: str, provenance: dict[str, Any]) -> str | None:
    if route == "lmsys_probe_gpt52" or provenance.get("source_corpus") == "lmsys":
        return None
    if provenance.get("source_corpus") == "sharechat":
        config = str(provenance.get("config", provenance.get("platform", ""))).lower()
        if not config:
            raise ValueError("ShareChat provenance is missing its config/platform")
        return f"sharechat_{config}"
    if provenance.get("source_corpus") == "existing_wilddelusion_combined":
        return "wildchat_full"
    if route in {"probe", "both", "verified"}:
        return "wildchat_full"
    raise ValueError(f"Unsupported legacy provenance: route={route!r}, provenance={provenance!r}")


def select_legacy_candidates(
    verified_rows: Iterable[dict[str, Any]],
    combined_rows: Iterable[dict[str, Any]],
    current_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    verified = list(verified_rows)
    combined_by_hash = {str(row["conversation_hash"]): row for row in combined_rows}
    current = list(current_rows)
    current_conversations = {
        conversation_fingerprint(row["messages"])
        for row in current
    }
    current_targets = {
        target_fingerprint(row.get("target_text") or row["messages"][int(row["target_message_index"])]["content"])
        for row in current
    }

    counts = {
        "historical_rows": len(verified),
        "historical_positive": 0,
        "lmsys_excluded": 0,
        "conversation_overlap_excluded": 0,
        "target_overlap_excluded": 0,
        "within_legacy_target_duplicates": 0,
    }
    eligible: list[dict[str, Any]] = []
    for row in verified:
        if str(row.get("judge_label", "")).lower() != "positive":
            continue
        counts["historical_positive"] += 1
        conversation_id = str(row["conversation_hash"])
        combined = combined_by_hash.get(conversation_id, {})
        provenance = parse_provenance(combined.get("provenance"))
        source = release_source(str(row.get("source", "")), provenance)
        if source is None:
            counts["lmsys_excluded"] += 1
            continue

        messages, target_index = full_conversation(combined, row)
        raw_target_text = str(
            combined.get("flagged_text") or row.get("target_text") or messages[target_index]["content"]
        )
        target_text = messages[target_index]["content"]
        if not normalize_text(target_text).startswith(normalize_text(raw_target_text)):
            raise ValueError(f"Target text does not match target message for {conversation_id}")
        conversation_fp = conversation_fingerprint(messages)
        target_fp = target_fingerprint(target_text)
        if conversation_fp in current_conversations:
            counts["conversation_overlap_excluded"] += 1
            continue
        if target_fp in current_targets:
            counts["target_overlap_excluded"] += 1
            continue

        eligible.append(
            {
                "source": source,
                "split": "train",
                "conversation_id": conversation_id,
                "conversation_hash": conversation_id,
                "messages": messages,
                "target_message_index": target_index,
                "target_text": target_text,
                "text": target_text,
                "message_hash": message_hash(source, conversation_id, target_index, target_text),
                "language": row.get("language"),
                "country": row.get("country"),
                "discovery_split": "legacy_probe_gpt52",
                "legacy_source_route": row.get("source"),
                "legacy_probe_score": row.get("probe_score"),
                "legacy_message_judge_score": row.get("gpt_score"),
                "legacy_context_judge_model": row.get("judge_model"),
                "legacy_context_judge_confidence": row.get("judge_confidence"),
                "legacy_context_judge_revision": LEGACY_VERIFIED_REVISION,
                "legacy_candidate_revision": LEGACY_COMBINED_REVISION,
                "legacy_provenance": provenance,
                "conversation_fingerprint": conversation_fp,
                "target_fingerprint": target_fp,
            }
        )

    eligible.sort(
        key=lambda row: (
            -float(row.get("legacy_context_judge_confidence") or 0.0),
            -float(row.get("legacy_message_judge_score") or 0.0),
            str(row["conversation_id"]),
        )
    )
    selected: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for row in eligible:
        if row["target_fingerprint"] in seen_targets:
            counts["within_legacy_target_duplicates"] += 1
            continue
        seen_targets.add(row["target_fingerprint"])
        selected.append(row)

    selected.sort(key=lambda row: (str(row["source"]), str(row["conversation_id"])))
    manifest = {
        "legacy_verified_repo": LEGACY_VERIFIED_REPO,
        "legacy_verified_revision": LEGACY_VERIFIED_REVISION,
        "legacy_combined_repo": LEGACY_COMBINED_REPO,
        "legacy_combined_revision": LEGACY_COMBINED_REVISION,
        "selection_rule": (
            "historical context-judge positive; exclude LMSYS; remove exact canonical "
            "conversation and normalized target overlap with the current release; retain the "
            "highest-confidence row for duplicate legacy target text"
        ),
        **counts,
        "selected_for_current_reverification": len(selected),
    }
    return selected, manifest
