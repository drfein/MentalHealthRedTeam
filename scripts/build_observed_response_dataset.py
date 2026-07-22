from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ASSISTANT_ROLES = {"assistant", "llm"}
USER_ROLES = {"human", "user"}


def canonical_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_provenance(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("hf_provenance") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def normalized_messages(messages: list[dict[str, Any]], end: int) -> list[dict[str, str]]:
    normalized = []
    for message in messages[: end + 1]:
        role = str(message.get("role", "")).lower()
        if role in USER_ROLES:
            role = "user"
        elif role in ASSISTANT_ROLES:
            role = "assistant"
        else:
            continue
        content = str(message.get("content", "") or "").strip()
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def candidate_matches_conversation(
    release_row: dict[str, Any], candidate_row: dict[str, Any]
) -> bool:
    conversation_id = str(
        release_row.get("conversation_id") or release_row.get("conversation_hash") or ""
    )
    provenance = parse_provenance(candidate_row)
    candidate_ids = {
        str(candidate_row.get("conversation_id") or ""),
        str(provenance.get("url") or ""),
        str(provenance.get("conversation_id") or ""),
    }
    return bool(conversation_id and conversation_id in candidate_ids)


def serving_metadata(
    release_row: dict[str, Any], candidate_row: dict[str, Any]
) -> tuple[str, str | None]:
    source = str(release_row.get("source") or "")
    provenance = parse_provenance(candidate_row)
    platform = str(provenance.get("platform") or "").strip().lower()
    if not platform:
        if source.startswith("sharechat_"):
            platform = source.removeprefix("sharechat_")
        elif source == "wildchat_full":
            platform = "wildchat_unspecified"
        else:
            platform = source or "unknown"
    model = str(provenance.get("model") or "").strip() or None
    # Some Grok exports use "human" as a role marker, not a serving-model name.
    if model == "human":
        model = None
    return platform, model


def build_index(
    canonical_rows: list[dict[str, Any]],
) -> dict[str, list[tuple[int, dict[str, Any], int, str]]]:
    index: dict[str, list[tuple[int, dict[str, Any], int, str]]] = defaultdict(list)
    for row_index, row in enumerate(canonical_rows):
        messages = list(row["messages"])
        for message_index, message in enumerate(messages[:-1]):
            if str(message.get("role", "")).lower() not in USER_ROLES:
                continue
            reply = messages[message_index + 1]
            if str(reply.get("role", "")).lower() not in ASSISTANT_ROLES:
                continue
            response = str(reply.get("content", "") or "").strip()
            if response:
                index[canonical_text(message.get("content"))].append(
                    (row_index, row, message_index, response)
                )
    return index


def choose_candidate(
    release_row: dict[str, Any],
    candidates: list[tuple[int, dict[str, Any], int, str]],
) -> tuple[tuple[int, dict[str, Any], int, str] | None, str]:
    if not candidates:
        return None, "no_canonical_match"

    by_response: dict[str, list[tuple[int, dict[str, Any], int, str]]] = defaultdict(list)
    for candidate in candidates:
        by_response[canonical_text(candidate[3])].append(candidate)
    if len(by_response) == 1:
        return next(iter(by_response.values()))[0], "unique_response"

    provenance_matches = [
        candidate
        for candidate in candidates
        if candidate_matches_conversation(release_row, candidate[1])
    ]
    matched_responses = {canonical_text(candidate[3]) for candidate in provenance_matches}
    if len(matched_responses) == 1:
        return provenance_matches[0], "provenance_resolved"
    return None, "ambiguous_response"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover immediate production assistant replies for the combined release."
    )
    parser.add_argument(
        "--release", type=Path, default=Path("data/releases/WildDelusionCombined/train.parquet")
    )
    parser.add_argument(
        "--canonical-conversations",
        type=Path,
        default=Path("data/verification/wilddelusion_combined_conversations.jsonl"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/observed_assistant_responses")
    )
    args = parser.parse_args()

    release_rows = pd.read_parquet(args.release).to_dict(orient="records")
    canonical_rows = read_jsonl(args.canonical_conversations)
    index = build_index(canonical_rows)

    recovered = []
    prompts = []
    match_status = Counter()
    for original_row_idx, release_row in enumerate(release_rows):
        candidates = index.get(canonical_text(release_row.get("target_text")), [])
        selected, status = choose_candidate(release_row, candidates)
        match_status[status] += 1
        if selected is None:
            continue
        canonical_row_idx, canonical_row, target_index, response = selected
        platform, serving_model = serving_metadata(release_row, canonical_row)
        conversation_id = str(
            release_row.get("conversation_id") or release_row.get("conversation_hash")
        )
        base = {
            "original_row_idx": original_row_idx,
            "intervention_arm": "observed_production_reply",
            "source": release_row.get("source"),
            "discovery_split": release_row.get("discovery_split"),
            "conversation_id": conversation_id,
            "message_hash": release_row.get("message_hash"),
            "target_message_index": int(release_row["target_message_index"]),
            "target_text": release_row.get("target_text"),
            "serving_platform": platform,
            "serving_model_reported": serving_model,
            "canonical_row_idx": canonical_row_idx,
            "canonical_target_message_index": target_index,
            "match_status": status,
        }
        recovered.append({**base, "response": response})
        prompts.append(
            {
                **base,
                "messages": normalized_messages(canonical_row["messages"], target_index),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    response_path = args.out_dir / "responses.jsonl"
    prompt_path = args.out_dir / "prompts.jsonl"
    with response_path.open("w", encoding="utf-8") as handle:
        for row in recovered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with prompt_path.open("w", encoding="utf-8") as handle:
        for row in prompts:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    conversation_counts = Counter(row["conversation_id"] for row in recovered)
    summary = {
        "release": str(args.release),
        "canonical_conversations": str(args.canonical_conversations),
        "release_rows": len(release_rows),
        "canonical_rows": len(canonical_rows),
        "recovered_replies": len(recovered),
        "match_status": dict(sorted(match_status.items())),
        "by_platform": dict(sorted(Counter(row["serving_platform"] for row in recovered).items())),
        "conversations_with_replies": len(conversation_counts),
        "multi_target_conversations_with_replies": sum(
            count > 1 for count in conversation_counts.values()
        ),
        "replies_in_multi_target_conversations": sum(
            count for count in conversation_counts.values() if count > 1
        ),
        "maximum_recovered_targets_per_conversation": max(conversation_counts.values()),
        "matching_policy": (
            "NFKC + whitespace + casefold target matching across every canonical user turn; "
            "distinct reply collisions resolved only by exact conversation URL/ID provenance"
        ),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
