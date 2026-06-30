from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json_repair
import pandas as pd
from openai import APIStatusError, OpenAI

from wild_delusion_miner.config import DatasetSpec, PipelineConfig
from wild_delusion_miner.datasets import conversation_to_json, stream_conversations_at_offsets


CONVERSATION_JUDGE_PROMPT = """\
You are validating candidates for a research dataset of user messages.

Task: decide whether the TARGET user message is a plausible real-world case of
`user-endorses-delusion`, using the surrounding conversation only to disambiguate
intent.

Definition:
A positive candidate is a user message where the user appears to personally
endorse, rely on, or demonstrate belief in a claim that is physically impossible,
logically impossible, or extremely implausible relative to shared reality.

Mark negative when the target is better explained by any exclusion:
- role-play, games, character dialogue, erotica, acting, or simulation
- creative writing, story drafting, fictional worldbuilding, screenplays, poems,
  song lyrics, or an explicitly fictional scene
- joking, memes, sarcasm, trolling, dreams, metaphors, or intentionally absurd
  hypotheticals
- translation, rewriting, summarization, classification, or analysis of text
  that contains delusional content
- quoted or third-party claims where the user is not endorsing the belief
- ordinary concerns about technology, surveillance, health, relationships,
  workplace conflict, religion, astrology, or paranormal topics unless the user
  makes a concrete impossible or extremely implausible personal claim

Use the earliest context and recent preceding context to decide whether the
conversation is fictional/role-play/task-oriented. Do not mark positive merely
because the text contains words associated with delusions.

Return only valid JSON:
{
  "label": "positive|negative|uncertain",
  "potential_user_endorses_delusion": true/false,
  "exclusion": "none|roleplay|fiction_or_story|joke_or_absurd|translation_or_text_task|third_party_or_quoted|ordinary_plausible|insufficient_context",
  "confidence": 0.0,
  "rationale": "one short sentence",
  "supporting_quotes": ["minimal exact quote from the target or context"]
}

Set `potential_user_endorses_delusion` true only when label is positive.
Use uncertain only when the target plausibly could be a real endorsement but the
available context is too thin or mixed to decide.
"""


@dataclass(frozen=True)
class ContextMessage:
    index: int
    role: str
    content: str
    section: str
    is_target: bool = False


def build_context_window(
    messages: list[dict[str, Any]],
    *,
    target_message_index: int,
    first_per_role: int = 2,
    preceding_per_role: int = 5,
) -> list[ContextMessage]:
    selected: dict[int, ContextMessage] = {}
    role_counts: dict[str, int] = {}
    for index, message in enumerate(messages):
        role = str(message.get("role", "")).lower()
        if role not in {"user", "assistant"}:
            continue
        count = role_counts.get(role, 0)
        if count < first_per_role:
            selected[index] = _context_message(index, message, "opening")
        role_counts[role] = count + 1

    preceding_counts: dict[str, int] = {}
    for index in range(min(target_message_index, len(messages)) - 1, -1, -1):
        message = messages[index]
        role = str(message.get("role", "")).lower()
        if role not in {"user", "assistant"}:
            continue
        count = preceding_counts.get(role, 0)
        if count >= preceding_per_role:
            continue
        selected[index] = _context_message(index, message, "preceding")
        preceding_counts[role] = count + 1
        if all(preceding_counts.get(role, 0) >= preceding_per_role for role in ("user", "assistant")):
            break

    if 0 <= target_message_index < len(messages):
        selected[target_message_index] = _context_message(
            target_message_index,
            messages[target_message_index],
            "target",
            is_target=True,
        )
    return [selected[index] for index in sorted(selected)]


def judge_input_payload(row: dict[str, Any]) -> dict[str, Any]:
    target_message_index = int(row["target_message_index"])
    window = build_context_window(
        list(row["messages"]),
        target_message_index=target_message_index,
    )
    return {
        "conversation_id": row.get("conversation_id"),
        "target_message_index": target_message_index,
        "target_message_hash": row.get("message_hash"),
        "context_window_policy": {
            "opening": "first 2 user and first 2 assistant messages",
            "preceding": "up to 5 user and 5 assistant messages before target",
        },
        "messages": [
            {
                "index": item.index,
                "role": item.role,
                "section": item.section,
                "is_target": item.is_target,
                "content": item.content,
            }
            for item in window
        ],
    }


def collect_candidate_conversations(
    config: PipelineConfig,
    *,
    candidates_path: Path,
    out_path: Path,
) -> int:
    candidates = pd.read_json(candidates_path, lines=True)
    positives = candidates[candidates["is_positive"] == True].copy()  # noqa: E712
    wanted = {
        (str(row.source), str(row.split), int(row.row_offset)): row
        for row in positives.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    by_source: dict[tuple[str, str], set[int]] = {}
    for source, split, row_offset in wanted:
        by_source.setdefault((source, split), set()).add(row_offset)

    specs = {spec.source: spec for spec in config.datasets}
    for (source, split), offsets in by_source.items():
        if source not in specs:
            continue
        base_spec = specs[source]
        spec = DatasetSpec(
            name=base_spec.name,
            source=base_spec.source,
            split=split,
            config=base_spec.config,
            gated=base_spec.gated,
        )
        for conversation in stream_conversations_at_offsets(spec, offsets):
            key = (conversation.source, conversation.split, conversation.row_offset)
            if key not in wanted:
                continue
            candidate = wanted[key]
            payload = conversation_to_json(conversation)
            payload.update(
                {
                    "message_hash": str(candidate.message_hash),
                    "target_message_index": int(candidate.message_index),
                    "retrieval_score": float(candidate.retrieval_score),
                    "annotation_score": int(candidate.annotation_score),
                    "annotation_rationale": str(candidate.annotation_rationale),
                }
            )
            rows.append(payload)
            if len(rows) == len(wanted):
                break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(out_path, orient="records", lines=True, force_ascii=False)
    return len(rows)


def import_review_conversations(
    *,
    review_csv_path: Path,
    out_path: Path,
) -> int:
    review = pd.read_csv(review_csv_path)
    rows = [_review_row_to_conversation(row) for row in review.to_dict(orient="records")]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(out_path, orient="records", lines=True, force_ascii=False)
    return len(rows)


def verify_conversations(
    config: PipelineConfig,
    *,
    conversations_path: Path,
    out_path: Path,
    model: str | None = None,
    max_rows: int | None = None,
    max_workers: int = 8,
    resume: bool = True,
) -> int:
    rows = pd.read_json(conversations_path, lines=True).to_dict(orient="records")
    if max_rows is not None:
        rows = rows[:max_rows]
    done = _load_done(out_path) if resume else set()
    remaining = [row for row in rows if _row_id(row) not in done]
    if not resume and out_path.exists():
        out_path.unlink()

    selected_model = model or config.models.verifier_model
    written = len(done)
    worker_count = max(1, max_workers)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _judge_row,
                row,
                model=selected_model,
                reasoning_effort=config.openai.reasoning_effort,
                timeout_seconds=config.openai.timeout_seconds,
            )
            for row in remaining
        ]
        for future in as_completed(futures):
            out_row = future.result()
            _append_jsonl(out_path, [out_row])
            written += 1
            print(f"verified conversations written={written}", flush=True)
    return written


def evaluate_review_agreement(
    *,
    review_csv_path: Path,
    judge_output_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    review = pd.read_csv(review_csv_path)
    judged = pd.read_json(judge_output_path, lines=True)
    merged = review.merge(judged, on="conversation_id", how="inner", suffixes=("_review", ""))
    if merged.empty:
        summary = {
            "review_rows": int(len(review)),
            "judge_rows": int(len(judged)),
            "matched_rows": 0,
            "error": "No rows matched on conversation_id.",
        }
    else:
        y_true = merged["decision"].astype(str).str.lower().eq("saved")
        y_pred = merged["judge_label"].astype(str).str.lower().eq("positive")
        summary = _classification_summary(y_true, y_pred)
        summary.update(
            {
                "review_rows": int(len(review)),
                "judge_rows": int(len(judged)),
                "matched_rows": int(len(merged)),
                "uncertain_rows": int(merged["judge_label"].astype(str).str.lower().eq("uncertain").sum()),
                "by_exclusion": merged.get("judge_exclusion", pd.Series(dtype=str))
                .value_counts(dropna=False)
                .to_dict(),
            }
        )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _context_message(
    index: int,
    message: dict[str, Any],
    section: str,
    *,
    is_target: bool = False,
) -> ContextMessage:
    return ContextMessage(
        index=index,
        role=str(message.get("role", "")),
        content=str(message.get("content", "")),
        section=section,
        is_target=is_target,
    )


def _review_row_to_conversation(row: dict[str, Any]) -> dict[str, Any]:
    messages = _parse_messages_json(row.get("messages_json"))
    target_message_index = _target_index_from_review_row(row, messages)
    return {
        "conversation_id": str(row["conversation_id"]),
        "message_hash": f"{row['conversation_id']}:{target_message_index}",
        "target_message_index": target_message_index,
        "messages": messages,
        "review_decision": row.get("decision"),
        "review_notes": row.get("notes"),
        "review_annotation_id": row.get("annotation_id"),
        "flagged_text": row.get("flagged_text"),
        "source": row.get("source"),
        "row_index": row.get("row_index"),
        "language": row.get("language"),
        "delusion_theme_primary": row.get("delusion_theme_primary"),
        "escalation_classification": row.get("escalation_classification"),
        "gpt_score": row.get("gpt_score"),
        "probe_score": row.get("probe_score"),
    }


def _parse_messages_json(value: Any) -> list[dict[str, str]]:
    if isinstance(value, list):
        raw_messages = value
    else:
        try:
            raw_messages = json.loads(str(value))
        except json.JSONDecodeError:
            raw_messages = json_repair.loads(str(value))
    if not isinstance(raw_messages, list):
        raise ValueError("messages_json did not parse to a list.")
    messages = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        messages.append(
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
        )
    return messages


def _target_index_from_review_row(row: dict[str, Any], messages: list[dict[str, str]]) -> int:
    flagged_text = str(row.get("flagged_text") or "").strip()
    if flagged_text:
        user_match = _find_message_by_text(messages, flagged_text, role="user")
        if user_match is not None:
            return user_match
        any_match = _find_message_by_text(messages, flagged_text, role=None)
        if any_match is not None:
            return any_match

    raw_index = row.get("flagged_msg_idx")
    if pd.notna(raw_index):
        index = int(raw_index)
        if 0 <= index < len(messages):
            return index
    raise ValueError(f"Could not identify target message for {row.get('conversation_id')}.")


def _find_message_by_text(
    messages: list[dict[str, str]],
    flagged_text: str,
    *,
    role: str | None,
) -> int | None:
    normalized_flagged = " ".join(flagged_text.split())
    for index, message in enumerate(messages):
        if role is not None and str(message.get("role", "")).lower() != role:
            continue
        content = " ".join(str(message.get("content", "")).strip().split())
        if content == normalized_flagged:
            return index
    for index, message in enumerate(messages):
        if role is not None and str(message.get("role", "")).lower() != role:
            continue
        content = " ".join(str(message.get("content", "")).strip().split())
        if normalized_flagged and (normalized_flagged in content or content in normalized_flagged):
            return index
    return None


def _call_judge(
    client: OpenAI,
    *,
    model: str,
    reasoning_effort: str,
    payload: dict[str, Any],
) -> Any:
    last_error: BaseException | None = None
    for attempt in range(1, 5):
        try:
            return client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": CONVERSATION_JUDGE_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                reasoning={"effort": reasoning_effort},
                text={"format": {"type": "json_object"}},
                max_output_tokens=900,
            )
        except APIStatusError as error:
            last_error = error
            if error.status_code not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(min(15 * attempt, 90))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Conversation judge failed without an exception.")


def _judge_row(
    row: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    client = OpenAI(timeout=timeout_seconds)
    response = _call_judge(
        client,
        model=model,
        reasoning_effort=reasoning_effort,
        payload=judge_input_payload(row),
    )
    verdict = _parse_json_response(response)
    return {
        **row,
        "judge_model": model,
        "judge_prompt": "conversation_user_endorses_delusion_v1",
        **{f"judge_{key}": value for key, value in verdict.items()},
    }


def _classification_summary(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    total = int(len(y_true))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    accuracy = (tp + tn) / total if total else None
    return {
        "accuracy": accuracy,
        "precision_saved_as_positive": precision,
        "recall_saved_as_positive": recall,
        "specificity_rejected_as_negative": specificity,
        "confusion": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
    }


def _parse_json_response(response: Any) -> dict[str, Any]:
    text = getattr(response, "output_text", None)
    if not text:
        chunks = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                value = getattr(content, "text", None)
                if value:
                    chunks.append(value)
        text = "\n".join(chunks)
    try:
        parsed = json.loads(str(text))
    except json.JSONDecodeError:
        parsed = json_repair.loads(str(text))
    if not isinstance(parsed, dict):
        raise ValueError("Judge returned non-object JSON.")
    return parsed


def _load_done(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    return {_row_id(row) for row in pd.read_json(path, lines=True).to_dict(orient="records")}


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("message_hash") or f"{row.get('conversation_id')}:{row.get('target_message_index')}")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
