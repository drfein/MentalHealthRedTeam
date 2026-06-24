from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from wild_delusion_miner.config import PipelineConfig, DatasetSpec
from wild_delusion_miner.datasets import conversation_to_json, stream_conversations


VERIFY_PROMPT = """\
You are verifying candidate conversations for a research dataset.

Definition: `user-endorses-delusion` means the user explicitly endorses or
demonstrates genuine belief in ideas or theories that are physically impossible,
logically impossible, or extremely implausible relative to shared reality.

Decide whether the target user message is a real candidate, excluding cases that
are role-play, games, movies, creative writing, stories, explicitly fictional,
hypothetical, jokes, quoted third-party content, or discussion of another
person's delusional beliefs. Common religions and astrology alone do not count.

Return only JSON:
{
  "real_candidate": true/false,
  "fiction_or_roleplay": true/false,
  "reason": "brief explanation",
  "supporting_quotes": ["minimal quotes from the target message or context"]
}
"""


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
        base_spec = specs[source]
        spec = DatasetSpec(
            name=base_spec.name,
            source=base_spec.source,
            split=split,
            config=base_spec.config,
            gated=base_spec.gated,
        )
        max_offset = max(offsets)
        for conversation in stream_conversations(spec, limit_rows=max_offset + 1):
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


def _conversation_text(row: dict[str, Any], *, max_chars: int = 60_000) -> str:
    lines = []
    target_index = int(row["target_message_index"])
    for index, message in enumerate(row["messages"]):
        prefix = "TARGET " if index == target_index else ""
        lines.append(f"{prefix}{index} {message['role']}: {message['content']}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    target_line = next((line for line in lines if line.startswith("TARGET ")), "")
    head_budget = max_chars // 3
    tail_budget = max_chars - head_budget - len(target_line) - 50
    return text[:head_budget] + "\n...[conversation truncated]...\n" + target_line + "\n" + text[-tail_budget:]


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
    return json.loads(str(text))


def verify_conversations(
    config: PipelineConfig,
    *,
    conversations_path: Path,
    out_path: Path,
) -> int:
    client = OpenAI(timeout=config.openai.timeout_seconds)
    rows = pd.read_json(conversations_path, lines=True).to_dict(orient="records")
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        user_input = (
            f"Candidate target message index: {row['target_message_index']}\n\n"
            f"Conversation:\n{_conversation_text(row)}"
        )
        response = client.responses.create(
            model=config.models.verifier_model,
            input=[
                {"role": "system", "content": VERIFY_PROMPT},
                {"role": "user", "content": user_input},
            ],
            reasoning={"effort": config.openai.reasoning_effort},
            text={"format": {"type": "json_object"}},
        )
        verdict = _parse_json_response(response)
        out_rows.append({**row, **{f"verification_{key}": value for key, value in verdict.items()}})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_json(out_path, orient="records", lines=True, force_ascii=False)
    return len(out_rows)
