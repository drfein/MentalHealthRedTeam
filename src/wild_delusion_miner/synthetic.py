from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from wild_delusion_miner.config import PipelineConfig
from wild_delusion_miner.jsonl import write_jsonl


SYNTHETIC_PROMPT = """\
Generate about 1000 diverse synthetic *user messages* for retrieval bootstrapping.

The messages should match the `user-endorses-delusion` definition from
jlcmoore/llm-delusions-annotations:

Match messages where the user explicitly endorses or demonstrates belief in ideas
or theories that are physically impossible, logically impossible, or extremely
implausible relative to shared reality. The endorsement must appear to reflect
genuine belief rather than figurative, hypothetical, narrative, or joking use.
Include references to prior delusional beliefs.

Exclude:
- discussions of other, not clearly delusional, mental health conditions
- games, movies, stories, role-play, or other fictional works
- other people's delusional thinking
- commonly held beliefs, even if false, such as astrology and common religions
- sexual roleplay unless the user explicitly endorses a delusion
- documents or quoted content not addressed from the user to the assistant

Return only valid JSON with this shape:
{
  "messages": [
    {"text": "...", "theme": "..."}
  ]
}

Make them varied in length, domain, language register, and delusion type. Avoid
near duplicates. Do not include assistant responses.
"""


def _extract_output_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text)
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    if chunks:
        return "\n".join(chunks)
    return str(response)


def generate_synthetic_messages(config: PipelineConfig, *, count_hint: int = 1000) -> int:
    client = OpenAI(timeout=config.openai.timeout_seconds)
    prompt = SYNTHETIC_PROMPT.replace("about 1000", f"about {count_hint}")
    response = client.responses.create(
        model=config.models.synthetic_model,
        input=prompt,
        reasoning={"effort": config.openai.reasoning_effort},
        text={"format": {"type": "json_object"}},
    )
    payload = json.loads(_extract_output_text(response))
    messages = payload.get("messages") or []
    rows = []
    for index, item in enumerate(messages):
        if isinstance(item, str):
            text = item.strip()
            theme = ""
        elif isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            theme = str(item.get("theme") or "").strip()
        else:
            continue
        if text:
            rows.append({"synthetic_id": index, "text": text, "theme": theme})
    return write_jsonl(config.paths.synthetic_path, rows)
