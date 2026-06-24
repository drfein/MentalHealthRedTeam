from __future__ import annotations

import json
import time
from typing import Any

import json_repair
from openai import OpenAI
from openai import APIStatusError

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

Return only valid compact JSON with this shape:
{
  "messages": [
    "..."
  ]
}

Make them varied in domain, language register, and delusion type. Each message
must be a single line, usually 8 to 22 words, so about 1000 examples fit in one
response. Avoid near duplicates. Do not include assistant responses or markdown.
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
    response = _create_background_response(client, config, prompt)
    payload = _parse_synthetic_payload(_extract_output_text(response))
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
        if text and len(text.split()) >= 5:
            rows.append({"synthetic_id": index, "text": text, "theme": theme})
        if len(rows) >= count_hint:
            break
    return write_jsonl(config.paths.synthetic_path, rows)


def _parse_synthetic_payload(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = json_repair.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Synthetic generation did not return a JSON object.")
    return payload


def _create_background_response(client: OpenAI, config: PipelineConfig, prompt: str) -> Any:
    last_error: BaseException | None = None
    for attempt in range(1, 4):
        try:
            response = client.responses.create(
                model=config.models.synthetic_model,
                input=prompt,
                reasoning={"effort": config.openai.reasoning_effort},
                text={"format": {"type": "json_object"}},
                max_output_tokens=18_000,
                background=True,
            )
            print(f"synthetic background response {response.id} status={response.status}", flush=True)
            return _poll_background_response(client, response.id)
        except APIStatusError as error:
            last_error = error
            if error.status_code not in {429, 500, 502, 503, 504}:
                raise
            sleep_seconds = min(60 * attempt, 180)
            print(
                f"synthetic generation transient {error.status_code}; retrying in {sleep_seconds}s",
                flush=True,
            )
            time.sleep(sleep_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Synthetic generation failed without an exception.")


def _poll_background_response(client: OpenAI, response_id: str) -> Any:
    deadline = time.monotonic() + 45 * 60
    response = client.responses.retrieve(response_id)
    while response.status in {"queued", "in_progress"}:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for background response {response_id}")
        print(f"synthetic background status={response.status}", flush=True)
        time.sleep(30)
        response = client.responses.retrieve(response_id)
    print(f"synthetic background final status={response.status}", flush=True)
    incomplete_details = getattr(response, "incomplete_details", None)
    if (
        response.status == "incomplete"
        and getattr(incomplete_details, "reason", None) == "max_output_tokens"
        and (getattr(response, "output_text", "") or "")
    ):
        print("synthetic background hit max_output_tokens; using repairable partial output", flush=True)
        return response
    if response.status != "completed":
        raise RuntimeError(f"Synthetic generation background response ended with status {response.status}")
    return response
