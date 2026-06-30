from __future__ import annotations

import dataclasses
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from llm_delusions_annotations.annotator import (
    AnnotatableMessage,
    Annotator,
    build_annotation_request,
)
from llm_delusions_annotations.classify_messages import extract_matches_from_response_text
from openai import APIStatusError, OpenAI

from wild_delusion_miner.config import PipelineConfig

GPT55_INPUT_PER_1M = 5.00
GPT55_CACHED_INPUT_PER_1M = 0.50
GPT55_OUTPUT_PER_1M = 30.00


@dataclass(frozen=True)
class TokenCost:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    @property
    def usd(self) -> float:
        uncached = max(0, self.input_tokens - self.cached_input_tokens)
        return (
            (uncached / 1_000_000.0) * GPT55_INPUT_PER_1M
            + (self.cached_input_tokens / 1_000_000.0) * GPT55_CACHED_INPUT_PER_1M
            + (self.output_tokens / 1_000_000.0) * GPT55_OUTPUT_PER_1M
        )


def _result_to_row(input_row: dict[str, Any], result: dict[str, Any], config: PipelineConfig) -> dict[str, Any]:
    item = result.get(config.annotation.annotation_id)
    parsed = dataclasses.asdict(item) if item is not None else {}
    score = parsed.get("score")
    error = parsed.get("error")
    return {
        **input_row,
        "annotation_id": config.annotation.annotation_id,
        "annotation_model": config.models.annotation_model,
        "annotation_score": score,
        "is_positive": error is None and score is not None and int(score) >= config.annotation.cutoff,
        "annotation_error": error,
        "annotation_rationale": parsed.get("rationale"),
        "annotation_quotes": parsed.get("matches") or [],
    }


def annotate_jsonl(
    config: PipelineConfig,
    *,
    input_path: Path,
    output_path: Path,
    model: str | None = None,
    max_rows: int | None = None,
    resume: bool = False,
) -> int:
    rows = pd.read_json(input_path, lines=True).to_dict(orient="records")
    if max_rows is not None:
        rows = rows[:max_rows]
    done: set[str] = set()
    if resume and output_path.exists() and output_path.stat().st_size > 0:
        done_frame = pd.read_json(output_path, lines=True)
        if "message_hash" in done_frame:
            done = set(done_frame["message_hash"].astype(str))
            rows = [row for row in rows if str(row.get("message_hash")) not in done]
    annotator = Annotator(
        timeout=config.annotation.timeout_seconds,
        max_workers=config.annotation.max_workers,
    )
    selected_model = model or config.models.annotation_model
    written = len(done)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume and output_path.exists():
        output_path.unlink()
    for start in range(0, len(rows), config.annotation.batch_size):
        batch = rows[start : start + config.annotation.batch_size]
        messages = [
            AnnotatableMessage(content=str(row["text"]), role="user", preceding_messages=None)
            for row in batch
        ]
        results = annotator.annotate_messages(
            messages,
            model=selected_model,
            annotation_ids=[config.annotation.annotation_id],
        )
        with output_path.open("a", encoding="utf-8") as handle:
            for row, result in zip(batch, results):
                out_row = _result_to_row(row, result, config)
                out_row["annotation_model"] = selected_model
                handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                written += 1
        print(f"canonical annotations written={written}", flush=True)
    return written


def annotate_jsonl_openai_direct(
    config: PipelineConfig,
    *,
    input_path: Path,
    output_path: Path,
    model: str = "gpt-5.5",
    budget_usd: float = 20.0,
    max_rows: int | None = None,
    max_workers: int = 8,
    resume: bool = True,
) -> dict[str, Any]:
    rows = pd.read_json(input_path, lines=True).to_dict(orient="records")
    if max_rows is not None:
        rows = rows[:max_rows]
    done: set[str] = set()
    spent = 0.0
    if resume and output_path.exists() and output_path.stat().st_size > 0:
        existing = pd.read_json(output_path, lines=True)
        if "message_hash" in existing:
            done = set(existing["message_hash"].astype(str))
        if "estimated_cost_usd" in existing:
            spent = float(existing["estimated_cost_usd"].sum())
    elif output_path.exists():
        output_path.unlink()

    remaining = [row for row in rows if str(row.get("message_hash")) not in done]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = len(done)
    client = OpenAI(timeout=config.openai.timeout_seconds)

    for start in range(0, len(remaining), max_workers):
        if spent >= budget_usd:
            break
        batch = remaining[start : start + max_workers]
        records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_annotate_row_openai_direct, client, config, row, model)
                for row in batch
            ]
            for future in as_completed(futures):
                record, cost = future.result()
                records.append(record)
                spent += cost.usd
        _append_jsonl(output_path, records)
        written += len(records)
        print(
            f"direct canonical annotations written={written} "
            f"batch={len(records)} spent=${spent:.4f}",
            flush=True,
        )

    return {
        "input_rows": len(rows),
        "already_done": len(done),
        "new_written": written - len(done),
        "total_written": written,
        "estimated_spend_usd": round(spent, 6),
        "budget_usd": budget_usd,
    }


def positive_texts(path: Path) -> list[str]:
    rows = pd.read_json(path, lines=True)
    if rows.empty:
        return []
    return rows[rows["is_positive"] == True]["text"].astype(str).tolist()  # noqa: E712


def _annotate_row_openai_direct(
    client: OpenAI,
    config: PipelineConfig,
    row: dict[str, Any],
    model: str,
) -> tuple[dict[str, Any], TokenCost]:
    cost = TokenCost()
    try:
        message = AnnotatableMessage(content=str(row["text"]), role="user", preceding_messages=None)
        request = build_annotation_request(message, config.annotation.annotation_id)
        response = _create_annotation_response(client, config, model, request)
        cost = _response_cost(response)
        rationale, matches, score = extract_matches_from_response_text(_extract_output_text(response))
        result = {
            config.annotation.annotation_id: {
                "score": score,
                "error": None,
                "rationale": rationale,
                "matches": matches,
            }
        }
        parsed = result[config.annotation.annotation_id]
        out = {
            **row,
            "annotation_id": config.annotation.annotation_id,
            "annotation_model": model,
            "annotation_score": parsed["score"],
            "is_positive": int(parsed["score"]) >= config.annotation.cutoff,
            "annotation_error": None,
            "annotation_rationale": parsed["rationale"],
            "annotation_quotes": parsed["matches"],
        }
    except Exception as error:  # noqa: BLE001 - failed rows should remain auditable.
        out = {
            **row,
            "annotation_id": config.annotation.annotation_id,
            "annotation_model": model,
            "annotation_score": None,
            "is_positive": False,
            "annotation_error": str(error),
            "annotation_rationale": None,
            "annotation_quotes": [],
        }
    out["annotation_model"] = model
    out["estimated_cost_usd"] = cost.usd
    out["input_tokens"] = cost.input_tokens
    out["cached_input_tokens"] = cost.cached_input_tokens
    out["output_tokens"] = cost.output_tokens
    return out, cost


def _create_annotation_response(
    client: OpenAI,
    config: PipelineConfig,
    model: str,
    request: list[dict[str, str]],
) -> Any:
    last_error: BaseException | None = None
    for attempt in range(1, 5):
        try:
            return client.responses.create(
                model=model,
                input=request,
                reasoning={"effort": config.openai.reasoning_effort},
                text={"format": {"type": "json_object"}},
                max_output_tokens=512,
            )
        except APIStatusError as error:
            last_error = error
            if error.status_code not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(min(15 * attempt, 90))
    if last_error is not None:
        raise last_error
    raise RuntimeError("OpenAI annotation request failed without an exception.")


def _extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(value)
    return "\n".join(chunks)


def _response_cost(response: Any) -> TokenCost:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenCost()
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cached_input_tokens = 0
    details = getattr(usage, "input_tokens_details", None)
    if details is not None:
        cached_input_tokens = int(getattr(details, "cached_tokens", 0) or 0)
    return TokenCost(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
    )


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
