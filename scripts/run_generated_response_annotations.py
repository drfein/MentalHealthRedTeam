#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources
from pathlib import Path
from typing import Any

import json_repair
from openai import APIStatusError, OpenAI, OpenAIError
from tqdm import tqdm

from llm_delusions_annotations.annotator import AnnotatableMessage, build_annotation_request


DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_INPUT = Path("results/response_analysis/10model_low_reasoning/latest_success_responses.jsonl")
DEFAULT_OUT_ROOT = Path("results/generated_response_annotations")


def annotation_slug(annotation_id: str) -> str:
    return annotation_id.replace("-", "_")


def load_cutoffs() -> dict[str, int]:
    cutoffs_path = resources.files("llm_delusions_annotations.data").joinpath("cutoffs.csv")
    cutoffs: dict[str, int] = {}
    with cutoffs_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            annotation_id = str(row.get("annotation_id") or "").strip()
            cutoff = str(row.get("cutoff") or "").strip()
            if annotation_id and cutoff:
                cutoffs[annotation_id] = int(float(cutoff))
    return cutoffs


def load_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            generation_id = str(row.get("generation_id") or "")
            if generation_id and not row.get("annotation_error"):
                done.add(generation_id)
    return done


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks)


def annotate_one(
    row: dict[str, Any],
    *,
    annotation_id: str,
    model: str,
    cutoff: int,
    timeout: float,
) -> dict[str, Any]:
    generation_id = str(row.get("generation_id") or "")
    target_text = str(row.get("target_text") or "")
    assistant_text = str(row.get("response_text") or "")
    out: dict[str, Any] = {
        "generation_id": generation_id,
        "generated_model": row.get("model"),
        "annotation_id": annotation_id,
        "annotation_model": model,
        "cutoff": cutoff,
        "annotation_score": None,
        "is_positive": False,
        "annotation_error": None,
        "annotation_rationale": None,
        "annotation_quotes": [],
        "target_text": target_text,
        "response_text": assistant_text,
    }
    try:
        message = AnnotatableMessage(
            content=assistant_text,
            role="assistant",
            preceding_messages=[{"role": "user", "content": target_text}],
        )
        request = build_annotation_request(message, annotation_id)
        client = OpenAI(timeout=timeout)
        last_error: BaseException | None = None
        for attempt in range(1, 6):
            try:
                response = client.responses.create(
                    model=model,
                    input=request,
                    max_output_tokens=512,
                    text={"format": {"type": "json_object"}},
                )
                parsed = json_repair.loads(response_text(response))
                score = parsed.get("score")
                out["annotation_score"] = int(score) if score is not None else None
                out["is_positive"] = out["annotation_score"] is not None and out["annotation_score"] >= cutoff
                out["annotation_rationale"] = parsed.get("rationale")
                out["annotation_quotes"] = parsed.get("quotes") or []
                usage = getattr(response, "usage", None)
                if usage is not None:
                    out["input_tokens"] = getattr(usage, "input_tokens", None)
                    out["output_tokens"] = getattr(usage, "output_tokens", None)
                return out
            except APIStatusError as error:
                last_error = error
                if error.status_code not in {429, 500, 502, 503, 504}:
                    raise
                time.sleep(min(10 * attempt, 60))
            except OpenAIError as error:
                last_error = error
                time.sleep(min(10 * attempt, 60))
        if last_error:
            raise last_error
        raise RuntimeError("annotation failed without exception")
    except Exception as error:
        out["annotation_error"] = f"{type(error).__name__}: {error}"
        return out


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(output_path: Path, summary_path: Path, *, annotation_id: str, cutoff: int) -> dict[str, Any]:
    rows = load_rows(output_path)
    by_model: dict[str, dict[str, Any]] = {}
    input_tokens = 0
    output_tokens = 0
    for row in rows:
        model = str(row.get("generated_model") or "")
        entry = by_model.setdefault(model, {"n": 0, "positive": 0, "errors": 0, "scores": {}})
        entry["n"] += 1
        if row.get("annotation_error"):
            entry["errors"] += 1
        if bool(row.get("is_positive")):
            entry["positive"] += 1
        score = row.get("annotation_score")
        if score is not None:
            entry["scores"][str(score)] = entry["scores"].get(str(score), 0) + 1
        input_tokens += int(row.get("input_tokens") or 0)
        output_tokens += int(row.get("output_tokens") or 0)
    for entry in by_model.values():
        denom = max(1, entry["n"] - entry["errors"])
        entry["positive_rate_non_error"] = entry["positive"] / denom
    summary = {
        "annotation_id": annotation_id,
        "cutoff": cutoff,
        "output_path": str(output_path),
        "total_rows": len(rows),
        "total_positive": sum(v["positive"] for v in by_model.values()),
        "total_errors": sum(v["errors"] for v in by_model.values()),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "by_generated_model": dict(sorted(by_model.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-id", required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--cutoff", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    cutoff = args.cutoff if args.cutoff is not None else load_cutoffs()[args.annotation_id]
    out_dir = args.out_root / args.model / args.annotation_id
    output_path = out_dir / f"{annotation_slug(args.annotation_id)}_annotations.jsonl"
    summary_path = out_dir / "summary.json"
    rows = load_rows(args.input, args.limit)
    done = load_done(output_path)
    remaining = [row for row in rows if str(row.get("generation_id") or "") not in done]
    print(
        json.dumps(
            {
                "annotation_id": args.annotation_id,
                "cutoff": cutoff,
                "input_rows": len(rows),
                "done": len(done),
                "remaining": len(remaining),
                "model": args.model,
                "output_path": str(output_path),
            },
            indent=2,
        ),
        flush=True,
    )
    for start in tqdm(range(0, len(remaining), args.batch_size), desc=args.annotation_id):
        batch = remaining[start : start + args.batch_size]
        records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    annotate_one,
                    row,
                    annotation_id=args.annotation_id,
                    model=args.model,
                    cutoff=cutoff,
                    timeout=args.timeout,
                )
                for row in batch
            ]
            for future in as_completed(futures):
                records.append(future.result())
        append_rows(output_path, records)
        summary = write_summary(output_path, summary_path, annotation_id=args.annotation_id, cutoff=cutoff)
        print(
            json.dumps(
                {
                    "annotation_id": args.annotation_id,
                    "written": summary["total_rows"],
                    "positive": summary["total_positive"],
                    "errors": summary["total_errors"],
                }
            ),
            flush=True,
        )
    write_summary(output_path, summary_path, annotation_id=args.annotation_id, cutoff=cutoff)


if __name__ == "__main__":
    main()
