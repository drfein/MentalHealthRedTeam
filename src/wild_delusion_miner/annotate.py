from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pandas as pd
from llm_delusions_annotations.annotator import AnnotatableMessage, Annotator

from wild_delusion_miner.config import PipelineConfig


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
) -> int:
    rows = pd.read_json(input_path, lines=True).to_dict(orient="records")
    annotator = Annotator(
        timeout=config.annotation.timeout_seconds,
        max_workers=config.annotation.max_workers,
    )
    out_rows: list[dict[str, Any]] = []
    for start in range(0, len(rows), config.annotation.batch_size):
        batch = rows[start : start + config.annotation.batch_size]
        messages = [
            AnnotatableMessage(content=str(row["text"]), role="user", preceding_messages=None)
            for row in batch
        ]
        results = annotator.annotate_messages(
            messages,
            model=config.models.annotation_model,
            annotation_ids=[config.annotation.annotation_id],
        )
        out_rows.extend(_result_to_row(row, result, config) for row, result in zip(batch, results))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_json(output_path, orient="records", lines=True, force_ascii=False)
    return len(out_rows)


def positive_texts(path: Path) -> list[str]:
    rows = pd.read_json(path, lines=True)
    if rows.empty:
        return []
    return rows[rows["is_positive"] == True]["text"].astype(str).tolist()  # noqa: E712
