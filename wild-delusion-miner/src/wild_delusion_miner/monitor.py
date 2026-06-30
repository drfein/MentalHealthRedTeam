from __future__ import annotations

import html
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from wild_delusion_miner.config import PipelineConfig


def _count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _count_parquet(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(pq.read_metadata(path).num_rows)


def _sqlite_counts(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    conn = sqlite3.connect(path)
    try:
        return {
            "unique_messages": int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]),
            "message_refs": int(conn.execute("SELECT COUNT(*) FROM refs").fetchone()[0]),
        }
    finally:
        conn.close()


def _annotation_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    frame = pd.read_json(path, lines=True)
    if frame.empty:
        return {"rows": 0, "positives": 0, "hit_rate": 0.0, "by_source": {}}
    positives = frame[frame.get("is_positive", False) == True]  # noqa: E712
    by_source = {}
    if "source" in frame.columns:
        for source, group in frame.groupby("source"):
            source_hits = group[group.get("is_positive", False) == True]  # noqa: E712
            by_source[str(source)] = {
                "rows": int(len(group)),
                "positives": int(len(source_hits)),
                "hit_rate": float(len(source_hits) / len(group)) if len(group) else 0.0,
            }
    return {
        "rows": int(len(frame)),
        "positives": int(len(positives)),
        "hit_rate": float(len(positives) / len(frame)) if len(frame) else 0.0,
        "by_source": by_source,
    }


def pipeline_status(config: PipelineConfig) -> dict[str, Any]:
    corpus_manifest_path = config.paths.corpus_embedding_dir / "manifest.json"
    synthetic_count = _count_jsonl(config.paths.synthetic_path)
    corpus_manifest = (
        json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
        if corpus_manifest_path.exists()
        else None
    )
    return {
        "sqlite": _sqlite_counts(config.paths.sqlite_path),
        "synthetic_messages": synthetic_count,
        "corpus_embeddings": corpus_manifest,
        "initial_scores": _count_parquet(config.paths.retrieval_dir / "initial" / "scores.parquet"),
        "initial_top": _count_parquet(config.paths.retrieval_dir / "initial" / "top.parquet"),
        "calibration": _annotation_summary(
            config.paths.annotation_dir / "calibration_annotations.jsonl"
        ),
        "initial_candidates": _annotation_summary(
            config.paths.annotation_dir / "initial_candidates_annotated.jsonl"
        ),
        "true_positive_scores": _count_parquet(
            config.paths.retrieval_dir / "true_positive" / "scores.parquet"
        ),
        "true_positive_top": _count_parquet(
            config.paths.retrieval_dir / "true_positive" / "top.parquet"
        ),
        "true_positive_candidates": _annotation_summary(
            config.paths.annotation_dir / "true_positive_candidates_annotated.jsonl"
        ),
        "verified": _count_jsonl(config.paths.verification_dir / "verified_candidates.jsonl"),
    }


def _load_best_available_candidates(config: PipelineConfig) -> tuple[Path | None, pd.DataFrame]:
    candidates = [
        config.paths.verification_dir / "verified_candidates.jsonl",
        config.paths.annotation_dir / "true_positive_candidates_annotated.jsonl",
        config.paths.annotation_dir / "initial_candidates_annotated.jsonl",
        config.paths.annotation_dir / "calibration_annotations.jsonl",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size:
            return path, pd.read_json(path, lines=True)
    return None, pd.DataFrame()


def write_inspection_html(
    config: PipelineConfig,
    *,
    out_path: Path,
    per_source: int = 25,
    positives_only: bool = True,
) -> int:
    source_path, frame = _load_best_available_candidates(config)
    if frame.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("<!doctype html><title>No candidates</title><p>No candidates yet.</p>")
        return 0

    if positives_only and "is_positive" in frame.columns:
        frame = frame[frame["is_positive"] == True]  # noqa: E712
    if "verification_real_candidate" in frame.columns:
        frame = frame[frame["verification_real_candidate"] == True]  # noqa: E712
    if frame.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("<!doctype html><title>No positive candidates</title><p>No positive candidates yet.</p>")
        return 0

    sort_cols = [col for col in ("retrieval_score", "annotation_score") if col in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols, ascending=False)

    source_counts = Counter(frame["source"].astype(str)) if "source" in frame.columns else Counter()
    selected = []
    if "source" in frame.columns:
        for _source, group in frame.groupby("source", sort=True):
            selected.append(group.head(per_source))
        review = pd.concat(selected, ignore_index=True)
    else:
        review = frame.head(per_source)

    rows_html = []
    for row in review.to_dict(orient="records"):
        source = html.escape(str(row.get("source", "unknown")))
        score = html.escape(str(row.get("retrieval_score", "")))
        ann = html.escape(str(row.get("annotation_score", "")))
        text = html.escape(str(row.get("text") or _target_text_from_conversation(row)))
        rationale = html.escape(str(row.get("annotation_rationale", "")))
        verification = html.escape(str(row.get("verification_reason", "")))
        rows_html.append(
            f"""
            <article class="sample">
              <header>
                <strong>{source}</strong>
                <span>retrieval {score}</span>
                <span>annotation {ann}</span>
              </header>
              <blockquote>{text}</blockquote>
              <p><b>Annotation:</b> {rationale}</p>
              <p><b>Verification:</b> {verification}</p>
            </article>
            """
        )

    counts_html = "".join(
        f"<li>{html.escape(source)}: {count}</li>" for source, count in sorted(source_counts.items())
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Wild Delusion Miner Inspection</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.45; color: #182026; }}
h1 {{ font-size: 24px; margin-bottom: 4px; }}
.meta {{ color: #5b6670; margin-top: 0; }}
.sample {{ border: 1px solid #d7dde2; border-radius: 8px; padding: 16px; margin: 18px 0; }}
.sample header {{ display: flex; gap: 12px; flex-wrap: wrap; color: #34424e; }}
blockquote {{ white-space: pre-wrap; margin: 12px 0; padding: 12px; background: #f6f8fa; border-left: 4px solid #8aa1b4; }}
p {{ margin: 8px 0; }}
</style>
</head>
<body>
<h1>Wild Delusion Miner Inspection</h1>
<p class="meta">Source file: {html.escape(str(source_path))}. Showing up to {per_source} samples per source.</p>
<h2>Candidate Counts In Review File</h2>
<ul>{counts_html}</ul>
{''.join(rows_html)}
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return len(review)


def _target_text_from_conversation(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    index = row.get("target_message_index")
    if not isinstance(messages, list) or index is None:
        return ""
    try:
        message = messages[int(index)]
    except (IndexError, TypeError, ValueError):
        return ""
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return ""
