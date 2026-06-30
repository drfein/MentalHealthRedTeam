from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from wild_delusion_miner.config import PipelineConfig
from wild_delusion_miner.store import DedupeStore


DEFAULT_RETRIEVAL_NAMES = (
    "inspect_top100_current",
    "inspect_story_negative_alpha_0p25",
    "inspect_story_negative_alpha_0p5",
    "inspect_top100_story_negative",
    "partial_bootstrap",
)


def build_rerank_candidates(
    config: PipelineConfig,
    *,
    out_path: Path,
    retrieval_names: tuple[str, ...] = DEFAULT_RETRIEVAL_NAMES,
    top_per_retrieval: int = 750,
    bin_samples_per_retrieval: int = 10,
    bins: int = 20,
    max_candidates: int = 2_500,
) -> int:
    selected: dict[str, dict[str, Any]] = {}
    score_by_hash: dict[str, dict[str, float]] = {}

    for retrieval_name in retrieval_names:
        retrieval_dir = config.paths.retrieval_dir / retrieval_name
        _add_top_candidates(
            selected,
            score_by_hash,
            retrieval_name=retrieval_name,
            top_path=retrieval_dir / "top.parquet",
            limit=top_per_retrieval,
        )
        _add_score_bin_samples(
            selected,
            score_by_hash,
            retrieval_name=retrieval_name,
            scores_path=retrieval_dir / "scores.parquet",
            samples_per_bin=bin_samples_per_retrieval,
            bins=bins,
        )

    if not selected:
        raise ValueError("No retrieval candidates found.")

    rows = _join_candidate_text_and_refs(config, selected, score_by_hash, retrieval_names)
    frame = pd.DataFrame(rows)
    frame = frame[frame["text"].astype(str).str.len() > 0].copy()
    frame = frame.sort_values("max_retrieval_score", ascending=False, na_position="last")
    frame = frame.head(max_candidates)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_json(out_path, orient="records", lines=True, force_ascii=False)
    return len(frame)


def _add_top_candidates(
    selected: dict[str, dict[str, Any]],
    score_by_hash: dict[str, dict[str, float]],
    *,
    retrieval_name: str,
    top_path: Path,
    limit: int,
) -> None:
    if not top_path.exists():
        return
    top = pq.read_table(top_path).to_pandas().sort_values("score", ascending=False)
    for rank, row in enumerate(top.head(limit).itertuples(index=False), start=1):
        message_hash = str(row.message_hash)
        score = float(row.score)
        entry = selected.setdefault(message_hash, {"message_hash": message_hash, "selection_sources": []})
        entry["selection_sources"].append(f"{retrieval_name}:top:{rank}")
        score_by_hash.setdefault(message_hash, {})[retrieval_name] = score


def _add_score_bin_samples(
    selected: dict[str, dict[str, Any]],
    score_by_hash: dict[str, dict[str, float]],
    *,
    retrieval_name: str,
    scores_path: Path,
    samples_per_bin: int,
    bins: int,
) -> None:
    if not scores_path.exists() or samples_per_bin <= 0:
        return
    scores = pq.read_table(scores_path).to_pandas().drop_duplicates("message_hash")
    if scores.empty:
        return
    scores["score_bin"] = pd.qcut(
        scores["score"],
        q=min(bins, len(scores)),
        labels=False,
        duplicates="drop",
    )
    samples = []
    for bin_id, frame in scores.groupby("score_bin", dropna=True):
        sampled = frame.sample(
            n=min(samples_per_bin, len(frame)),
            random_state=20260624,
        ).copy()
        sampled["score_bin"] = int(bin_id)
        samples.append(sampled)
    if not samples:
        return
    sample = pd.concat(samples).sort_values("score", ascending=False)
    for row in sample.to_dict(orient="records"):
        message_hash = str(row["message_hash"])
        score = float(row["score"])
        bin_id = int(row["score_bin"])
        entry = selected.setdefault(message_hash, {"message_hash": message_hash, "selection_sources": []})
        entry["selection_sources"].append(f"{retrieval_name}:bin:{bin_id}")
        score_by_hash.setdefault(message_hash, {})[retrieval_name] = score


def _join_candidate_text_and_refs(
    config: PipelineConfig,
    selected: dict[str, dict[str, Any]],
    score_by_hash: dict[str, dict[str, float]],
    retrieval_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    store = DedupeStore(config.paths.sqlite_path)
    hashes = list(selected)
    texts = store.get_messages(hashes)
    refs = store.get_first_refs(hashes)
    store.close()

    rows = []
    for message_hash, entry in selected.items():
        scores = score_by_hash.get(message_hash, {})
        source, split, row_offset, conversation_id, message_index = refs.get(
            message_hash, ("", "", -1, "", -1)
        )
        row = {
            "message_hash": message_hash,
            "text": texts.get(message_hash, ""),
            "selection_sources": sorted(set(entry["selection_sources"])),
            "max_retrieval_score": max(scores.values()) if scores else None,
            "source": source,
            "split": split,
            "row_offset": row_offset,
            "conversation_id": conversation_id,
            "message_index": message_index,
        }
        row.update({f"score_{name}": scores.get(name) for name in retrieval_names})
        rows.append(row)
    return rows
