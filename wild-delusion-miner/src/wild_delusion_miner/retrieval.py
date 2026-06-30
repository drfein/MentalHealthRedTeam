from __future__ import annotations

import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from wild_delusion_miner.config import PipelineConfig
from wild_delusion_miner.store import DedupeStore


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Zero-length query vector.")
    return (vector / norm).astype(np.float32)


def build_query_from_means(document_mean_path: Path, positive_mean_path: Path, out_path: Path) -> np.ndarray:
    document_mean = np.load(document_mean_path).astype(np.float32)
    positive_mean = np.load(positive_mean_path).astype(np.float32)
    query = _normalize(positive_mean - document_mean)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, query)
    return query


def _load_manifest(embedding_dir: Path) -> dict:
    return json.loads((embedding_dir / "manifest.json").read_text(encoding="utf-8"))


def score_corpus(
    config: PipelineConfig,
    *,
    query_path: Path,
    out_name: str,
    top_k: int,
) -> Path:
    embedding_dir = config.paths.corpus_embedding_dir
    retrieval_dir = config.paths.retrieval_dir / out_name
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    query = np.load(query_path).astype(np.float32)
    query = _normalize(query)
    manifest = _load_manifest(embedding_dir)

    heap: list[tuple[float, str]] = []
    writer: pq.ParquetWriter | None = None
    all_scores_path = retrieval_dir / "scores.parquet"
    total = 0
    for shard in manifest["shards"]:
        embeddings = np.load(embedding_dir / shard["embedding_path"]).astype(np.float32)
        scores = embeddings @ query
        metadata = pq.read_table(embedding_dir / shard["metadata_path"]).to_pandas()
        hashes = metadata["message_hash"].astype(str).tolist()
        table = pa.table({"message_hash": hashes, "score": scores.astype(np.float32)})
        if writer is None:
            writer = pq.ParquetWriter(all_scores_path, table.schema)
        writer.write_table(table)

        for message_hash, score in zip(hashes, scores):
            item = (float(score), message_hash)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
        total += len(hashes)
    if writer is not None:
        writer.close()

    top_rows = sorted(heap, reverse=True)
    pq.write_table(
        pa.table(
            {
                "message_hash": [row[1] for row in top_rows],
                "score": [row[0] for row in top_rows],
            }
        ),
        retrieval_dir / "top.parquet",
    )
    (retrieval_dir / "manifest.json").write_text(
        json.dumps({"query_path": str(query_path), "rows_scored": total, "top_k": top_k}, indent=2),
        encoding="utf-8",
    )
    return retrieval_dir


def sample_calibration_set(
    config: PipelineConfig,
    *,
    scores_path: Path,
    out_path: Path,
) -> Path:
    scores = pq.read_table(scores_path).to_pandas()
    scores = scores.drop_duplicates("message_hash")
    bins = min(config.retrieval.calibration_bins, len(scores))
    scores["bin"] = pd.qcut(scores["score"], q=bins, labels=False, duplicates="drop")
    sampled = (
        scores.groupby("bin", group_keys=False)
        .apply(
            lambda frame: frame.sample(
                n=min(config.retrieval.calibration_sample_per_bin, len(frame)),
                random_state=20260624,
            )
        )
        .sort_values("score", ascending=False)
    )
    store = DedupeStore(config.paths.sqlite_path)
    texts = store.get_messages(sampled["message_hash"].astype(str).tolist())
    rows = []
    for row in sampled.itertuples(index=False):
        rows.append(
            {
                "message_hash": str(row.message_hash),
                "score": float(row.score),
                "bin": int(row.bin),
                "text": texts.get(str(row.message_hash), ""),
            }
        )
    store.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(out_path, orient="records", lines=True, force_ascii=False)
    return out_path


def choose_threshold_from_calibration(
    calibration_annotations_path: Path,
    *,
    target_hit_rate: float,
) -> float:
    rows = pd.read_json(calibration_annotations_path, lines=True)
    if rows.empty:
        raise ValueError("No calibration annotations found.")
    grouped = (
        rows.groupby("bin")
        .agg(hit_rate=("is_positive", "mean"), min_score=("retrieval_score", "min"), n=("is_positive", "count"))
        .reset_index()
        .sort_values("min_score", ascending=False)
    )
    candidates = grouped[grouped["hit_rate"] >= target_hit_rate]
    if candidates.empty:
        chosen = grouped.iloc[0]
    else:
        candidates = candidates.assign(distance=(candidates["hit_rate"] - target_hit_rate).abs())
        chosen = candidates.sort_values(["distance", "min_score"], ascending=[True, False]).iloc[0]
    return float(chosen["min_score"])


def export_above_threshold(
    config: PipelineConfig,
    *,
    scores_path: Path,
    threshold: float,
    out_path: Path,
) -> int:
    scores = pq.read_table(scores_path).to_pandas()
    selected = scores[scores["score"] >= threshold].drop_duplicates("message_hash")
    store = DedupeStore(config.paths.sqlite_path)
    texts = store.get_messages(selected["message_hash"].astype(str).tolist())
    refs = store.get_first_refs(selected["message_hash"].astype(str).tolist())
    rows = []
    for row in selected.sort_values("score", ascending=False).itertuples(index=False):
        message_hash = str(row.message_hash)
        source, split, row_offset, conversation_id, message_index = refs.get(
            message_hash, ("", "", -1, "", -1)
        )
        rows.append(
            {
                "message_hash": message_hash,
                "retrieval_score": float(row.score),
                "text": texts.get(message_hash, ""),
                "source": source,
                "split": split,
                "row_offset": row_offset,
                "conversation_id": conversation_id,
                "message_index": message_index,
            }
        )
    store.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(out_path, orient="records", lines=True, force_ascii=False)
    return len(rows)
