#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from wild_delusion_miner.config import ensure_dirs, load_config
from wild_delusion_miner.retrieval import build_query_from_means, score_corpus
from wild_delusion_miner.store import DedupeStore


BUCKETS = (
    ("rank_000001_000500", 1, 500),
    ("rank_000501_002000", 501, 2_000),
    ("rank_002001_010000", 2_001, 10_000),
    ("rank_010001_050000", 10_001, 50_000),
    ("rank_050001_500000", 50_001, 500_000),
)


@dataclass(frozen=True)
class ExperimentHparams:
    experiment_name: str
    source_retrieval_name: str
    source_top_k: int
    annotation_model: str
    mini_eval_model: str
    top_k: int
    samples_per_bucket: int
    seed: int
    whitening_sample_rows: int
    whitening_eps: float
    whiten_normalize_rows: bool


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} {message}", flush=True)


def write_hparams(out_dir: Path, hparams: ExperimentHparams) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hparams.json").write_text(
        json.dumps(asdict(hparams), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_manifest(embedding_dir: Path) -> dict[str, Any]:
    return json.loads((embedding_dir / "manifest.json").read_text(encoding="utf-8"))


def _join_text_and_refs(config, rows: pd.DataFrame) -> pd.DataFrame:
    hashes = rows["message_hash"].astype(str).tolist()
    store = DedupeStore(config.paths.sqlite_path)
    try:
        texts = store.get_messages(hashes)
        refs = store.get_first_refs(hashes)
    finally:
        store.close()

    out_rows = []
    for row in rows.itertuples(index=False):
        message_hash = str(row.message_hash)
        source, split, row_offset, conversation_id, message_index = refs.get(
            message_hash, ("", "", -1, "", -1)
        )
        payload = row._asdict()
        payload.update(
            {
                "message_hash": message_hash,
                "text": texts.get(message_hash, ""),
                "source": source,
                "split": split,
                "row_offset": row_offset,
                "conversation_id": conversation_id,
                "message_index": message_index,
            }
        )
        out_rows.append(payload)
    return pd.DataFrame(out_rows)


def export_top_candidates(config, *, source_retrieval_name: str, top_k: int, out_path: Path) -> int:
    top_path = config.paths.retrieval_dir / source_retrieval_name / "top.parquet"
    top = pq.read_table(top_path).to_pandas()
    top = top.sort_values("score", ascending=False).drop_duplicates("message_hash").head(top_k)
    top = top.reset_index(drop=True)
    top["rank"] = top.index + 1
    top = top.rename(columns={"score": "retrieval_score"})
    out = _join_text_and_refs(config, top)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_json(out_path, orient="records", lines=True, force_ascii=False)
    return len(out)


def _iter_embedding_shards(embedding_dir: Path):
    manifest = _load_manifest(embedding_dir)
    for shard in manifest["shards"]:
        embeddings = np.load(embedding_dir / shard["embedding_path"]).astype(np.float32)
        metadata = pq.read_table(embedding_dir / shard["metadata_path"]).to_pandas()
        yield shard, embeddings, metadata


def collect_positive_embeddings(config, *, annotations_path: Path, out_dir: Path) -> int:
    annotations = pd.read_json(annotations_path, lines=True)
    positives = annotations[annotations["is_positive"] == True].copy()  # noqa: E712
    positive_hashes = set(positives["message_hash"].astype(str))
    if not positive_hashes:
        raise ValueError(f"No positives in {annotations_path}")

    vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for _, embeddings, metadata in _iter_embedding_shards(config.paths.corpus_embedding_dir):
        metadata = metadata.copy()
        metadata["message_hash"] = metadata["message_hash"].astype(str)
        mask = metadata["message_hash"].isin(positive_hashes).to_numpy()
        if not mask.any():
            continue
        vectors.append(embeddings[mask])
        rows.extend(metadata.loc[mask, ["message_hash"]].to_dict(orient="records"))

    if not vectors:
        raise ValueError("Could not find positive hashes in materialized embedding shards.")

    matrix = np.concatenate(vectors, axis=0).astype(np.float32)
    positive_mean = matrix.mean(axis=0)
    positive_mean /= max(float(np.linalg.norm(positive_mean)), 1e-12)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "positive_embeddings.npy", matrix.astype(np.float16))
    np.save(out_dir / "positive_mean.npy", positive_mean.astype(np.float32))
    pd.DataFrame(rows).to_parquet(out_dir / "positive_embedding_rows.parquet", index=False)
    positives.to_json(out_dir / "positive_annotations.jsonl", orient="records", lines=True, force_ascii=False)
    (out_dir / "positive_manifest.json").write_text(
        json.dumps(
            {
                "annotations_path": str(annotations_path),
                "annotation_rows": int(len(annotations)),
                "positive_rows": int(len(positives)),
                "positive_embeddings_found": int(len(matrix)),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return int(len(matrix))


def fit_whitening(config, *, out_dir: Path, sample_rows: int, seed: int, eps: float) -> None:
    rng = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    seen = 0
    for _, embeddings, _ in _iter_embedding_shards(config.paths.corpus_embedding_dir):
        remaining = max(0, sample_rows - sum(len(x) for x in sampled))
        if remaining <= 0:
            break
        take = min(len(embeddings), max(1, remaining))
        if take < len(embeddings):
            indices = rng.choice(len(embeddings), size=take, replace=False)
            sampled.append(embeddings[indices].astype(np.float32))
        else:
            sampled.append(embeddings.astype(np.float32))
        seen += len(embeddings)

    sample = np.concatenate(sampled, axis=0)
    if len(sample) > sample_rows:
        sample = sample[rng.choice(len(sample), size=sample_rows, replace=False)]
    mean = sample.mean(axis=0)
    centered = sample - mean
    cov = (centered.T @ centered) / max(1, len(centered) - 1)
    values, vectors = np.linalg.eigh(cov)
    values = np.maximum(values, eps)
    whitening = (vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "whitening_mean.npy", mean.astype(np.float32))
    np.save(out_dir / "whitening_matrix.npy", whitening)
    (out_dir / "whitening_manifest.json").write_text(
        json.dumps(
            {
                "sample_rows_requested": sample_rows,
                "sample_rows_used": int(len(sample)),
                "seen_before_stop": int(seen),
                "seed": seed,
                "eps": eps,
                "min_eigenvalue": float(values.min()),
                "max_eigenvalue": float(values.max()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def score_whitened(
    config,
    *,
    positive_mean_path: Path,
    whitening_dir: Path,
    out_name: str,
    top_k: int,
    normalize_rows: bool,
) -> Path:
    embedding_dir = config.paths.corpus_embedding_dir
    retrieval_dir = config.paths.retrieval_dir / out_name
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    positive_mean = np.load(positive_mean_path).astype(np.float32)
    whitening_mean = np.load(whitening_dir / "whitening_mean.npy").astype(np.float32)
    whitening = np.load(whitening_dir / "whitening_matrix.npy").astype(np.float32)
    query = (positive_mean - whitening_mean) @ whitening
    query /= max(float(np.linalg.norm(query)), 1e-12)
    np.save(retrieval_dir / "query.npy", query.astype(np.float32))

    heap: list[tuple[float, str]] = []
    writer: pq.ParquetWriter | None = None
    scores_path = retrieval_dir / "scores.parquet"
    total = 0
    for _, embeddings, metadata in _iter_embedding_shards(embedding_dir):
        transformed = (embeddings - whitening_mean) @ whitening
        if normalize_rows:
            norms = np.linalg.norm(transformed, axis=1, keepdims=True)
            transformed = transformed / np.maximum(norms, 1e-12)
        scores = transformed @ query
        hashes = metadata["message_hash"].astype(str).tolist()
        table = pa.table({"message_hash": hashes, "score": scores.astype(np.float32)})
        if writer is None:
            writer = pq.ParquetWriter(scores_path, table.schema)
        writer.write_table(table)
        for message_hash, score in zip(hashes, scores):
            item = (float(score), message_hash)
            if len(heap) < top_k:
                import heapq

                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                import heapq

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
        json.dumps(
            {
                "method": "positive_mean_whitened",
                "positive_mean_path": str(positive_mean_path),
                "whitening_dir": str(whitening_dir),
                "rows_scored": total,
                "top_k": top_k,
                "normalize_rows": normalize_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return retrieval_dir


def score_positive_raw(config, *, positive_mean_path: Path, out_name: str, top_k: int) -> Path:
    retrieval_dir = config.paths.retrieval_dir / out_name
    query_path = retrieval_dir / "query.npy"
    build_query_from_means(
        config.paths.corpus_embedding_dir / "document_mean.npy",
        positive_mean_path,
        query_path,
    )
    return score_corpus(config, query_path=query_path, out_name=out_name, top_k=top_k)


def sample_retrieval(config, *, retrieval_name: str, out_path: Path, samples_per_bucket: int, seed: int) -> int:
    top = pq.read_table(config.paths.retrieval_dir / retrieval_name / "top.parquet").to_pandas()
    top = top.sort_values("score", ascending=False).drop_duplicates("message_hash").reset_index(drop=True)
    top["rank"] = top.index + 1
    frames = []
    manifest = []
    for bucket, start, end in BUCKETS:
        frame = top[(top["rank"] >= start) & (top["rank"] <= end)].copy()
        if frame.empty:
            continue
        sample = frame.sample(n=min(samples_per_bucket, len(frame)), random_state=seed)
        sample["bucket"] = bucket
        sample["bucket_rank_start"] = start
        sample["bucket_rank_end"] = end
        sample["retrieval_name"] = retrieval_name
        frames.append(sample)
        manifest.append(
            {
                "bucket": bucket,
                "rank_start": start,
                "rank_end": end,
                "sampled": int(len(sample)),
                "available": int(len(frame)),
                "min_score": float(frame["score"].min()),
                "max_score": float(frame["score"].max()),
            }
        )
    sampled = pd.concat(frames, ignore_index=True).sort_values(["bucket_rank_start", "rank"])
    sampled = sampled.rename(columns={"score": "retrieval_score"})
    out = _join_text_and_refs(config, sampled)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_json(out_path, orient="records", lines=True, force_ascii=False)
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "retrieval_name": retrieval_name,
                "samples_per_bucket": samples_per_bucket,
                "seed": seed,
                "buckets": manifest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return len(out)


def summarize_annotations(path: Path, out_path: Path) -> None:
    frame = pd.read_json(path, lines=True)
    summary = (
        frame.groupby(["retrieval_name", "bucket"], sort=False)
        .agg(
            n=("message_hash", "count"),
            positives=("is_positive", "sum"),
            hit_rate=("is_positive", "mean"),
            min_score=("retrieval_score", "min"),
            max_score=("retrieval_score", "max"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
            errors=("annotation_error", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Positive-bootstrap embedding mechanism experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--experiment-name", default="positive_bootstrap_v1")
    parser.add_argument("--source-retrieval-name", default="hypothetical_bucket_calibration")
    parser.add_argument("--source-top-k", type=int, default=1_000)
    parser.add_argument("--annotation-model", default="gpt-5.5")
    parser.add_argument("--mini-eval-model", default="gpt-5.4-mini")
    parser.add_argument("--top-k", type=int, default=500_000)
    parser.add_argument("--samples-per-bucket", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--whitening-sample-rows", type=int, default=200_000)
    parser.add_argument("--whitening-eps", type=float, default=1e-5)
    parser.add_argument("--whiten-normalize-rows", action="store_true", default=True)
    parser.add_argument(
        "--stage",
        choices=["export-top", "build-means", "score-raw", "score-whitened", "sample", "summarize"],
        required=True,
    )
    parser.add_argument("--annotations-path", type=Path)
    parser.add_argument("--sample-annotations-path", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    hparams = ExperimentHparams(
        experiment_name=args.experiment_name,
        source_retrieval_name=args.source_retrieval_name,
        source_top_k=args.source_top_k,
        annotation_model=args.annotation_model,
        mini_eval_model=args.mini_eval_model,
        top_k=args.top_k,
        samples_per_bucket=args.samples_per_bucket,
        seed=args.seed,
        whitening_sample_rows=args.whitening_sample_rows,
        whitening_eps=args.whitening_eps,
        whiten_normalize_rows=args.whiten_normalize_rows,
    )
    exp_dir = config.paths.annotation_dir / "experiments" / args.experiment_name
    write_hparams(exp_dir, hparams)

    if args.stage == "export-top":
        count = export_top_candidates(
            config,
            source_retrieval_name=args.source_retrieval_name,
            top_k=args.source_top_k,
            out_path=exp_dir / "top_candidates.jsonl",
        )
        log(f"exported top candidates count={count} path={exp_dir / 'top_candidates.jsonl'}")
        return

    positive_dir = exp_dir / "positive_embeddings"
    if args.stage == "build-means":
        if args.annotations_path is None:
            raise ValueError("--annotations-path is required for build-means")
        count = collect_positive_embeddings(config, annotations_path=args.annotations_path, out_dir=positive_dir)
        log(f"collected positive embeddings count={count} path={positive_dir}")
        return

    if args.stage == "score-raw":
        out = score_positive_raw(
            config,
            positive_mean_path=positive_dir / "positive_mean.npy",
            out_name=f"{args.experiment_name}_positive_raw",
            top_k=args.top_k,
        )
        log(f"scored raw positive retrieval path={out}")
        return

    if args.stage == "score-whitened":
        whitening_dir = exp_dir / "whitening"
        if not (whitening_dir / "whitening_matrix.npy").exists():
            log("fitting whitening transform")
            fit_whitening(
                config,
                out_dir=whitening_dir,
                sample_rows=args.whitening_sample_rows,
                seed=args.seed,
                eps=args.whitening_eps,
            )
        out = score_whitened(
            config,
            positive_mean_path=positive_dir / "positive_mean.npy",
            whitening_dir=whitening_dir,
            out_name=f"{args.experiment_name}_positive_whitened",
            top_k=args.top_k,
            normalize_rows=args.whiten_normalize_rows,
        )
        log(f"scored whitened positive retrieval path={out}")
        return

    if args.stage == "sample":
        rows = []
        for retrieval_name in (
            args.source_retrieval_name,
            f"{args.experiment_name}_positive_raw",
            f"{args.experiment_name}_positive_whitened",
        ):
            out_path = exp_dir / f"{retrieval_name}_mini_sample.jsonl"
            count = sample_retrieval(
                config,
                retrieval_name=retrieval_name,
                out_path=out_path,
                samples_per_bucket=args.samples_per_bucket,
                seed=args.seed,
            )
            rows.append({"retrieval_name": retrieval_name, "path": str(out_path), "rows": count})
        combined = []
        for row in rows:
            combined.extend(pd.read_json(row["path"], lines=True).to_dict(orient="records"))
        pd.DataFrame(combined).to_json(
            exp_dir / "combined_mini_eval_sample.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        (exp_dir / "sample_manifest.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        log(f"wrote combined mini eval sample rows={len(combined)} path={exp_dir / 'combined_mini_eval_sample.jsonl'}")
        return

    if args.stage == "summarize":
        if args.sample_annotations_path is None:
            raise ValueError("--sample-annotations-path is required for summarize")
        summarize_annotations(args.sample_annotations_path, exp_dir / "mini_eval_summary.jsonl")
        return


if __name__ == "__main__":
    main()
