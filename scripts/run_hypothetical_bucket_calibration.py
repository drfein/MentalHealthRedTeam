#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from wild_delusion_miner.config import ensure_dirs, load_config
from wild_delusion_miner.retrieval import build_query_from_means, score_corpus
from wild_delusion_miner.store import DedupeStore


DEFAULT_BUCKETS = (
    ("rank_000001_000500", 1, 500),
    ("rank_000501_002000", 501, 2_000),
    ("rank_002001_010000", 2_001, 10_000),
    ("rank_010001_050000", 10_001, 50_000),
    ("rank_050001_500000", 50_001, 500_000),
)


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} {message}", flush=True)


def sample_rank_buckets(
    config,
    *,
    retrieval_dir: Path,
    out_path: Path,
    samples_per_bucket: int,
    seed: int,
) -> int:
    top = pq.read_table(retrieval_dir / "top.parquet").to_pandas()
    top = top.sort_values("score", ascending=False).drop_duplicates("message_hash").reset_index(drop=True)
    top["rank"] = top.index + 1

    sampled_frames = []
    bucket_manifest = []
    for bucket_name, rank_start, rank_end in DEFAULT_BUCKETS:
        frame = top[(top["rank"] >= rank_start) & (top["rank"] <= rank_end)].copy()
        if frame.empty:
            bucket_manifest.append(
                {
                    "bucket": bucket_name,
                    "rank_start": rank_start,
                    "rank_end": rank_end,
                    "available": 0,
                    "sampled": 0,
                    "min_score": None,
                    "max_score": None,
                }
            )
            continue
        sample = frame.sample(n=min(samples_per_bucket, len(frame)), random_state=seed)
        sample["bucket"] = bucket_name
        sample["bucket_rank_start"] = rank_start
        sample["bucket_rank_end"] = rank_end
        sampled_frames.append(sample)
        bucket_manifest.append(
            {
                "bucket": bucket_name,
                "rank_start": rank_start,
                "rank_end": rank_end,
                "available": int(len(frame)),
                "sampled": int(len(sample)),
                "min_score": float(frame["score"].min()),
                "max_score": float(frame["score"].max()),
            }
        )

    if not sampled_frames:
        raise ValueError(f"No rows available in {retrieval_dir / 'top.parquet'} for bucket sampling.")

    sampled = pd.concat(sampled_frames, ignore_index=True).sort_values(["bucket_rank_start", "rank"])
    hashes = sampled["message_hash"].astype(str).tolist()
    store = DedupeStore(config.paths.sqlite_path)
    try:
        texts = store.get_messages(hashes)
        refs = store.get_first_refs(hashes)
    finally:
        store.close()

    rows = []
    for row in sampled.itertuples(index=False):
        message_hash = str(row.message_hash)
        source, split, row_offset, conversation_id, message_index = refs.get(
            message_hash, ("", "", -1, "", -1)
        )
        rows.append(
            {
                "message_hash": message_hash,
                "text": texts.get(message_hash, ""),
                "retrieval_score": float(row.score),
                "rank": int(row.rank),
                "bucket": str(row.bucket),
                "bucket_rank_start": int(row.bucket_rank_start),
                "bucket_rank_end": int(row.bucket_rank_end),
                "source": source,
                "split": split,
                "row_offset": row_offset,
                "conversation_id": conversation_id,
                "message_index": message_index,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(out_path, orient="records", lines=True, force_ascii=False)
    (out_path.with_suffix(".manifest.json")).write_text(
        json.dumps(
            {
                "retrieval_dir": str(retrieval_dir),
                "samples_per_bucket": samples_per_bucket,
                "seed": seed,
                "buckets": bucket_manifest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return len(rows)


def summarize_annotations(path: Path, out_path: Path) -> None:
    rows = pd.read_json(path, lines=True)
    summary = (
        rows.groupby("bucket", sort=False)
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
    parser = argparse.ArgumentParser(description="Top-heavy bucket calibration for hypothetical retrieval.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--retrieval-name", default="hypothetical_bucket_calibration")
    parser.add_argument("--top-k", type=int, default=500_000)
    parser.add_argument("--samples-per-bucket", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--summarize-annotations", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    retrieval_dir = config.paths.retrieval_dir / args.retrieval_name
    query_path = retrieval_dir / "query.npy"
    sample_path = config.paths.annotation_dir / f"{args.retrieval_name}_bucket_sample.jsonl"

    if args.summarize_annotations:
        summarize_annotations(
            args.summarize_annotations,
            config.paths.annotation_dir / f"{args.retrieval_name}_bucket_summary.jsonl",
        )
        return

    if not args.skip_score:
        build_query_from_means(
            config.paths.corpus_embedding_dir / "document_mean.npy",
            config.paths.synthetic_embedding_dir / "mean.npy",
            query_path,
        )
        log(f"scoring corpus retrieval_name={args.retrieval_name} top_k={args.top_k}")
        score_corpus(config, query_path=query_path, out_name=args.retrieval_name, top_k=args.top_k)
    else:
        if not (retrieval_dir / "top.parquet").exists():
            raise FileNotFoundError(f"Missing {retrieval_dir / 'top.parquet'}")

    count = sample_rank_buckets(
        config,
        retrieval_dir=retrieval_dir,
        out_path=sample_path,
        samples_per_bucket=args.samples_per_bucket,
        seed=args.seed,
    )
    log(f"wrote bucket sample rows={count} path={sample_path}")


if __name__ == "__main__":
    main()
