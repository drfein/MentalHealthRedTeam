#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from openai import OpenAI

from wild_delusion_miner.config import ensure_dirs, load_config
from wild_delusion_miner.embed import (
    materialize_corpus_embeddings,
    materialize_synthetic_embeddings,
    refresh_embedding_batches,
)
from wild_delusion_miner.retrieval import build_query_from_means, score_corpus
from wild_delusion_miner.store import DedupeStore

TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} {message}", flush=True)


def batch_status(manifest: dict[str, Any]) -> dict[str, Any]:
    batches = manifest.get("batches", [])
    return {
        "batch_count": len(batches),
        "statuses": dict(Counter(batch.get("status") for batch in batches)),
        "missing_outputs": sum(
            1
            for batch in batches
            if batch.get("status") == "completed" and not batch.get("output_path")
        ),
        "failed_requests": sum(
            int((batch.get("request_counts") or {}).get("failed") or 0) for batch in batches
        ),
        "total_requests": sum(
            int((batch.get("request_counts") or {}).get("total") or 0) for batch in batches
        ),
    }


def _load_manifest(batch_dir: Path) -> dict[str, Any]:
    path = batch_dir / "batch_manifest.json"
    if not path.exists():
        return {"batches": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _has_terminal_statuses(manifest: dict[str, Any]) -> bool:
    statuses = {str(batch.get("status")) for batch in manifest.get("batches", [])}
    return bool(statuses) and statuses.issubset(TERMINAL_BATCH_STATUSES)


def wait_for_terminal_batches(config, *, batch_dir: Path, poll_seconds: int) -> dict[str, Any]:
    cached = _load_manifest(batch_dir)
    if _has_terminal_statuses(cached):
        log(f"cached batch status: {json.dumps(batch_status(cached), sort_keys=True)}")
        return cached

    while True:
        manifest = refresh_embedding_batches(config, batch_dir=batch_dir)
        status = batch_status(manifest)
        log(f"batch status: {json.dumps(status, sort_keys=True)}")
        if _has_terminal_statuses(manifest):
            return manifest
        time.sleep(poll_seconds)


def download_until_complete(config, *, batch_dir: Path, poll_seconds: int) -> dict[str, Any]:
    while True:
        manifest = download_missing_outputs(config, batch_dir=batch_dir, max_workers=4)
        status = batch_status(manifest)
        log(f"download status: {json.dumps(status, sort_keys=True)}")
        if not status["missing_outputs"]:
            return manifest
        time.sleep(poll_seconds)


def _save_manifest(batch_dir: Path, manifest: dict[str, Any]) -> None:
    path = batch_dir / "batch_manifest.json"
    tmp_path = path.with_suffix(f"{path.suffix}.tmp.{time.time_ns()}")
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _download_file(file_id: str, out_path: Path) -> tuple[str, int]:
    client = OpenAI(timeout=300, max_retries=3)
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{time.time_ns()}")
    try:
        with client.files.with_streaming_response.content(file_id) as response:
            response.stream_to_file(tmp_path)
        tmp_path.replace(out_path)
        return str(out_path), out_path.stat().st_size
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def download_missing_outputs(
    config,
    *,
    batch_dir: Path,
    max_workers: int,
) -> dict[str, Any]:
    manifest = _load_manifest(batch_dir)
    output_dir = batch_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    work: list[tuple[int, str, Path]] = []
    missing_output_ids = 0
    for item in manifest["batches"]:
        output_file_id = item.get("output_file_id")
        output_path = output_dir / f"output-{int(item['index']):06d}.jsonl"
        if item.get("output_path") and Path(item["output_path"]).exists():
            continue
        if output_path.exists():
            item["output_path"] = str(output_path)
            continue
        if not output_file_id:
            missing_output_ids += 1
            continue
        work.append((int(item["index"]), str(output_file_id), output_path))

    _save_manifest(batch_dir, manifest)
    if missing_output_ids:
        log(f"refreshing because completed batches lack output file ids count={missing_output_ids}")
        manifest = refresh_embedding_batches(config, batch_dir=batch_dir)
        return manifest
    if not work:
        return manifest

    log(f"downloading missing output files count={len(work)} workers={max_workers}")
    index_to_item = {int(item["index"]): item for item in manifest["batches"]}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_file, file_id, output_path): (index, output_path)
            for index, file_id, output_path in work
        }
        completed = 0
        for future in as_completed(futures):
            index, output_path = futures[future]
            completed += 1
            try:
                path, size = future.result()
            except Exception as exc:
                log(f"download failed index={index} path={output_path} error={exc!r}")
                continue
            index_to_item[index]["output_path"] = path
            _save_manifest(batch_dir, manifest)
            log(f"downloaded index={index} bytes={size} progress={completed}/{len(work)}")
    return manifest


def export_top_with_text(config, *, retrieval_dir: Path, out_path: Path, limit: int) -> int:
    top = pq.read_table(retrieval_dir / "top.parquet").to_pandas().head(limit)
    hashes = top["message_hash"].astype(str).tolist()
    store = DedupeStore(config.paths.sqlite_path)
    try:
        texts = store.get_messages(hashes)
        refs = store.get_first_refs(hashes)
    finally:
        store.close()

    rows = []
    for row in top.itertuples(index=False):
        message_hash = str(row.message_hash)
        source, split, row_offset, conversation_id, message_index = refs.get(
            message_hash, ("", "", -1, "", -1)
        )
        rows.append(
            {
                "rank": len(rows) + 1,
                "message_hash": message_hash,
                "score": float(row.score),
                "source": source,
                "split": split,
                "row_offset": row_offset,
                "conversation_id": conversation_id,
                "message_index": message_index,
                "text": texts.get(message_hash, ""),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(out_path, orient="records", lines=True, force_ascii=False)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize OpenAI embedding batches and run synthetic-minus-mean retrieval."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--retrieval-name", default="hypothetical_minus_corpus_mean")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-text-rows", type=int, default=5000)
    parser.add_argument("--skip-materialize", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    corpus_batch_dir = config.paths.corpus_embedding_dir / "batches"
    synthetic_batch_dir = config.paths.synthetic_embedding_dir / "batches"

    log("refreshing corpus batches")
    wait_for_terminal_batches(config, batch_dir=corpus_batch_dir, poll_seconds=args.poll_seconds)
    download_until_complete(config, batch_dir=corpus_batch_dir, poll_seconds=args.poll_seconds)

    log("refreshing synthetic batches")
    wait_for_terminal_batches(config, batch_dir=synthetic_batch_dir, poll_seconds=args.poll_seconds)
    download_until_complete(config, batch_dir=synthetic_batch_dir, poll_seconds=args.poll_seconds)

    if args.skip_materialize:
        log("skipping materialization by request")
    else:
        corpus_rows = materialize_corpus_embeddings(config)
        log(f"materialized corpus rows={corpus_rows}")
        synthetic_rows = materialize_synthetic_embeddings(config)
        log(f"materialized synthetic rows={synthetic_rows}")

    query_path = config.paths.retrieval_dir / args.retrieval_name / "query.npy"
    build_query_from_means(
        config.paths.corpus_embedding_dir / "document_mean.npy",
        config.paths.synthetic_embedding_dir / "mean.npy",
        query_path,
    )
    log(f"wrote query={query_path}")

    top_k = args.top_k or config.retrieval.initial_top_k
    retrieval_dir = score_corpus(
        config,
        query_path=query_path,
        out_name=args.retrieval_name,
        top_k=top_k,
    )
    log(f"scored corpus into {retrieval_dir}")

    top_text_path = retrieval_dir / f"top{args.top_text_rows}_with_text.jsonl"
    exported = export_top_with_text(
        config,
        retrieval_dir=retrieval_dir,
        out_path=top_text_path,
        limit=args.top_text_rows,
    )
    log(f"exported top rows with text={exported} path={top_text_path}")


if __name__ == "__main__":
    main()
