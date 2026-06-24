from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from openai import OpenAI
from tqdm import tqdm

from wild_delusion_miner.config import PipelineConfig
from wild_delusion_miner.jsonl import read_jsonl
from wild_delusion_miner.store import DedupeStore
from wild_delusion_miner.text import dedupe_key, normalize_text


BatchSourceRow = tuple[str, str, int | None]


def _embedding_body(config: PipelineConfig, text: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": config.models.embedding_model,
        "input": text,
        "encoding_format": "float",
    }
    if config.embedding.dimensions is not None:
        body["dimensions"] = int(config.embedding.dimensions)
    return body


def _batch_request(custom_id: str, config: PipelineConfig, text: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/embeddings",
        "body": _embedding_body(config, text),
    }


def _write_batch_file(path: Path, rows: Iterable[tuple[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for custom_id, request_json in rows:
            handle.write(request_json)
            handle.write("\n")
            count += 1
    return count


def _manifest_path(out_dir: Path) -> Path:
    return out_dir / "batch_manifest.json"


def _load_batch_manifest(out_dir: Path) -> dict[str, Any]:
    path = _manifest_path(out_dir)
    if not path.exists():
        return {"batches": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_batch_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _manifest_path(out_dir).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def prepare_corpus_embedding_batches(config: PipelineConfig, *, limit_messages: int | None = None) -> int:
    store = DedupeStore(config.paths.sqlite_path)
    out_dir = config.paths.corpus_embedding_dir / "batches"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "kind": "corpus",
        "model": config.models.embedding_model,
        "endpoint": "/v1/embeddings",
        "batches": [],
    }

    request_limit = config.embedding.batch_request_size
    batch_index = 0
    request_rows: list[tuple[str, str]] = []
    total = 0
    try:
        for db_batch in store.iter_messages(batch_size=10_000):
            for rowid, message_hash, text in db_batch:
                if limit_messages is not None and total >= limit_messages:
                    break
                custom_id = f"corpus:{rowid}:{message_hash}"
                request_rows.append(
                    (
                        custom_id,
                        json.dumps(
                            _batch_request(custom_id, config, text),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                )
                total += 1
                if len(request_rows) >= request_limit:
                    path = out_dir / f"input-{batch_index:06d}.jsonl"
                    count = _write_batch_file(path, request_rows)
                    manifest["batches"].append(
                        {"index": batch_index, "input_path": str(path), "request_count": count}
                    )
                    batch_index += 1
                    request_rows = []
            if limit_messages is not None and total >= limit_messages:
                break
        if request_rows:
            path = out_dir / f"input-{batch_index:06d}.jsonl"
            count = _write_batch_file(path, request_rows)
            manifest["batches"].append(
                {"index": batch_index, "input_path": str(path), "request_count": count}
            )
    finally:
        store.close()

    manifest["request_count"] = total
    _save_batch_manifest(out_dir, manifest)
    return total


def prepare_synthetic_embedding_batches(config: PipelineConfig) -> int:
    rows = list(read_jsonl(config.paths.synthetic_path))
    source_rows = [
        (f"synthetic:{int(row['synthetic_id'])}", str(row["text"]), int(row["synthetic_id"]))
        for row in rows
    ]
    return _prepare_small_embedding_batches(
        config,
        source_rows,
        out_dir=config.paths.synthetic_embedding_dir / "batches",
        kind="synthetic",
    )


def prepare_text_embedding_batches(
    config: PipelineConfig,
    texts: Iterable[str],
    *,
    out_dir: Path,
    kind: str,
) -> int:
    rows = []
    for index, text in enumerate(texts):
        normalized = normalize_text(text)
        if not normalized:
            continue
        rows.append((f"{kind}:{index}:{dedupe_key(normalized)}", normalized, index))
    return _prepare_small_embedding_batches(config, rows, out_dir=out_dir, kind=kind)


def _prepare_small_embedding_batches(
    config: PipelineConfig,
    rows: list[tuple[str, str, int]],
    *,
    out_dir: Path,
    kind: str,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "kind": kind,
        "model": config.models.embedding_model,
        "endpoint": "/v1/embeddings",
        "batches": [],
    }
    request_limit = config.embedding.batch_request_size
    for batch_index, start in enumerate(range(0, len(rows), request_limit)):
        chunk = rows[start : start + request_limit]
        path = out_dir / f"input-{batch_index:06d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for custom_id, text, _index in chunk:
                handle.write(
                    json.dumps(
                        _batch_request(custom_id, config, text),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        manifest["batches"].append(
            {"index": batch_index, "input_path": str(path), "request_count": len(chunk)}
        )
    manifest["request_count"] = len(rows)
    _save_batch_manifest(out_dir, manifest)
    return len(rows)


def submit_embedding_batches(config: PipelineConfig, *, batch_dir: Path) -> dict[str, Any]:
    client = OpenAI()
    manifest = _load_batch_manifest(batch_dir)
    for item in manifest["batches"]:
        if item.get("batch_id"):
            continue
        with Path(item["input_path"]).open("rb") as handle:
            upload = client.files.create(file=handle, purpose="batch")
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={"kind": str(manifest.get("kind", "embedding"))},
        )
        item["input_file_id"] = upload.id
        item["batch_id"] = batch.id
        item["status"] = batch.status
        _save_batch_manifest(batch_dir, manifest)
    return manifest


def refresh_embedding_batches(config: PipelineConfig, *, batch_dir: Path) -> dict[str, Any]:
    client = OpenAI()
    manifest = _load_batch_manifest(batch_dir)
    for item in manifest["batches"]:
        batch_id = item.get("batch_id")
        if not batch_id:
            continue
        batch = client.batches.retrieve(str(batch_id))
        item["status"] = batch.status
        item["output_file_id"] = batch.output_file_id
        item["error_file_id"] = batch.error_file_id
        item["request_counts"] = (
            batch.request_counts.model_dump()
            if hasattr(batch.request_counts, "model_dump") and batch.request_counts
            else None
        )
    _save_batch_manifest(batch_dir, manifest)
    return manifest


def wait_for_embedding_batches(config: PipelineConfig, *, batch_dir: Path) -> dict[str, Any]:
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        manifest = refresh_embedding_batches(config, batch_dir=batch_dir)
        statuses = {str(item.get("status")) for item in manifest["batches"]}
        if statuses and statuses.issubset(terminal):
            return manifest
        time.sleep(config.embedding.poll_seconds)


def download_embedding_batch_outputs(config: PipelineConfig, *, batch_dir: Path) -> dict[str, Any]:
    client = OpenAI()
    manifest = refresh_embedding_batches(config, batch_dir=batch_dir)
    output_dir = batch_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest["batches"]:
        output_file_id = item.get("output_file_id")
        if output_file_id and not item.get("output_path"):
            content = client.files.content(str(output_file_id)).read()
            output_path = output_dir / f"output-{int(item['index']):06d}.jsonl"
            output_path.write_bytes(content)
            item["output_path"] = str(output_path)
        error_file_id = item.get("error_file_id")
        if error_file_id and not item.get("error_path"):
            content = client.files.content(str(error_file_id)).read()
            error_path = output_dir / f"errors-{int(item['index']):06d}.jsonl"
            error_path.write_bytes(content)
            item["error_path"] = str(error_path)
    _save_batch_manifest(batch_dir, manifest)
    return manifest


def _iter_batch_embedding_outputs(batch_dir: Path) -> Iterable[tuple[str, np.ndarray]]:
    manifest = _load_batch_manifest(batch_dir)
    for item in manifest["batches"]:
        output_path = item.get("output_path")
        if not output_path:
            continue
        with Path(output_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("error"):
                    continue
                response = payload.get("response") or {}
                body = response.get("body") or {}
                data = body.get("data") or []
                if not data:
                    continue
                embedding = np.asarray(data[0]["embedding"], dtype=np.float32)
                yield str(payload["custom_id"]), embedding


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def materialize_corpus_embeddings(config: PipelineConfig) -> int:
    batch_dir = config.paths.corpus_embedding_dir / "batches"
    out_dir = config.paths.corpus_embedding_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_index = 0
    ids: list[str] = []
    rowids: list[int] = []
    embeddings: list[np.ndarray] = []
    mean_sum: np.ndarray | None = None
    total = 0
    manifest: dict[str, Any] = {
        "model": config.models.embedding_model,
        "normalize": config.embedding.normalize,
        "shards": [],
        "rows": 0,
        "source": "openai_batch",
    }

    for custom_id, embedding in tqdm(_iter_batch_embedding_outputs(batch_dir), desc="materializing corpus"):
        _kind, rowid, message_hash = custom_id.split(":", 2)
        ids.append(message_hash)
        rowids.append(int(rowid))
        embeddings.append(embedding)
        total += 1
        if len(embeddings) >= config.embedding.shard_size:
            shard = _write_embedding_shard(
                out_dir, shard_index, ids, rowids, embeddings, config.embedding.normalize, config.embedding.dtype
            )
            manifest["shards"].append(shard)
            shard_array = np.load(out_dir / shard["embedding_path"]).astype(np.float32)
            mean_sum = shard_array.sum(axis=0) if mean_sum is None else mean_sum + shard_array.sum(axis=0)
            shard_index += 1
            ids, rowids, embeddings = [], [], []

    if embeddings:
        shard = _write_embedding_shard(
            out_dir, shard_index, ids, rowids, embeddings, config.embedding.normalize, config.embedding.dtype
        )
        manifest["shards"].append(shard)
        shard_array = np.load(out_dir / shard["embedding_path"]).astype(np.float32)
        mean_sum = shard_array.sum(axis=0) if mean_sum is None else mean_sum + shard_array.sum(axis=0)

    manifest["rows"] = total
    if total and mean_sum is not None:
        mean = (mean_sum / total).astype(np.float32)
        np.save(out_dir / "document_mean.npy", mean)
        manifest["document_mean_path"] = "document_mean.npy"
        manifest["dim"] = int(mean.shape[0])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return total


def materialize_synthetic_embeddings(config: PipelineConfig) -> int:
    rows = list(read_jsonl(config.paths.synthetic_path))
    text_by_id = {int(row["synthetic_id"]): str(row["text"]) for row in rows}
    theme_by_id = {int(row["synthetic_id"]): str(row.get("theme") or "") for row in rows}
    output_rows = []
    embeddings = []
    for custom_id, embedding in _iter_batch_embedding_outputs(config.paths.synthetic_embedding_dir / "batches"):
        _kind, synthetic_id_raw = custom_id.split(":", 1)
        synthetic_id = int(synthetic_id_raw)
        output_rows.append((synthetic_id, text_by_id.get(synthetic_id, ""), theme_by_id.get(synthetic_id, "")))
        embeddings.append(embedding)
    return _materialize_small_embeddings(
        config,
        embeddings,
        output_rows,
        out_dir=config.paths.synthetic_embedding_dir,
        id_column="synthetic_id",
        mean_name="mean.npy",
    )


def materialize_text_mean_embeddings(
    config: PipelineConfig,
    *,
    batch_dir: Path,
    out_path: Path,
) -> int:
    embeddings = [embedding for _custom_id, embedding in _iter_batch_embedding_outputs(batch_dir)]
    if not embeddings:
        raise ValueError(f"No embeddings found in {batch_dir}")
    array = np.stack(embeddings).astype(np.float32)
    if config.embedding.normalize:
        array = _normalize_rows(array)
    mean = array.mean(axis=0).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mean)
    return int(array.shape[0])


def _materialize_small_embeddings(
    config: PipelineConfig,
    embeddings: list[np.ndarray],
    metadata_rows: list[tuple[int, str, str]],
    *,
    out_dir: Path,
    id_column: str,
    mean_name: str,
) -> int:
    if not embeddings:
        raise ValueError(f"No embeddings found for {out_dir}")
    order = np.argsort([row[0] for row in metadata_rows])
    array = np.stack(embeddings).astype(np.float32)[order]
    if config.embedding.normalize:
        array = _normalize_rows(array)
    metadata_rows = [metadata_rows[int(index)] for index in order]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", array.astype(config.embedding.dtype))
    np.save(out_dir / mean_name, array.mean(axis=0).astype(np.float32))
    pq.write_table(
        pa.table(
            {
                id_column: [row[0] for row in metadata_rows],
                "text": [row[1] for row in metadata_rows],
                "theme": [row[2] for row in metadata_rows],
            }
        ),
        out_dir / "metadata.parquet",
    )
    return int(array.shape[0])


def _write_embedding_shard(
    out_dir: Path,
    shard_index: int,
    ids: list[str],
    rowids: list[int],
    embeddings: list[np.ndarray],
    normalize: bool,
    dtype: str,
) -> dict[str, object]:
    array = np.stack(embeddings).astype(np.float32)
    if normalize:
        array = _normalize_rows(array)
    emb_path = out_dir / f"embeddings-{shard_index:06d}.npy"
    meta_path = out_dir / f"metadata-{shard_index:06d}.parquet"
    np.save(emb_path, array.astype(dtype))
    pq.write_table(
        pa.table(
            {
                "rowid": pa.array(rowids, type=pa.int64()),
                "message_hash": pa.array(ids, type=pa.string()),
            }
        ),
        meta_path,
    )
    return {
        "shard_index": shard_index,
        "embedding_path": emb_path.name,
        "metadata_path": meta_path.name,
        "rows": len(ids),
        "dim": int(array.shape[1]),
        "dtype": dtype,
    }
