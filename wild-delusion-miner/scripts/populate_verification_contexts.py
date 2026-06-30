from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from wild_delusion_miner.config import DatasetSpec, load_config
from wild_delusion_miner.datasets import normalize_role, row_to_conversation
from wild_delusion_miner.text import normalize_text, stringify


SHARECHAT_CONTEXT_ROWS = 24
HF_ROWS_API_MAX_LENGTH = 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--candidates-path", type=Path, required=True)
    parser.add_argument("--base-conversations-path", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config_path)
    specs = {spec.source: spec for spec in config.datasets}
    candidates = pd.read_json(args.candidates_path, lines=True)
    positives = candidates[candidates["is_positive"] == True].copy()  # noqa: E712
    positives = positives.drop_duplicates(["source", "split", "row_offset"])
    base = pd.read_json(args.base_conversations_path, lines=True)

    sharechat = positives[positives["source"].astype(str).str.startswith("sharechat_")]
    expanded_sharechat = _expand_sharechat_contexts(sharechat, specs)

    sharechat_keys = {
        (row["source"], row["split"], int(row["row_offset"]))
        for row in expanded_sharechat
    }
    rows: list[dict[str, Any]] = []
    for row in base.to_dict(orient="records"):
        key = (str(row["source"]), str(row["split"]), int(row["row_offset"]))
        if key not in sharechat_keys:
            rows.append(row)
    rows.extend(expanded_sharechat)
    rows.sort(key=lambda row: (str(row["source"]), str(row["split"]), int(row["row_offset"])))

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} context-populated rows to {args.out_path}", flush=True)


def _expand_sharechat_contexts(
    candidates: pd.DataFrame,
    specs: dict[str, DatasetSpec],
) -> list[dict[str, Any]]:
    if candidates.empty:
        return []
    target_rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for (source, split), group in candidates.groupby(["source", "split"], sort=False):
        spec = _spec_for_source(specs, str(source), str(split))
        offsets = sorted(set(int(value) for value in group["row_offset"]))
        target_rows = _fetch_rows_for_offsets(spec, offsets)
        for offset, row in target_rows.items():
            target_rows_by_key[(str(source), str(split), offset)] = row
        print(
            f"fetched sharechat target rows source={source} split={split} "
            f"targets={len(target_rows)}/{len(offsets)}",
            flush=True,
        )

    context_ranges_by_source: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    target_meta: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in candidates.to_dict(orient="records"):
        key = (str(row["source"]), str(row["split"]), int(row["row_offset"]))
        raw = target_rows_by_key.get(key)
        if not raw:
            continue
        message_index = _int_or_zero(raw.get("message_index"))
        start_offset = max(0, key[2] - message_index)
        opening_end = min(key[2], start_offset + SHARECHAT_CONTEXT_ROWS - 1)
        preceding_start = max(start_offset, key[2] - SHARECHAT_CONTEXT_ROWS + 1)
        context_ranges_by_source[(key[0], key[1])].append((start_offset, opening_end))
        context_ranges_by_source[(key[0], key[1])].append((preceding_start, key[2]))
        target_meta[key] = {
            "candidate": row,
            "conversation_id": _sharechat_conversation_id(raw),
            "message_index": message_index,
        }

    rows_by_source: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for (source, split), ranges in context_ranges_by_source.items():
        spec = _spec_for_source(specs, source, split)
        rows_by_source[(source, split)] = _fetch_rows_for_ranges(spec, ranges)
        print(
            f"fetched sharechat context rows source={source} split={split} "
            f"rows={len(rows_by_source[(source, split)])} ranges={len(_coalesce_ranges(ranges))}",
            flush=True,
        )

    out_rows: list[dict[str, Any]] = []
    for key, meta in target_meta.items():
        source, split, row_offset = key
        context_rows = rows_by_source.get((source, split), {})
        conversation_id = meta["conversation_id"]
        messages_with_offsets = []
        for offset, raw in sorted(context_rows.items()):
            if _sharechat_conversation_id(raw) != conversation_id:
                continue
            role = normalize_role(raw.get("role"))
            content = normalize_text(stringify(raw.get("plain_text") or raw.get("content")))
            if not role or not content:
                continue
            messages_with_offsets.append(
                {
                    "offset": offset,
                    "message_index": _int_or_zero(raw.get("message_index")),
                    "message": {"role": role, "content": content},
                }
            )
        deduped = {
            item["message_index"]: item
            for item in messages_with_offsets
        }
        ordered = [deduped[index] for index in sorted(deduped)]
        target_message_index = next(
            (
                index
                for index, item in enumerate(ordered)
                if item["offset"] == row_offset or item["message_index"] == meta["message_index"]
            ),
            None,
        )
        if target_message_index is None:
            continue
        candidate = meta["candidate"]
        out_rows.append(
            {
                "source": source,
                "split": split,
                "row_offset": row_offset,
                "conversation_id": conversation_id,
                "raw_keys": sorted(str(key) for key in target_rows_by_key[(source, split, row_offset)].keys()),
                "messages": [item["message"] for item in ordered],
                "stable_key": json.dumps([source, split, conversation_id], ensure_ascii=False),
                "message_hash": str(candidate["message_hash"]),
                "target_message_index": target_message_index,
                "retrieval_score": float(candidate["retrieval_score"]),
                "annotation_score": int(candidate["annotation_score"]),
                "annotation_rationale": str(candidate["annotation_rationale"]),
            }
        )
    return out_rows


def _fetch_rows_for_offsets(spec: DatasetSpec, offsets: Iterable[int]) -> dict[int, dict[str, Any]]:
    ranges = [(offset, offset) for offset in offsets]
    return _fetch_rows_for_ranges(spec, ranges)


def _fetch_rows_for_ranges(
    spec: DatasetSpec,
    ranges: Iterable[tuple[int, int]],
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    chunks = _coalesce_ranges(ranges)
    for index, (start, end) in enumerate(chunks, start=1):
        response = _get_hf_rows_api_range(spec, start, end - start + 1)
        for item in response.json().get("rows") or []:
            row_index = int(item.get("row_idx"))
            row = item.get("row")
            if isinstance(row, Mapping):
                rows[row_index] = dict(row)
        if index % 25 == 0 or index == len(chunks):
            print(
                f"range fetch source={spec.source} completed={index}/{len(chunks)} rows={len(rows)}",
                flush=True,
            )
    return rows


def _coalesce_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted((max(0, int(start)), max(0, int(end))) for start, end in ranges):
        if end < start:
            continue
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        new_end = max(prev_end, end)
        if start <= prev_end + 1 and new_end - prev_start + 1 <= HF_ROWS_API_MAX_LENGTH:
            merged[-1] = (prev_start, new_end)
        else:
            merged.append((start, end))
    return merged


def _get_hf_rows_api_range(spec: DatasetSpec, offset: int, length: int) -> requests.Response:
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None
    params = {
        "dataset": spec.name,
        "config": spec.config or "default",
        "split": spec.split,
        "offset": offset,
        "length": min(length, HF_ROWS_API_MAX_LENGTH),
    }
    last_error: requests.RequestException | None = None
    for attempt in range(1, 9):
        try:
            response = requests.get(
                "https://datasets-server.huggingface.co/rows",
                params=params,
                headers=headers,
                timeout=60,
            )
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return response
            retry_after = response.headers.get("retry-after")
            if retry_after and retry_after.isdigit():
                time.sleep(min(int(retry_after), 120))
                continue
            response.raise_for_status()
        except requests.RequestException as error:
            last_error = error
            time.sleep(min(5 * attempt, 120))
    if last_error is not None:
        raise last_error
    raise RuntimeError("HF rows API failed without an exception.")


def _spec_for_source(specs: dict[str, DatasetSpec], source: str, split: str) -> DatasetSpec:
    base = specs[source]
    return DatasetSpec(
        name=base.name,
        source=base.source,
        split=split,
        config=base.config,
        gated=base.gated,
    )


def _sharechat_conversation_id(row: Mapping[str, Any]) -> str:
    for key in ("url", "share_id", "conversation_id", "id"):
        value = normalize_text(stringify(row.get(key)))
        if value:
            return value
    return ""


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
