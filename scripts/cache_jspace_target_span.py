#!/usr/bin/env python3
"""Cache full-vocabulary J-space logits averaged over each target user span."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoTokenizer

import jlens

from cache_jspace_full_vocabulary import select_rows
from run_jspace_context_ablation import (
    DEFAULT_SYSTEM_PROMPT,
    Variant,
    fit_prompt,
    load_hf_model,
)


def target_span_positions(
    tokenizer: Any,
    prompt: str,
    target_text: str,
) -> list[int]:
    """Return prompt-token positions overlapping the literal target text."""
    char_start = prompt.rfind(target_text)
    if char_start < 0:
        raise ValueError("Target text is not present in rendered prompt")
    char_end = char_start + len(target_text)
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    positions = [
        index
        for index, (start, end) in enumerate(encoded["offset_mapping"])
        if end > start and start < char_end and end > char_start
    ]
    expected_length = len(
        tokenizer(prompt, add_special_tokens=True)["input_ids"]
    )
    if len(encoded["input_ids"]) != expected_length:
        raise ValueError("Offset and model tokenizations disagree")
    if not positions:
        raise ValueError("Target span maps to no prompt tokens")
    return positions


def write_token_metadata(
    path: Path,
    tokenizer: Any,
    vocab_size: int,
) -> None:
    if path.exists():
        return
    special_ids = set(tokenizer.all_special_ids)
    with path.open("w", encoding="utf-8") as handle:
        for token_id in range(vocab_size):
            handle.write(
                json.dumps(
                    {
                        "token_id": token_id,
                        "decoded": tokenizer.decode(
                            [token_id],
                            clean_up_tokenization_spaces=False,
                        ),
                        "token_piece": tokenizer.convert_ids_to_tokens(token_id),
                        "is_special": token_id in special_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def initialize_cache(
    output_dir: Path,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    vocab_size: int,
    manifest: dict[str, Any],
) -> tuple[np.memmap, np.memmap]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items_path = output_dir / "items.jsonl"
    expected_ids = [int(row["original_row_idx"]) for row in rows]
    if items_path.exists():
        cached_ids = [
            int(json.loads(line)["original_row_idx"])
            for line in items_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if cached_ids != expected_ids:
            raise ValueError("Existing cache has a different item ordering")
    else:
        with items_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        {
                            "original_row_idx": int(row["original_row_idx"]),
                            "conversation_id": row["conversation_id"],
                            "message_hash": row["message_hash"],
                            "target_text": row["target_text"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    write_token_metadata(output_dir / "tokens.jsonl", tokenizer, vocab_size)
    shape = (len(rows), vocab_size)
    logits_path = output_dir / "target_span_mean_logits.float16.npy"
    completed_path = output_dir / "completed.uint8.npy"
    if logits_path.exists():
        logits = np.lib.format.open_memmap(logits_path, mode="r+")
        if logits.shape != shape:
            raise ValueError(f"Existing logits shape {logits.shape} != {shape}")
    else:
        logits = np.lib.format.open_memmap(
            logits_path,
            mode="w+",
            dtype=np.float16,
            shape=shape,
        )
    if completed_path.exists():
        completed = np.lib.format.open_memmap(completed_path, mode="r+")
        if completed.shape != (len(rows),):
            raise ValueError("Existing completion vector has the wrong shape")
    else:
        completed = np.lib.format.open_memmap(
            completed_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(rows),),
        )
        completed[:] = 0
        completed.flush()

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                **manifest,
                "n_items": len(rows),
                "vocab_size": vocab_size,
                "matrix_shape": list(shape),
                "matrix_dtype": "float16",
                "aggregation": (
                    "Mean across tokenizer positions overlapping target user content "
                    "within each item; items remain separate in the cache."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return logits, completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--selected-indices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-revision",
        default="a4114d7752d11eb546e6cf372213d7e75526d3a1",
    )
    parser.add_argument("--lens-file", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--arm", default="direct_assertion")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--position-chunk-size", type=int, default=128)
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument("--flush-every", type=int, default=5)
    args = parser.parse_args()

    rows = select_rows(args.input, args.selected_indices, args.arm)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    lens_path = hf_hub_download(
        args.lens_repo,
        args.lens_file,
        revision=args.lens_revision,
    )
    lens = jlens.JacobianLens.load(lens_path)
    hf_model = load_hf_model(
        args.model_id,
        quantize_4bit=args.quantize_4bit,
        revision=args.model_revision,
    )
    model = jlens.from_hf(hf_model, tokenizer, force_bos=False)
    vocab_size = int(hf_model.config.vocab_size)
    lens.jacobians[args.layer] = lens.jacobians[args.layer].to(hf_model.device)
    manifest = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    logits_cache, completed = initialize_cache(
        args.output_dir,
        rows,
        tokenizer,
        vocab_size,
        manifest,
    )

    pending = [index for index, done in enumerate(completed) if not done]
    for progress, index in enumerate(tqdm(pending, desc=args.model_id), start=1):
        row = rows[index]
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            *row["messages"],
        ]
        prompt, prompt_tokens, _ = fit_prompt(
            tokenizer,
            Variant(args.arm, None, messages),
            args.max_seq_len,
        )
        if prompt_tokens > args.max_seq_len:
            raise ValueError(f"Item {row['original_row_idx']} exceeds max length")
        positions = target_span_positions(tokenizer, prompt, row["framed_text"])
        item_sum = np.zeros(vocab_size, dtype=np.float64)
        for start in range(0, len(positions), args.position_chunk_size):
            chunk = positions[start : start + args.position_chunk_size]
            lens_logits, _, _ = lens.apply(
                model,
                prompt,
                layers=[args.layer],
                positions=chunk,
                max_seq_len=args.max_seq_len,
                use_jacobian=True,
            )
            item_sum += (
                lens_logits[args.layer]
                .detach()
                .float()
                .sum(dim=0)
                .cpu()
                .numpy()
            )
        logits_cache[index] = (item_sum / len(positions)).astype(np.float16)
        completed[index] = 1
        if progress % args.flush_every == 0:
            logits_cache.flush()
            completed.flush()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logits_cache.flush()
    completed.flush()


if __name__ == "__main__":
    main()
