#!/usr/bin/env python3
"""Cache assistant-boundary J-space logits for every vocabulary token."""

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

from run_jspace_context_ablation import (
    DEFAULT_SYSTEM_PROMPT,
    Variant,
    fit_prompt,
    load_hf_model,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_rows(
    input_path: Path,
    selected_indices_path: Path,
    arm: str,
) -> list[dict[str, Any]]:
    selected = {
        int(value)
        for value in selected_indices_path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    }
    rows = [
        row
        for row in read_jsonl(input_path)
        if int(row["original_row_idx"]) in selected
        and str(row.get("counterfactual_family", row["intervention_arm"])) == arm
    ]
    rows.sort(key=lambda row: int(row["original_row_idx"]))
    ids = [int(row["original_row_idx"]) for row in rows]
    if len(ids) != len(selected) or len(ids) != len(set(ids)):
        raise ValueError("Expected exactly one selected row per original item")
    return rows


def initialize_cache(
    output_dir: Path,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    vocab_size: int,
    hparams: dict[str, Any],
) -> tuple[np.memmap, np.memmap]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "items.jsonl"
    expected_ids = [int(row["original_row_idx"]) for row in rows]
    if metadata_path.exists():
        cached_ids = [
            int(row["original_row_idx"]) for row in read_jsonl(metadata_path)
        ]
        if cached_ids != expected_ids:
            raise ValueError("Existing cache has a different item ordering")
    else:
        with metadata_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        {
                            "original_row_idx": int(row["original_row_idx"]),
                            "source": row["source"],
                            "conversation_id": row["conversation_id"],
                            "message_hash": row["message_hash"],
                            "target_text": row["target_text"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    tokens_path = output_dir / "tokens.jsonl"
    if not tokens_path.exists():
        special_ids = set(tokenizer.all_special_ids)
        with tokens_path.open("w", encoding="utf-8") as handle:
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

    logits_path = output_dir / "logits.float16.npy"
    completed_path = output_dir / "completed.uint8.npy"
    shape = (len(rows), vocab_size)
    if logits_path.exists():
        logits = np.lib.format.open_memmap(logits_path, mode="r+", dtype=np.float16)
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
        completed = np.lib.format.open_memmap(
            completed_path,
            mode="r+",
            dtype=np.uint8,
        )
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

    manifest = {
        **hparams,
        "n_items": len(rows),
        "vocab_size": vocab_size,
        "matrix_shape": list(shape),
        "matrix_dtype": "float16",
        "position": "final prompt token before assistant response",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
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
    parser.add_argument("--lens-revision", default=None)
    parser.add_argument("--lens-file", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--arm", default="direct_assertion")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument("--flush-every", type=int, default=10)
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

    hparams = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    logits_cache, completed = initialize_cache(
        args.output_dir,
        rows,
        tokenizer,
        vocab_size,
        hparams,
    )
    pending = [index for index, done in enumerate(completed) if not done]
    for progress, index in enumerate(tqdm(pending, desc=f"{args.model_id} full vocab"), start=1):
        row = rows[index]
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, *row["messages"]]
        variant = Variant(name=args.arm, context_messages=None, messages=messages)
        prompt, prompt_tokens, _ = fit_prompt(tokenizer, variant, args.max_seq_len)
        if prompt_tokens > args.max_seq_len:
            raise ValueError(
                f"Item {row['original_row_idx']} exceeds max sequence length"
            )
        lens_logits, _, _ = lens.apply(
            model,
            prompt,
            layers=[args.layer],
            positions=[-1],
            max_seq_len=args.max_seq_len,
            use_jacobian=True,
        )
        values = lens_logits[args.layer][0].detach().float().cpu().numpy()
        if values.shape != (vocab_size,):
            raise ValueError(f"Unexpected J-space shape {values.shape}")
        logits_cache[index] = values.astype(np.float16)
        completed[index] = 1
        if progress % args.flush_every == 0:
            logits_cache.flush()
            completed.flush()
        torch.cuda.empty_cache()
    logits_cache.flush()
    completed.flush()


if __name__ == "__main__":
    main()
