#!/usr/bin/env python3
"""Cache selected J-space word readouts across user turns before each target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def load_config(path: Path, model_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    matches = [model for model in config["models"] if model["key"] == model_key]
    if len(matches) != 1:
        raise ValueError(f"Expected one model with key {model_key!r}; found {len(matches)}")
    return config, matches[0]


def merge_same_role_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[int, int]]:
    merged: list[dict[str, str]] = []
    original_to_merged: dict[int, int] = {}
    for original_index, message in enumerate(messages):
        role = str(message.get("role", "")).lower()
        content = str(message.get("content", ""))
        if role not in {"system", "user", "assistant"} or not content:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n\n" + content
            original_to_merged[original_index] = len(merged) - 1
        else:
            merged.append({"role": role, "content": content})
            original_to_merged[original_index] = len(merged) - 1
    return merged, original_to_merged


def token_ids_for_words(
    tokenizer: Any,
    words: list[str],
) -> dict[str, list[int]]:
    token_ids: dict[str, list[int]] = {}
    for word in words:
        ids: set[int] = set()
        for form in [word, f" {word}", word.capitalize(), f" {word.capitalize()}"]:
            ids.update(tokenizer.encode(form, add_special_tokens=False))
        token_ids[word] = sorted(ids)
    return token_ids


def select_target_prefix(
    messages: list[dict[str, Any]],
    target_message_index: int,
    max_prior_user_turns: int,
) -> tuple[list[dict[str, str]], int]:
    merged, original_to_merged = merge_same_role_messages(messages)
    if target_message_index not in original_to_merged:
        raise ValueError("Target message was removed while cleaning messages")
    target_index = original_to_merged[target_message_index]
    if merged[target_index]["role"] != "user":
        raise ValueError("Target message must have role=user")
    user_indices = [
        index
        for index, message in enumerate(merged[: target_index + 1])
        if message["role"] == "user"
    ]
    kept_user_indices = user_indices[-(max_prior_user_turns + 1) :]
    start = kept_user_indices[0]
    return merged[start : target_index + 1], len(kept_user_indices) - 1


def render_with_user_positions(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_seq_len: int,
) -> tuple[str, list[int], int]:
    kept = list(messages)
    dropped_user_turns = 0
    while True:
        prompt = tokenizer.apply_chat_template(
            kept,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        encoded = tokenizer(
            prompt,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        if len(encoded["input_ids"]) <= max_seq_len or len(kept) == 1:
            break
        if kept[0]["role"] == "user":
            dropped_user_turns += 1
        kept.pop(0)

    cursor = 0
    user_spans: list[tuple[int, int]] = []
    for message in kept:
        content = message["content"]
        start = prompt.find(content, cursor)
        if start < 0:
            raise ValueError("Rendered prompt does not contain a source message verbatim")
        end = start + len(content)
        cursor = end
        if message["role"] == "user":
            user_spans.append((start, end))

    offsets = encoded["offset_mapping"]
    positions = []
    for start, end in user_spans:
        overlapping = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start and token_start < end and token_end > start
        ]
        if not overlapping:
            raise ValueError("User message maps to no tokenizer positions")
        positions.append(overlapping[-1])
    return prompt, positions, dropped_user_turns


def completed_rows(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {
        int(json.loads(line)["row_idx"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/jspace_conversation_trajectories.json"),
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/jspace_conversation_trajectories"),
    )
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--lens-path", type=Path, default=None)
    parser.add_argument("--quantize-4bit", action="store_true")
    args = parser.parse_args()

    import jlens
    from run_jspace_context_ablation import load_hf_model

    config, model_config = load_config(args.config, args.model_key)
    output_dir = args.output_root / args.model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "readouts.jsonl"
    done = completed_rows(output_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"],
        revision=model_config["model_revision"],
        trust_remote_code=True,
        cache_dir=args.model_cache_dir,
    )
    word_token_ids = token_ids_for_words(tokenizer, config["words"])
    lens_path = args.lens_path or Path(
        hf_hub_download(
            config["lens_repo"],
            model_config["lens_file"],
            revision=config["lens_revision"],
        )
    )
    lens = jlens.JacobianLens.load(lens_path)
    hf_model = load_hf_model(
        model_config["model_id"],
        quantize_4bit=args.quantize_4bit,
        revision=model_config["model_revision"],
        cache_dir=args.model_cache_dir,
    )
    model = jlens.from_hf(hf_model, tokenizer, force_bos=False)
    layer = int(model_config["layer"])
    lens.jacobians[layer] = lens.jacobians[layer].to(hf_model.device)

    data = pd.read_parquet(config["dataset"])
    hparams = {
        "config": str(args.config),
        "model": model_config,
        "dataset_rows": len(data),
        "dataset_conversations": int(data["conversation_id"].nunique()),
        "word_token_ids": word_token_ids,
        "lens_path": str(lens_path),
        "quantize_4bit": args.quantize_4bit,
        "no_system_prompt": True,
    }
    (output_dir / "hparams.json").write_text(
        json.dumps(hparams, indent=2) + "\n",
        encoding="utf-8",
    )

    with output_path.open("a", encoding="utf-8") as handle:
        for row_idx, row in tqdm(data.iterrows(), total=len(data), desc=args.model_key):
            if int(row_idx) in done:
                continue
            messages, expected_prior_turns = select_target_prefix(
                list(row["messages"]),
                int(row["target_message_index"]),
                int(config["max_prior_user_turns"]),
            )
            prompt, positions, dropped_user_turns = render_with_user_positions(
                tokenizer,
                messages,
                int(config["max_seq_len"]),
            )
            lens_logits, _, _ = lens.apply(
                model,
                prompt,
                layers=[layer],
                positions=positions,
                max_seq_len=int(config["max_seq_len"]),
                use_jacobian=True,
            )
            logits = lens_logits[layer].detach().float()
            n_prior_turns = len(positions) - 1
            records = []
            for position_index, position_logits in enumerate(logits):
                scores = {}
                for word, token_ids in word_token_ids.items():
                    valid_ids = [
                        token_id
                        for token_id in token_ids
                        if 0 <= token_id < position_logits.numel()
                    ]
                    values = position_logits[valid_ids]
                    scores[word] = {
                        "mean_logit": float(values.mean().item()),
                        "max_logit": float(values.max().item()),
                    }
                records.append(
                    {
                        "relative_user_turn": position_index - n_prior_turns,
                        "prompt_position": positions[position_index],
                        "word_scores": scores,
                    }
                )
            result = {
                "row_idx": int(row_idx),
                "model_key": args.model_key,
                "model_label": model_config["label"],
                "layer": layer,
                "source": row["source"],
                "discovery_split": row["discovery_split"],
                "conversation_id": row["conversation_id"],
                "message_hash": row["message_hash"],
                "target_message_index": int(row["target_message_index"]),
                "prompt_tokens": len(tokenizer(prompt, add_special_tokens=True)["input_ids"]),
                "expected_prior_user_turns": expected_prior_turns,
                "dropped_user_turns_for_length": dropped_user_turns,
                "readouts": records,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
