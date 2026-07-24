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
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def load_config(path: Path, model_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    matches = [model for model in config["models"] if model["key"] == model_key]
    if len(matches) != 1:
        raise ValueError(f"Expected one model with key {model_key!r}; found {len(matches)}")
    return config, matches[0]


def normalize_role(role: Any) -> str | None:
    normalized = str(role).lower()
    if normalized == "user":
        return "user"
    if normalized in {"assistant", "llm"}:
        return "assistant"
    return None


def merge_same_role_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if merged and merged[-1]["role"] == role:
            merged[-1]["segments"].append(content)
            merged[-1]["content"] += "\n\n" + content
        else:
            merged.append(
                {
                    "role": role,
                    "content": content,
                    "segments": [content],
                }
            )
    return merged


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


def load_hf_model(
    model_id: str,
    quantize_4bit: bool,
    revision: str,
    cache_dir: Path | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "revision": revision,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    if quantize_4bit and getattr(config, "quantization_config", None) is None:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as causal_error:
        print(
            "AutoModelForCausalLM failed; trying AutoModelForImageTextToText: "
            f"{causal_error}",
            flush=True,
        )
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


def select_target_prefix(
    messages: list[dict[str, Any]],
    target_message_index: int,
    max_prior_user_turns: int,
) -> tuple[list[dict[str, Any]], int]:
    if not 0 <= target_message_index < len(messages):
        raise ValueError("Target message index is outside the conversation")
    if normalize_role(messages[target_message_index].get("role")) != "user":
        raise ValueError("Target message must have role=user")

    prefix: list[dict[str, str]] = []
    for message in messages[: target_message_index + 1]:
        role = normalize_role(message.get("role"))
        content = str(message.get("content", ""))
        if role is not None and content:
            prefix.append({"role": role, "content": content})
    if not prefix or prefix[-1]["role"] != "user":
        raise ValueError("Target message was removed while cleaning messages")

    user_indices = [
        index
        for index, message in enumerate(prefix)
        if message["role"] == "user"
    ]
    kept_user_indices = user_indices[-(max_prior_user_turns + 1) :]
    start = kept_user_indices[0]
    return merge_same_role_messages(prefix[start:]), len(kept_user_indices) - 1


def render_with_user_positions(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_seq_len: int,
) -> tuple[str, list[int], int, int]:
    kept = [
        {
            "role": message["role"],
            "content": message["content"],
            "segments": list(message["segments"]),
        }
        for message in messages
    ]
    dropped_user_turns = 0
    truncated_target_tokens = 0
    while True:
        template_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in kept
        ]
        prompt = tokenizer.apply_chat_template(
            template_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        encoded = tokenizer(
            prompt,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        if len(encoded["input_ids"]) <= max_seq_len:
            break
        if len(kept) == 1:
            if len(kept[0]["segments"]) > 1:
                kept[0]["segments"].pop(0)
                kept[0]["content"] = "\n\n".join(kept[0]["segments"])
                dropped_user_turns += 1
                continue
            original_content = kept[0]["content"]
            original_token_count = len(
                tokenizer(
                    original_content,
                    add_special_tokens=False,
                )["input_ids"]
            )
            low, high = 1, len(original_content) - 1
            while low < high:
                midpoint = (low + high) // 2
                candidate = [
                    {
                        "role": kept[0]["role"],
                        "content": original_content[midpoint:],
                    }
                ]
                candidate_prompt = tokenizer.apply_chat_template(
                    candidate,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                candidate_length = len(
                    tokenizer(
                        candidate_prompt,
                        add_special_tokens=True,
                    )["input_ids"]
                )
                if candidate_length > max_seq_len:
                    low = midpoint + 1
                else:
                    high = midpoint
            kept[0] = {
                **kept[0],
                "content": original_content[low:],
                "segments": [original_content[low:]],
            }
            retained_token_count = len(
                tokenizer(
                    kept[0]["content"],
                    add_special_tokens=False,
                )["input_ids"]
            )
            truncated_target_tokens += original_token_count - retained_token_count
            continue
        if kept[0]["role"] == "user":
            dropped_user_turns += len(kept[0]["segments"])
        kept.pop(0)
        if kept and kept[0]["role"] == "assistant":
            kept.pop(0)

    cursor = 0
    user_spans: list[tuple[int, int]] = []
    for message in kept:
        if message["role"] == "user":
            for segment in message["segments"]:
                content = segment.strip()
                start = prompt.find(content, cursor)
                if start < 0:
                    raise ValueError(
                        "Rendered prompt does not contain a user message verbatim"
                    )
                end = start + len(content)
                cursor = end
                user_spans.append((start, end))
        else:
            content = message["content"].strip()
            start = prompt.find(content, cursor)
            if start < 0:
                raise ValueError(
                    "Rendered prompt does not contain an assistant message verbatim"
                )
            cursor = start + len(content)

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
    if max(positions) >= len(encoded["input_ids"]):
        raise ValueError("Saved user position exceeds the rendered prompt")
    return prompt, positions, dropped_user_turns, truncated_target_tokens


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
            (
                prompt,
                positions,
                dropped_user_turns,
                truncated_target_tokens,
            ) = render_with_user_positions(
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
                "truncated_target_tokens_for_length": truncated_target_tokens,
                "readouts": records,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
