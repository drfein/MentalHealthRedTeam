#!/usr/bin/env python3
"""Cache selected J-space word readouts across layers at each flagged user turn."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from cache_jspace_conversation_trajectories import (
    load_config,
    load_hf_model,
    render_with_user_positions,
    select_target_prefix,
    token_ids_for_words,
)


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
        default=Path("results/jspace_target_layer_trajectories"),
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
    layers = sorted(int(layer) for layer in lens.jacobians)
    for layer in layers:
        lens.jacobians[layer] = lens.jacobians[layer].to(hf_model.device)

    data = pd.read_parquet(config["dataset"])
    hparams = {
        "config": str(args.config),
        "model": model_config,
        "layers": layers,
        "dataset_rows": len(data),
        "dataset_conversations": int(data["conversation_id"].nunique()),
        "word_token_ids": word_token_ids,
        "lens_path": str(lens_path),
        "quantize_4bit": args.quantize_4bit,
        "position": "final token overlapping the flagged user message",
        "context": (
            f"up to {config['max_prior_user_turns']} prior user messages, "
            "with intervening assistant messages and no system prompt"
        ),
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
                layers=layers,
                positions=[positions[-1]],
                max_seq_len=int(config["max_seq_len"]),
                use_jacobian=True,
            )

            layer_readouts: list[dict[str, Any]] = []
            for layer in layers:
                logits = lens_logits[layer].detach().float().reshape(-1)
                scores = {}
                for word, token_ids in word_token_ids.items():
                    valid_ids = [
                        token_id
                        for token_id in token_ids
                        if 0 <= token_id < logits.numel()
                    ]
                    values = logits[valid_ids]
                    scores[word] = {
                        "mean_logit": float(values.mean().item()),
                        "max_logit": float(values.max().item()),
                    }
                layer_readouts.append({"layer": layer, "word_scores": scores})

            record = {
                "row_idx": int(row_idx),
                "model_key": args.model_key,
                "model_label": model_config["label"],
                "source": row["source"],
                "conversation_id": row["conversation_id"],
                "message_hash": row["message_hash"],
                "target_message_index": int(row["target_message_index"]),
                "prompt_tokens": int(
                    len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
                ),
                "expected_prior_user_turns": expected_prior_turns,
                "dropped_user_turns_for_length": dropped_user_turns,
                "truncated_target_tokens_for_length": truncated_target_tokens,
                "readouts": layer_readouts,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
