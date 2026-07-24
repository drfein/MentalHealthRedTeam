#!/usr/bin/env python3
"""Cache target-turn J-space layer readouts after fiction-framing prefixes."""

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


def completed_keys(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    return {
        (int(item["row_idx"]), item["arm"])
        for item in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/jspace_fiction_prefixes.json"),
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/jspace_fiction_prefix_layers"),
    )
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--lens-path", type=Path, default=None)
    parser.add_argument("--quantize-4bit", action="store_true")
    args = parser.parse_args()

    import jlens

    experiment = json.loads(
        args.experiment_config.read_text(encoding="utf-8")
    )
    base_config_path = Path(experiment["base_config"])
    config, model_config = load_config(base_config_path, args.model_key)
    output_dir = args.output_root / args.model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "readouts.jsonl"
    done = completed_keys(output_path)

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
    (output_dir / "hparams.json").write_text(
        json.dumps(
            {
                "experiment_config": str(args.experiment_config),
                "base_config": str(base_config_path),
                "model": model_config,
                "layers": layers,
                "arms": experiment["arms"],
                "dataset_rows": len(data),
                "dataset_conversations": int(
                    data["conversation_id"].nunique()
                ),
                "word_token_ids": word_token_ids,
                "quantize_4bit": args.quantize_4bit,
                "position": "final token overlapping the prefixed target message",
                "context": (
                    f"up to {config['max_prior_user_turns']} prior user messages, "
                    "with intervening assistant messages and no system prompt"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with output_path.open("a", encoding="utf-8") as handle:
        total = len(data) * len(experiment["arms"])
        progress = tqdm(total=total, desc=args.model_key)
        for row_idx, row in data.iterrows():
            for arm in experiment["arms"]:
                key = (int(row_idx), arm["key"])
                if key in done:
                    progress.update(1)
                    continue
                messages = [dict(message) for message in list(row["messages"])]
                target_index = int(row["target_message_index"])
                messages[target_index]["content"] = (
                    arm["prefix"] + str(messages[target_index]["content"])
                )
                selected, _ = select_target_prefix(
                    messages,
                    target_index,
                    int(config["max_prior_user_turns"]),
                )
                prompt, positions, dropped, truncated = (
                    render_with_user_positions(
                        tokenizer,
                        selected,
                        int(config["max_seq_len"]),
                    )
                )
                lens_logits, _, _ = lens.apply(
                    model,
                    prompt,
                    layers=layers,
                    positions=[positions[-1]],
                    max_seq_len=int(config["max_seq_len"]),
                    use_jacobian=True,
                )
                readouts: list[dict[str, Any]] = []
                for layer in layers:
                    logits = lens_logits[layer].detach().float().reshape(-1)
                    scores = {}
                    for word, token_ids in word_token_ids.items():
                        values = logits[token_ids]
                        scores[word] = {
                            "mean_logit": float(values.mean().item()),
                            "max_logit": float(values.max().item()),
                        }
                    readouts.append({"layer": layer, "word_scores": scores})
                record = {
                    "row_idx": int(row_idx),
                    "arm": arm["key"],
                    "model_key": args.model_key,
                    "model_label": model_config["label"],
                    "conversation_id": row["conversation_id"],
                    "message_hash": row["message_hash"],
                    "dropped_user_turns_for_length": dropped,
                    "truncated_target_tokens_for_length": truncated,
                    "readouts": readouts,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                progress.update(1)
        progress.close()

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
