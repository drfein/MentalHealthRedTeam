from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoTokenizer

import jlens

from run_jspace_context_ablation import (
    DEFAULT_SYSTEM_PROMPT,
    Variant,
    build_concept_token_map,
    decode_top_tokens,
    fit_prompt,
    load_hf_model,
    summarize_group,
    summarize_tokens,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream assistant-boundary J-space readouts for semantic counterfactuals."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concept-spec", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument("--lens-revision", default=None)
    parser.add_argument(
        "--lens-file",
        default="qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[26])
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--selected-original-indices", type=Path, default=None)
    parser.add_argument("--selected-arms", nargs="+", default=None)
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.selected_original_indices is not None:
        selected = {
            int(value)
            for value in args.selected_original_indices.read_text(encoding="utf-8").splitlines()
            if value.strip()
        }
        rows = [row for row in rows if int(row["original_row_idx"]) in selected]
    if args.selected_arms is not None:
        selected_arms = set(args.selected_arms)
        rows = [
            row
            for row in rows
            if str(row.get("counterfactual_family", row["intervention_arm"]))
            in selected_arms
        ]
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    completed: set[tuple[int, str, int]] = set()
    if args.resume and args.output.exists():
        completed = {
            (
                int(row["original_row_idx"]),
                str(row["condition"]),
                int(row["layer"]),
            )
            for row in read_jsonl(args.output)
            if not row.get("skipped_oversize")
        }
    rows = [
        row
        for row in rows
        if not all(
            (
                int(row["original_row_idx"]),
                str(row.get("counterfactual_family", row["intervention_arm"])),
                layer,
            )
            in completed
            for layer in args.layers
        )
    ]

    concept_groups = json.loads(args.concept_spec.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        revision=args.model_revision,
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
    concept_map = build_concept_token_map(tokenizer, concept_groups)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    hparams = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    args.output.with_suffix(f"{args.output.suffix}.hparams.json").write_text(
        json.dumps(hparams, indent=2), encoding="utf-8"
    )
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for row in tqdm(rows, desc="counterfactual J-space"):
            condition = str(row.get("counterfactual_family", row["intervention_arm"]))
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                *row["messages"],
            ]
            variant = Variant(name="full_prefix", context_messages=None, messages=messages)
            prompt, prompt_tokens, dropped = fit_prompt(tokenizer, variant, args.max_seq_len)
            if prompt_tokens > args.max_seq_len:
                result = {
                    "original_row_idx": int(row["original_row_idx"]),
                    "condition": condition,
                    "skipped_oversize": True,
                    "prompt_tokens": prompt_tokens,
                    "dropped_context_messages": dropped,
                }
            else:
                needed_layers = [
                    layer
                    for layer in args.layers
                    if (int(row["original_row_idx"]), condition, layer) not in completed
                ]
                lens_logits, _, _ = lens.apply(
                    model,
                    prompt,
                    layers=needed_layers,
                    positions=[-1],
                    max_seq_len=args.max_seq_len,
                    use_jacobian=True,
                )
                for layer in needed_layers:
                    logits = lens_logits[layer][0]
                    result = {
                        "original_row_idx": int(row["original_row_idx"]),
                        "source": row["source"],
                        "conversation_id": row["conversation_id"],
                        "message_hash": row["message_hash"],
                        "condition": condition,
                        "target_text": row["target_text"],
                        "framed_text": row.get("framed_text", row["messages"][-1]["content"]),
                        "layer": layer,
                        "prompt_tokens": prompt_tokens,
                        "dropped_context_messages": dropped,
                        "skipped_oversize": False,
                        "top_tokens": decode_top_tokens(tokenizer, logits, args.top_k),
                        "token_scores": summarize_tokens(logits, concept_map),
                        "concept_scores": {
                            group: summarize_group(logits, token_map)
                            for group, token_map in concept_map.items()
                        },
                    }
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
            if prompt_tokens > args.max_seq_len:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
