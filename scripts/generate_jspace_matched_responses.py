from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from run_jspace_context_ablation import (
    DEFAULT_SYSTEM_PROMPT,
    build_variants,
    fit_prompt,
    load_hf_model,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_indices(path: Path | None) -> set[int]:
    if path is None:
        return set()
    return {int(line) for line in path.read_text().splitlines() if line.strip()}


def decode_final_response(
    tokenizer: Any, token_ids: torch.Tensor
) -> tuple[str, str, bool, bool]:
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    complete = tokenizer.eos_token_id in token_ids.tolist()
    marker = "<|channel|>final<|message|>"
    if marker not in raw:
        return (
            tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
            raw,
            False,
            complete,
        )
    final = raw.rsplit(marker, 1)[1]
    for end_marker in ("<|return|>", "<|end|>", "<|endoftext|>"):
        final = final.split(end_marker, 1)[0]
    return final.strip(), raw, True, complete


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate responses from exactly the prompts used for J-space readouts."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--intervention-arm", default="neutral")
    parser.add_argument("--selected-original-indices", type=Path, default=None)
    parser.add_argument("--oversize-indices", type=Path, default=None)
    parser.add_argument("--standard-max-seq-len", type=int, default=1024)
    parser.add_argument("--oversize-max-seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--oversize-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    oversize = read_indices(args.oversize_indices)
    selected = read_indices(args.selected_original_indices)
    rows = [
        (idx, row)
        for idx, row in enumerate(read_jsonl(args.input))
        if (
            args.intervention_arm == "all"
            or row.get("intervention_arm") == args.intervention_arm
        )
        and (not selected or int(row["original_row_idx"]) in selected)
    ]
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    completed: set[tuple[int, str]] = set()
    if args.resume and args.output.exists():
        completed = {
            (int(row["original_row_idx"]), str(row["intervention_arm"]))
            for row in read_jsonl(args.output)
        }
        rows = [
            (idx, row)
            for idx, row in rows
            if (int(row["original_row_idx"]), str(row["intervention_arm"])) not in completed
        ]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        revision=args.model_revision,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = load_hf_model(
        args.model_id,
        quantize_4bit=args.quantize_4bit,
        revision=args.model_revision,
    )

    prepared = []
    for input_row_idx, row in rows:
        variants, _ = build_variants(
            row,
            context_message_counts=[-1],
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        max_seq_len = (
            args.oversize_max_seq_len if input_row_idx in oversize else args.standard_max_seq_len
        )
        prompt, prompt_tokens, dropped = fit_prompt(tokenizer, variants[0], max_seq_len)
        if prompt_tokens > max_seq_len:
            raise ValueError(f"Prompt {input_row_idx} still exceeds its matched context limit")
        prepared.append(
            {
                "input_row_idx": input_row_idx,
                "row": row,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "dropped_context_messages": dropped,
                "max_seq_len": max_seq_len,
            }
        )

    normal = [item for item in prepared if item["input_row_idx"] not in oversize]
    large = [item for item in prepared if item["input_row_idx"] in oversize]
    batches: list[list[dict[str, Any]]] = []
    for items, batch_size in [(normal, args.batch_size), (large, args.oversize_batch_size)]:
        batches.extend(items[start : start + batch_size] for start in range(0, len(items), batch_size))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for batch in tqdm(batches, desc="generation batches"):
            prompts = [item["prompt"] for item in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = outputs[:, encoded.input_ids.shape[1] :]
            responses = [decode_final_response(tokenizer, token_ids) for token_ids in generated]
            for item, (response, raw_response, parsed_final, generation_complete) in zip(
                batch, responses, strict=True
            ):
                row = item["row"]
                result = {
                    "input_row_idx": item["input_row_idx"],
                    "original_row_idx": row["original_row_idx"],
                    "intervention_arm": row["intervention_arm"],
                    "source": row["source"],
                    "conversation_id": row["conversation_id"],
                    "message_hash": row["message_hash"],
                    "target_text": row["target_text"],
                    "model_id": args.model_id,
                    "decoding": "greedy",
                    "max_new_tokens": args.max_new_tokens,
                    "quantize_4bit": args.quantize_4bit,
                    "prompt_tokens": item["prompt_tokens"],
                    "max_seq_len": item["max_seq_len"],
                    "dropped_context_messages": item["dropped_context_messages"],
                    "response": response.strip(),
                    "raw_response": raw_response,
                    "parsed_final_channel": parsed_final,
                    "generation_complete": generation_complete,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
