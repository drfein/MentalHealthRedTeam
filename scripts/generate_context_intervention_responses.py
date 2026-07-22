from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_jspace_context_ablation import DEFAULT_SYSTEM_PROMPT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_key(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["original_row_idx"]), str(row["intervention_arm"])


def render(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, *row["messages"]]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate replies for paired context interventions.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--selected-original-indices", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    if args.selected_original_indices is not None:
        selected = {
            int(line)
            for line in args.selected_original_indices.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        rows = [row for row in rows if int(row["original_row_idx"]) in selected]
    if args.resume and args.output.exists():
        completed = {row_key(row) for row in read_jsonl(args.output)}
        rows = [row for row in rows if row_key(row) not in completed]
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for start in tqdm(range(0, len(rows), args.batch_size), desc="batches"):
            batch = rows[start : start + args.batch_size]
            prompts = [render(tokenizer, row) for row in batch]
            original_input_tokens = [
                len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = outputs[:, encoded.input_ids.shape[1] :]
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, text, input_tokens in zip(
                batch, texts, original_input_tokens, strict=True
            ):
                result = {
                    "original_row_idx": row["original_row_idx"],
                    "intervention_arm": row["intervention_arm"],
                    "source": row["source"],
                    "conversation_id": row["conversation_id"],
                    "message_hash": row["message_hash"],
                    "target_text": row["target_text"],
                    "model_id": args.model_id,
                    "decoding": "greedy",
                    "input_tokens": input_tokens,
                    "max_input_tokens": args.max_input_tokens,
                    "response": text.strip(),
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
