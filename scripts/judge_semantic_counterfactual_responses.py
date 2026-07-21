from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from wild_delusion_miner.counterfactual_judge import (
    RUBRIC,
    counterfactual_row_key as row_key,
    parse_judgment as parse,
)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the framing-aware counterfactual judge.")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="microsoft/Phi-4-mini-instruct")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--selected-original-indices", type=Path, default=None)
    parser.add_argument("--sort-by-length", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    prompts = {row_key(row): row for row in read_jsonl(args.prompts)}
    rows = read_jsonl(args.generations)
    if args.selected_original_indices is not None:
        selected = {
            int(value)
            for value in args.selected_original_indices.read_text(encoding="utf-8").splitlines()
            if value.strip()
        }
        rows = [row for row in rows if int(row["original_row_idx"]) in selected]
    if args.resume and args.output.exists():
        completed = {row_key(row) for row in read_jsonl(args.output)}
        rows = [row for row in rows if row_key(row) not in completed]
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    if args.sort_by_length:
        rows.sort(
            key=lambda row: len(row["response"])
            + len(prompts[row_key(row)]["framed_text"])
        )

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
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for start in tqdm(range(0, len(rows), args.batch_size), desc="judge"):
            batch = rows[start : start + args.batch_size]
            rendered = []
            for row in batch:
                prompt = prompts[row_key(row)]
                payload = {
                    "framing": row["intervention_arm"],
                    "user_message": prompt["framed_text"],
                    "assistant_response": row["response"],
                }
                messages = [
                    {"role": "system", "content": RUBRIC},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ]
                kwargs = {"tokenize": False, "add_generation_prompt": True}
                try:
                    text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
                except TypeError:
                    text = tokenizer.apply_chat_template(messages, **kwargs)
                rendered.append(text)
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            outputs = tokenizer.batch_decode(
                generated[:, encoded.input_ids.shape[1] :], skip_special_tokens=True
            )
            for row, raw in zip(batch, outputs, strict=True):
                result = dict(row)
                result.update(judge_model=args.model_id, raw_judge_output=raw)
                try:
                    label = parse(raw)
                    result.update(
                        reality_endorsement_score=int(label["reality_endorsement_score"]),
                        frame_adherence_score=int(label["frame_adherence_score"]),
                        behavior_label=str(label["behavior_label"]),
                        judge_confidence=float(label.get("confidence", 0.0)),
                        judge_rationale=str(label.get("rationale", "")),
                    )
                except Exception as error:
                    result["judge_error"] = repr(error)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()

    output_rows = read_jsonl(args.output)
    errors = sum("judge_error" in row for row in output_rows)
    print(
        f"Wrote {len(rows)} new rows; output has {len(output_rows)} rows "
        f"({errors} errors): {args.output}"
    )


if __name__ == "__main__":
    main()
