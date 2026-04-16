"""
generate_responses.py
=====================
For each conversation in WildDelusion, build a prompt from all turns up to and
including the flagged delusional user message, then sample n_gen independent
completions from a Llama-Instruct model.

Output JSONL format (one line per generation):
  { conversation_hash, source, gpt_score, probe_score,
    flagged_msg_idx, flagged_text, n_prior_turns, gen_idx, generation }

Checkpointed on (conversation_hash, gen_idx) — safe to resume by rerunning.

Usage:
  python generate_responses.py
  python generate_responses.py --n_gen 64 --out generations.jsonl
  python generate_responses.py --model meta-llama/Llama-3.1-8B-Instruct --batch_size 32
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model responses to delusional user messages."
    )
    parser.add_argument("--dataset", default="danielfein/WildDelusion",
                        help="HuggingFace dataset ID (default: danielfein/WildDelusion).")
    parser.add_argument("--config", default="combined",
                        help="Dataset config/split name (default: combined).")
    parser.add_argument("--n_gen", type=int, default=64,
                        help="Number of independent completions per conversation (default: 64).")
    parser.add_argument("--max_new", type=int, default=512,
                        help="Maximum new tokens per generation (default: 512).")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Number of generations to run in one forward pass (default: 64).")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID (default: meta-llama/Llama-3.1-8B-Instruct).")
    parser.add_argument("--out", default="generations.jsonl",
                        help="Output JSONL file (default: generations.jsonl).")
    return parser.parse_args()


def build_conversations(ds) -> list[dict]:
    """
    Parse dataset rows into conversation dicts suitable for generation.

    Each entry contains:
      - row: the original dataset record
      - messages: list of {role, content} dicts up to and including the flagged turn
    """
    conversations = []
    for r in ds:
        conv = json.loads(r["full_conversation"])
        idx = r["flagged_msg_idx"]
        if idx is None or idx >= len(conv):
            continue
        if conv[idx].get("role") != "user":
            continue
        messages = []
        for turn in conv[: idx + 1]:
            role = turn.get("role", "")
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        if not messages or messages[-1]["role"] != "user":
            continue
        conversations.append({"row": r, "messages": messages})
    return conversations


def load_done(out_path: Path) -> set[tuple[str, int]]:
    """Return the set of (conversation_hash, gen_idx) pairs already written."""
    done: set[tuple[str, int]] = set()
    if not out_path.exists():
        return done
    for line in out_path.open():
        if line.strip():
            obj = json.loads(line)
            done.add((obj["conversation_hash"], obj["gen_idx"]))
    return done


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading {args.dataset} ({args.config}) ...")
    ds = load_dataset(args.dataset, args.config)["train"]
    conversations = build_conversations(ds)
    print(f"Valid conversations: {len(conversations)}")
    print(f"Total generations planned: {len(conversations) * args.n_gen:,}")

    # Resume
    done = load_done(out_path)
    if done:
        print(f"Resuming -- {len(done)} generations already done")

    # Load model
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print("Model loaded.")

    out_f = out_path.open("a")

    for item in tqdm(conversations, desc="conversations"):
        r = item["row"]
        messages = item["messages"]
        h = r["conversation_hash"]

        remaining = [i for i in range(args.n_gen) if (h, i) not in done]
        if not remaining:
            continue

        # Tokenize prompt once; replicate per batch
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        enc_single = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=3072)
        prompt_len = enc_single["input_ids"].shape[1]

        all_texts: list[tuple[int, str]] = []
        for batch_start in range(0, len(remaining), args.batch_size):
            batch_idxs = remaining[batch_start: batch_start + args.batch_size]
            B = len(batch_idxs)

            input_ids = enc_single["input_ids"].expand(B, -1).to(model.device)
            attention_mask = enc_single["attention_mask"].expand(B, -1).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            for i, gen_idx in enumerate(batch_idxs):
                text = tokenizer.decode(
                    out[i][prompt_len:], skip_special_tokens=True
                ).strip()
                all_texts.append((gen_idx, text))

        for gen_idx, text in all_texts:
            record = {
                "conversation_hash": h,
                "source": r["source"],
                "gpt_score": r["gpt_score"],
                "probe_score": r["probe_score"],
                "flagged_msg_idx": r["flagged_msg_idx"],
                "flagged_text": r["flagged_text"],
                "n_prior_turns": len(messages) - 1,
                "gen_idx": gen_idx,
                "generation": text,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()

    out_f.close()
    total = sum(1 for _ in out_path.open() if _.strip())
    print(f"\nDone. {total:,} total generations written to {out_path}")


if __name__ == "__main__":
    main()
