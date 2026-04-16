"""
classify_roleplay.py
====================
Use GPT-5.2 to classify whether the first user message in each conversation
establishes a fictional/roleplay frame rather than expressing genuine personal
belief. Conversations flagged as roleplay should be excluded from downstream
analysis.

Loads the WildDelusion dataset (or another HF dataset with the same schema),
finds the first user turn per conversation, and calls the GPT classifier.

Output JSONL format (one line per conversation):
  { conversation_hash, first_msg, is_roleplay, reason }

Checkpointed on conversation_hash — safe to resume by rerunning.

Usage:
  python classify_roleplay.py
  python classify_roleplay.py --dataset danielfein/WildDelusion --config combined
  python classify_roleplay.py --output data/roleplay_classifications.jsonl
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

MAX_TOKENS = 80
N_WORKERS = 30
MODEL = "gpt-5.2"

SYSTEM = """You are a classifier. Your task: given the FIRST message a user sent to a chatbot, decide whether it establishes a fictional or roleplay frame — i.e., the user is asking the model to play a character, write fiction, do a creative scenario, or engage in a jailbreak persona — rather than expressing genuine personal beliefs.

Respond with JSON only:
{"is_roleplay": true/false, "reason": "<one short sentence>"}

Guidelines:
- TRUE: explicit roleplay, "write a story", "pretend you are X", "play as X", jailbreak personas ("DAN", "Narotica"), fiction scenarios labeled as such
- FALSE: genuine personal questions/beliefs, cosmological/philosophical musings (even if grandiose), asking for help with real problems
- If in doubt, lean FALSE."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify first user messages as roleplay/fiction vs genuine belief."
    )
    parser.add_argument("--dataset", default="danielfein/WildDelusion",
                        help="HuggingFace dataset ID (default: danielfein/WildDelusion).")
    parser.add_argument("--config", default="combined",
                        help="Dataset config/split name (default: combined).")
    parser.add_argument("--output", default="data/roleplay_classifications.jsonl",
                        help="Output JSONL file (default: data/roleplay_classifications.jsonl).")
    return parser.parse_args()


def classify_one(client: OpenAI, conv_hash: str, first_msg: str) -> dict:
    """Call GPT to classify whether the first message establishes a roleplay frame."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"FIRST MESSAGE:\n{first_msg[:600]}"},
        ],
        max_completion_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content.strip()
    obj = json.loads(raw)
    return {
        "conversation_hash": conv_hash,
        "first_msg": first_msg[:300],
        "is_roleplay": bool(obj.get("is_roleplay", False)),
        "reason": obj.get("reason", ""),
    }


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset and collect (conversation_hash, first_user_message) pairs
    print(f"Loading {args.dataset} ({args.config}) ...")
    ds = load_dataset(args.dataset, args.config)["train"]

    tasks: list[tuple[str, str]] = []
    for r in ds:
        conv = json.loads(r["full_conversation"])
        first_user = next((t for t in conv if t.get("role") == "user"), None)
        if first_user:
            tasks.append((r["conversation_hash"], first_user["content"]))

    print(f"Conversations to classify: {len(tasks)}")

    # Resume from checkpoint
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            if line.strip():
                done.add(json.loads(line)["conversation_hash"])
    remaining = [(h, m) for h, m in tasks if h not in done]
    print(f"Already done: {len(done)} | Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
    else:
        out_f = out_path.open("a")
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(classify_one, client, h, m): h for h, m in remaining}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="classifying"):
                try:
                    result = fut.result()
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                except Exception as e:
                    h = futures[fut]
                    print(f"Error on {h}: {e}")
        out_f.close()

    # Summary
    results = [json.loads(l) for l in out_path.open() if l.strip()]
    roleplay = [r for r in results if r["is_roleplay"]]
    print(f"\nTotal classified: {len(results)}")
    print(f"Roleplay (exclude): {len(roleplay)}")
    print(f"Keep: {len(results) - len(roleplay)}")
    if roleplay:
        print("\nRoleplay examples:")
        for r in roleplay[:10]:
            print(f"  {r['conversation_hash'][:12]} | {r['reason']} | {r['first_msg'][:80]}")


if __name__ == "__main__":
    main()
