"""
scan_corpus.py
==============
Scan all user messages in a WildChat parquet corpus through a linear probe
trained on Llama layer activations, writing hits above a score threshold.

Uses an early-exit forward hook after the target layer to avoid computing
the remaining transformer layers, reducing memory and compute.

Checkpoint file tracks completed shards so the scan can be safely interrupted
and resumed.

Output JSONL format (one line per hit):
  { conversation_hash, msg_idx, score, text }

Usage:
  python scan_corpus.py --corpus_dir /path/to/wildchat --probe output/probe.json
  python scan_corpus.py --corpus_dir /data/wc --probe probe.json --threshold 3.0 --out hits.jsonl
"""

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_LENGTH = 128  # most user messages fit; longer ones truncated


class _EarlyExit(Exception):
    """Raised by the layer hook to short-circuit remaining forward passes."""
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan WildChat user messages with a linear probe."
    )
    parser.add_argument("--corpus_dir", required=True,
                        help="Directory containing WildChat .parquet shards.")
    parser.add_argument("--probe", required=True,
                        help="Path to probe JSON (output of train_probe.py).")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B",
                        help="HuggingFace model ID (default: meta-llama/Llama-3.1-8B).")
    parser.add_argument("--layer", type=int, default=13,
                        help="Layer index for activation extraction (default: 13).")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Minimum probe score to record a hit (default: 2.0).")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Tokenization/inference batch size (default: 256).")
    parser.add_argument("--out", default="probe_hits.jsonl",
                        help="Output JSONL file for hits (default: probe_hits.jsonl).")
    return parser.parse_args()


def load_probe(probe_path: str) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Load probe weights and scaler parameters from JSON."""
    data = json.loads(Path(probe_path).read_text())
    direction = np.array(data["direction"], dtype=np.float32)
    intercept = float(data["intercept"])
    scaler_mean = np.array(data["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(data["scaler_scale"], dtype=np.float32)
    return direction, intercept, scaler_mean, scaler_scale


def load_model_with_hook(
    model_name: str,
    layer: int,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, dict]:
    """
    Load the model and register an early-exit hook on the target layer.

    The hook stores the layer output in `layer_cache` and raises _EarlyExit
    to prevent the forward pass from running subsequent layers.
    """
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    layer_cache: dict = {}

    def _hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        layer_cache["h"] = h.detach()
        raise _EarlyExit()

    model.model.layers[layer].register_forward_hook(_hook)
    print(f"Early-exit hook registered on layer {layer}.")
    return model, tokenizer, layer_cache


@torch.no_grad()
def probe_batch(
    texts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer_cache: dict,
    mean_t: torch.Tensor,
    scale_t: torch.Tensor,
    direction_t: torch.Tensor,
    intercept: float,
) -> np.ndarray:
    """Return probe scores (N,) for a batch of texts."""
    enc = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH,
    ).to(model.device)
    try:
        model(**enc)
    except _EarlyExit:
        pass
    h = layer_cache["h"].float()                       # (B, S, D)
    mask = enc["attention_mask"].float().unsqueeze(-1)
    pooled = (h * mask).sum(1) / mask.sum(1)           # (B, D)
    X = (pooled - mean_t) / scale_t
    scores = (X @ direction_t + intercept).cpu().numpy()
    return scores


def gather_parquet_shards(corpus_dir: str) -> list[str]:
    """Recursively collect all .parquet files under corpus_dir."""
    parquets = sorted([
        str(Path(root) / f)
        for root, _, files in os.walk(corpus_dir)
        for f in files if f.endswith(".parquet")
    ])
    return parquets


def main() -> None:
    args = parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_path.with_suffix(".progress.json")

    # Load probe
    direction, intercept, scaler_mean, scaler_scale = load_probe(args.probe)

    # Load model with early-exit hook
    model, tokenizer, layer_cache = load_model_with_hook(args.model, args.layer)

    # Move probe tensors to GPU for fast dot products
    device = model.device
    direction_t = torch.from_numpy(direction).to(device)
    mean_t = torch.from_numpy(scaler_mean).to(device)
    scale_t = torch.from_numpy(scaler_scale).to(device)

    # Load checkpoint
    progress = (json.loads(progress_path.read_text())
                if progress_path.exists() else {"done_shards": []})
    done_shards = set(progress["done_shards"])

    parquets = gather_parquet_shards(args.corpus_dir)
    print(f"Found {len(parquets)} shards, {len(done_shards)} already done.")

    out_f = out_path.open("a")
    total_scanned = 0
    total_hits = 0

    for shard_path in tqdm(parquets, desc="shards"):
        shard_name = Path(shard_path).name
        if shard_name in done_shards:
            continue

        df = pd.read_parquet(shard_path)

        # Flatten to (conv_hash, msg_idx, text) for user messages only
        rows: list[tuple[str, int, str]] = []
        for _, row in df.iterrows():
            conv_hash = row["conversation_hash"]
            for mi, msg in enumerate(row["conversation"]):
                if msg.get("role") == "user":
                    text = (msg.get("content") or "").strip()
                    if text:
                        rows.append((conv_hash, mi, text))

        # Score in batches
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i: i + args.batch_size]
            texts = [r[2] for r in batch]
            scores = probe_batch(
                texts, model, tokenizer, layer_cache,
                mean_t, scale_t, direction_t, intercept,
            )
            for (conv_hash, mi, text), score in zip(batch, scores):
                total_scanned += 1
                if score >= args.threshold:
                    total_hits += 1
                    out_f.write(json.dumps({
                        "conversation_hash": conv_hash,
                        "msg_idx": mi,
                        "score": round(float(score), 3),
                        "text": text[:500],
                    }, ensure_ascii=False) + "\n")
            out_f.flush()

        done_shards.add(shard_name)
        progress_path.write_text(json.dumps({"done_shards": list(done_shards)}))
        tqdm.write(
            f"  {shard_name}: {len(rows)} user msgs | "
            f"total scanned={total_scanned:,}  hits={total_hits:,}"
        )
        gc.collect()

    out_f.close()
    print(
        f"\nDone. Scanned {total_scanned:,} user messages. "
        f"Hits (score >= {args.threshold}): {total_hits:,} "
        f"({total_hits / max(total_scanned, 1):.3%})"
    )
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
