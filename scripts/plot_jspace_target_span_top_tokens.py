#!/usr/bin/env python3
"""Plot top mean J-space tokens from target-user-message spans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wild_delusion_miner.plot_style import apply_paper_style  # noqa: E402


MODEL_COLORS = {
    "Qwen2.5-7B": "#0072B2",
    "Llama-3.1-8B-Instruct": "#D55E00",
    "Gemma-3-1B-IT": "#009E73",
}
MODEL_ORDER = ["Qwen2.5-7B", "Llama-3.1-8B-Instruct", "Gemma-3-1B-IT"]


def read_token_metadata(
    path: Path,
    token_ids: np.ndarray,
) -> dict[int, dict[str, Any]]:
    """Read only requested rows from the token-id-ordered JSONL vocabulary."""
    wanted = {int(token_id) for token_id in token_ids}
    metadata: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for token_id, line in enumerate(handle):
            if token_id in wanted:
                metadata[token_id] = json.loads(line)
                if len(metadata) == len(wanted):
                    break
    missing = wanted.difference(metadata)
    if missing:
        raise ValueError(f"Missing {len(missing)} token rows from {path}")
    return metadata


def display_token(decoded: str) -> str:
    return (
        decoded.replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .strip()
    )


def lexical_token(text: str) -> bool:
    return (
        bool(text)
        and len(text) <= 40
        and "\ufffd" not in text
        and any(character.isalnum() for character in text)
    )


def top_tokens(cache_dir: Path, top_k: int, pool_size: int) -> pd.DataFrame:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    completed = np.load(cache_dir / "completed.uint8.npy", mmap_mode="r")
    if not np.all(completed):
        raise ValueError(f"Incomplete cache: {cache_dir}")
    logits = np.load(
        cache_dir / "target_span_mean_logits.float16.npy",
        mmap_mode="r",
    )
    mean_logits = np.asarray(logits, dtype=np.float32).mean(axis=0)
    candidate_ids = np.argpartition(mean_logits, -pool_size)[-pool_size:]
    candidate_ids = candidate_ids[np.argsort(mean_logits[candidate_ids])[::-1]]
    metadata = read_token_metadata(cache_dir / "tokens.jsonl", candidate_ids)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for token_id in candidate_ids:
        token = metadata[int(token_id)]
        raw_decoded = str(token["decoded"])
        decoded = display_token(raw_decoded)
        if (
            token["is_special"]
            or not lexical_token(raw_decoded.strip())
            or decoded in seen
        ):
            continue
        seen.add(decoded)
        rows.append(
            {
                "model": manifest["model_id"],
                "model_name": manifest.get("name", cache_dir.name),
                "layer": int(manifest["layer"]),
                "token_id": int(token_id),
                "token": decoded,
                "token_piece": token["token_piece"],
                "mean_logit": float(mean_logits[token_id]),
                "rank": len(rows) + 1,
                "n_items": int(manifest["n_items"]),
            }
        )
        if len(rows) == top_k:
            break
    if len(rows) < top_k:
        raise ValueError(f"Only found {len(rows)} displayable tokens in {cache_dir}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/jspace_target_span_models.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/jspace_target_span/analysis"),
    )
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--candidate-pool", type=int, default=2000)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    by_name = {model["name"]: model for model in config["models"]}
    frames = []
    for name in MODEL_ORDER:
        model = by_name[name]
        frame = top_tokens(
            Path(model["output_dir"]),
            top_k=args.top_k,
            pool_size=args.candidate_pool,
        )
        frame["model_name"] = name
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "target_span_top_tokens.csv", index=False)

    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 6.6))
    for ax, name in zip(axes, MODEL_ORDER, strict=True):
        group = (
            results[results["model_name"] == name]
            .sort_values("mean_logit")
            .reset_index(drop=True)
        )
        ax.barh(
            group["token"],
            group["mean_logit"],
            color=MODEL_COLORS[name],
            alpha=0.9,
        )
        ax.set_title(name)
        ax.set_xlabel("Mean J-space logit")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.9, bottom=0.12, left=0.08, right=0.99, wspace=0.48)
    fig.savefig(args.output_dir / "target_span_top_tokens.png", dpi=300)
    fig.savefig(args.output_dir / "target_span_top_tokens.pdf")
    plt.close(fig)

    (args.output_dir / "hparams.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "top_k": args.top_k,
                "candidate_pool": args.candidate_pool,
                "aggregation": (
                    "For each item, average full-vocabulary J-space logits over all "
                    "tokenizer positions overlapping the target user content. Then "
                    "average item vectors with equal item weight."
                ),
                "display_filter": (
                    "Exclude special, blank, control-only, non-alphanumeric, duplicate "
                    "decoded strings, replacement characters, and strings over 40 chars."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
