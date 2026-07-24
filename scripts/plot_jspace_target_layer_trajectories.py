#!/usr/bin/env python3
"""Plot selected J-space word readouts across layers at flagged user turns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wild_delusion_miner.plot_style import apply_paper_style  # noqa: E402

from plot_jspace_conversation_trajectories import WORD_COLORS


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_readouts(config: dict[str, Any], input_root: Path) -> pd.DataFrame:
    rows = []
    for model in config["models"]:
        path = input_root / model["key"] / "readouts.jsonl"
        for item in read_jsonl(path):
            for readout in item["readouts"]:
                for word in config["words"]:
                    rows.append(
                        {
                            "model_key": model["key"],
                            "model_label": model["label"],
                            "conversation_id": item["conversation_id"],
                            "row_idx": item["row_idx"],
                            "layer": readout["layer"],
                            "word": word,
                            "raw_max_logit": readout["word_scores"][word][
                                "max_logit"
                            ],
                        }
                    )
    return pd.DataFrame(rows)


def summarize(
    raw: pd.DataFrame,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conversation_layer = (
        raw.groupby(
            [
                "model_key",
                "model_label",
                "word",
                "conversation_id",
                "layer",
            ],
            as_index=False,
        )["raw_max_logit"]
        .mean()
    )
    conversation_layer["z_loading"] = conversation_layer.groupby(
        ["model_key", "word"]
    )["raw_max_logit"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0)
    )

    rng = np.random.default_rng(seed)
    rows = []
    for keys, group in conversation_layer.groupby(
        ["model_key", "model_label", "word", "layer"],
        sort=False,
    ):
        values = group["z_loading"].to_numpy(dtype=float)
        sampled = rng.choice(values, size=(draws, len(values)), replace=True)
        means = sampled.mean(axis=1)
        rows.append(
            {
                "model_key": keys[0],
                "model_label": keys[1],
                "word": keys[2],
                "layer": keys[3],
                "n_conversations": len(values),
                "mean_z_loading": float(values.mean()),
                "ci_low": float(np.quantile(means, 0.025)),
                "ci_high": float(np.quantile(means, 0.975)),
            }
        )
    return conversation_layer, pd.DataFrame(rows)


def plot(summary: pd.DataFrame, config: dict[str, Any], output_dir: Path) -> None:
    apply_paper_style()
    models = config["models"]
    fig, axes = plt.subplots(len(models), 1, figsize=(9.2, 8.6))
    for ax, model in zip(axes, models, strict=True):
        model_data = summary[summary["model_key"] == model["key"]]
        for word in config["words"]:
            data = model_data[model_data["word"] == word].sort_values("layer")
            x = data["layer"].to_numpy()
            ax.plot(
                x,
                data["mean_z_loading"].to_numpy(),
                marker="o",
                markersize=3,
                linewidth=1.8,
                color=WORD_COLORS[word],
                label=word.capitalize(),
            )
            ax.fill_between(
                x,
                data["ci_low"].to_numpy(),
                data["ci_high"].to_numpy(),
                color=WORD_COLORS[word],
                alpha=0.14,
                linewidth=0,
            )
        ax.axhline(0, color="#9CA3AF", linewidth=0.8)
        ax.set_title(model["label"], loc="left")
        ax.set_xlabel("Layer")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[1].set_ylabel("Within-model standardized J-space loading")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(config["words"]),
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "jspace_target_layer_trajectories.png", dpi=300)
    fig.savefig(output_dir / "jspace_target_layer_trajectories.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/jspace_conversation_trajectories.json"),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/jspace_target_layer_trajectories"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/jspace_target_layer_trajectories/analysis"),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw = load_readouts(config, args.input_root)
    expected_models = {model["key"] for model in config["models"]}
    if set(raw["model_key"]) != expected_models:
        raise ValueError("Readout files do not cover every configured model")
    conversation_layer, summary = summarize(
        raw,
        int(config["plot"]["bootstrap_draws"]),
        int(config["plot"]["seed"]),
    )
    plot(summary, config, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(args.output_dir / "readouts_long.parquet", index=False)
    conversation_layer.to_csv(
        args.output_dir / "conversation_layer_readouts.csv",
        index=False,
    )
    summary.to_csv(args.output_dir / "layer_summary.csv", index=False)
    (args.output_dir / "hparams.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "input_root": str(args.input_root),
                **config["plot"],
                "position": "final token of the flagged user message",
                "point_estimate": (
                    "Mean after averaging repeated targets within source "
                    "conversation, layer, and word"
                ),
                "interval": (
                    "Pointwise percentile interval from resampling source "
                    "conversations with replacement"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
