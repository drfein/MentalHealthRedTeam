#!/usr/bin/env python3
"""Plot selected J-space word trajectories approaching flagged user turns."""

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


WORD_COLORS = {
    "misinformation": "#D55E00",
    "vision": "#0072B2",
    "conspiracy": "#009E73",
}


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
                            "layer": model["layer"],
                            "conversation_id": item["conversation_id"],
                            "row_idx": item["row_idx"],
                            "relative_user_turn": readout["relative_user_turn"],
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
    conversation_turn = (
        raw.groupby(
            [
                "model_key",
                "model_label",
                "layer",
                "word",
                "conversation_id",
                "relative_user_turn",
            ],
            as_index=False,
        )["raw_max_logit"]
        .mean()
    )
    conversation_turn["z_loading"] = conversation_turn.groupby(
        ["model_key", "word"]
    )["raw_max_logit"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0)
    )

    rng = np.random.default_rng(seed)
    summary_rows = []
    for keys, group in conversation_turn.groupby(
        ["model_key", "model_label", "layer", "word", "relative_user_turn"],
        sort=False,
    ):
        values = group["z_loading"].to_numpy(dtype=float)
        sampled = rng.choice(values, size=(draws, len(values)), replace=True)
        bootstrap_means = sampled.mean(axis=1)
        summary_rows.append(
            {
                "model_key": keys[0],
                "model_label": keys[1],
                "layer": keys[2],
                "word": keys[3],
                "relative_user_turn": keys[4],
                "n_conversations": len(values),
                "mean_z_loading": float(values.mean()),
                "ci_low": float(np.quantile(bootstrap_means, 0.025)),
                "ci_high": float(np.quantile(bootstrap_means, 0.975)),
            }
        )
    return conversation_turn, pd.DataFrame(summary_rows)


def plot(summary: pd.DataFrame, config: dict[str, Any], output_dir: Path) -> None:
    apply_paper_style()
    models = config["models"]
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(9.2, 8.6),
        sharex=True,
        sharey=True,
    )
    for ax, model in zip(axes, models, strict=True):
        model_data = summary[summary["model_key"] == model["key"]]
        for word in config["words"]:
            data = model_data[model_data["word"] == word].sort_values(
                "relative_user_turn"
            )
            x = data["relative_user_turn"].to_numpy()
            y = data["mean_z_loading"].to_numpy()
            ax.plot(
                x,
                y,
                marker="o",
                markersize=4,
                linewidth=2,
                color=WORD_COLORS[word],
                label=word.capitalize(),
            )
            ax.fill_between(
                x,
                data["ci_low"].to_numpy(),
                data["ci_high"].to_numpy(),
                color=WORD_COLORS[word],
                alpha=0.16,
                linewidth=0,
            )
        ax.axvline(0, color="#374151", linestyle=(0, (3, 3)), linewidth=1)
        ax.axhline(0, color="#9CA3AF", linewidth=0.8)
        ax.set_title(
            f"{model['label']} (layer {model['layer']})",
            loc="left",
        )
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[1].set_ylabel("Within-model standardized J-space loading")
    axes[-1].set_xlabel("User turn relative to flagged turn")
    axes[-1].set_xticks(
        list(range(-int(config["max_prior_user_turns"]), 1))
    )
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
    fig.savefig(output_dir / "jspace_conversation_trajectories.png", dpi=300)
    fig.savefig(output_dir / "jspace_conversation_trajectories.pdf")
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
        default=Path("results/jspace_conversation_trajectories"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/jspace_conversation_trajectories/analysis"),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw = load_readouts(config, args.input_root)
    expected_models = {model["key"] for model in config["models"]}
    if set(raw["model_key"]) != expected_models:
        raise ValueError("Readout files do not cover every configured model")
    conversation_turn, summary = summarize(
        raw,
        int(config["plot"]["bootstrap_draws"]),
        int(config["plot"]["seed"]),
    )
    plot(summary, config, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(args.output_dir / "readouts_long.parquet", index=False)
    conversation_turn.to_csv(
        args.output_dir / "conversation_turn_readouts.csv",
        index=False,
    )
    summary.to_csv(args.output_dir / "trajectory_summary.csv", index=False)
    (args.output_dir / "hparams.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "input_root": str(args.input_root),
                **config["plot"],
                "point_estimate": (
                    "Mean after averaging repeated target alignments within each "
                    "source conversation and relative user turn"
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
