#!/usr/bin/env python3
"""Plot paired J-space changes caused by fiction-framing prefixes."""

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


def load_scores(
    config: dict[str, Any],
    original_root: Path,
    intervention_root: Path,
) -> pd.DataFrame:
    rows = []
    for model in config["models"]:
        sources = [
            ("original", original_root / model["key"] / "readouts.jsonl"),
            (None, intervention_root / model["key"] / "readouts.jsonl"),
        ]
        for fixed_arm, path in sources:
            for item in read_jsonl(path):
                arm = fixed_arm or item["arm"]
                for readout in item["readouts"]:
                    for word in config["words"]:
                        rows.append(
                            {
                                "model_key": model["key"],
                                "model_label": model["label"],
                                "conversation_id": item["conversation_id"],
                                "row_idx": item["row_idx"],
                                "arm": arm,
                                "layer": readout["layer"],
                                "word": word,
                                "raw_max_logit": readout["word_scores"][word][
                                    "max_logit"
                                ],
                            }
                        )
    return pd.DataFrame(rows)


def paired_differences(scores: pd.DataFrame) -> pd.DataFrame:
    conversation_scores = (
        scores.groupby(
            [
                "model_key",
                "model_label",
                "conversation_id",
                "arm",
                "layer",
                "word",
            ],
            as_index=False,
        )["raw_max_logit"]
        .mean()
    )
    wide = conversation_scores.pivot(
        index=[
            "model_key",
            "model_label",
            "conversation_id",
            "layer",
            "word",
        ],
        columns="arm",
        values="raw_max_logit",
    ).reset_index()
    rows = []
    for arm in ["hypothetical", "story"]:
        subset = wide.dropna(subset=["original", arm]).copy()
        subset["arm"] = arm
        subset["delta_raw_logit"] = subset[arm] - subset["original"]
        rows.append(
            subset[
                [
                    "model_key",
                    "model_label",
                    "conversation_id",
                    "layer",
                    "word",
                    "arm",
                    "delta_raw_logit",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def summarize(
    differences: pd.DataFrame,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for keys, group in differences.groupby(
        ["model_key", "model_label", "arm", "word", "layer"],
        sort=False,
    ):
        values = group["delta_raw_logit"].to_numpy(dtype=float)
        sampled = rng.choice(values, size=(draws, len(values)), replace=True)
        means = sampled.mean(axis=1)
        rows.append(
            {
                "model_key": keys[0],
                "model_label": keys[1],
                "arm": keys[2],
                "word": keys[3],
                "layer": keys[4],
                "n_conversations": len(values),
                "mean_delta_raw_logit": float(values.mean()),
                "ci_low": float(np.quantile(means, 0.025)),
                "ci_high": float(np.quantile(means, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def plot(
    summary: pd.DataFrame,
    config: dict[str, Any],
    experiment: dict[str, Any],
    output_dir: Path,
) -> None:
    apply_paper_style()
    models = config["models"]
    arms = experiment["arms"]
    fig, axes = plt.subplots(
        len(models),
        len(arms),
        figsize=(11.2, 8.6),
        squeeze=False,
    )
    for row_index, model in enumerate(models):
        for column_index, arm in enumerate(arms):
            ax = axes[row_index, column_index]
            panel = summary[
                (summary["model_key"] == model["key"])
                & (summary["arm"] == arm["key"])
            ]
            for word in config["words"]:
                data = panel[panel["word"] == word].sort_values("layer")
                x = data["layer"].to_numpy()
                ax.plot(
                    x,
                    data["mean_delta_raw_logit"].to_numpy(),
                    marker="o",
                    markersize=2.8,
                    linewidth=1.7,
                    color=WORD_COLORS[word],
                    label=word.capitalize(),
                )
                ax.fill_between(
                    x,
                    data["ci_low"].to_numpy(),
                    data["ci_high"].to_numpy(),
                    color=WORD_COLORS[word],
                    alpha=0.13,
                    linewidth=0,
                )
            ax.axhline(0, color="#6B7280", linewidth=0.9)
            if row_index == 0:
                ax.set_title(arm["label"])
            if column_index == 0:
                ax.text(
                    0,
                    1.04,
                    model["label"],
                    transform=ax.transAxes,
                    fontweight="bold",
                    ha="left",
                )
            if row_index == len(models) - 1:
                ax.set_xlabel("Layer")
            ax.grid(axis="y")
            ax.set_axisbelow(True)
    fig.supylabel("Prefix-induced change in raw J-space logit")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(config["words"]),
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "jspace_fiction_prefix_layer_effects.png", dpi=300)
    fig.savefig(output_dir / "jspace_fiction_prefix_layer_effects.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/jspace_fiction_prefixes.json"),
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path("results/jspace_target_layer_trajectories"),
    )
    parser.add_argument(
        "--intervention-root",
        type=Path,
        default=Path("results/jspace_fiction_prefix_layers"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/jspace_fiction_prefix_layers/analysis"),
    )
    args = parser.parse_args()

    experiment = json.loads(
        args.experiment_config.read_text(encoding="utf-8")
    )
    config = json.loads(
        Path(experiment["base_config"]).read_text(encoding="utf-8")
    )
    scores = load_scores(config, args.original_root, args.intervention_root)
    differences = paired_differences(scores)
    summary = summarize(
        differences,
        int(config["plot"]["bootstrap_draws"]),
        int(config["plot"]["seed"]),
    )
    plot(summary, config, experiment, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    differences.to_csv(
        args.output_dir / "conversation_paired_differences.csv",
        index=False,
    )
    summary.to_csv(args.output_dir / "layer_effect_summary.csv", index=False)
    (args.output_dir / "hparams.json").write_text(
        json.dumps(
            {
                "experiment_config": str(args.experiment_config),
                "original_root": str(args.original_root),
                "intervention_root": str(args.intervention_root),
                "outcome": "counterfactual raw max logit minus original raw max logit",
                "unit": "source conversation after averaging repeated targets",
                **config["plot"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
