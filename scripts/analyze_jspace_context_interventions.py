from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


CONTRASTS = {
    "validating_minus_neutral": ("validating", "neutral"),
    "reality_testing_minus_neutral": ("reality_testing", "neutral"),
    "validating_minus_reality_testing": ("validating", "reality_testing"),
}

CORE_METRICS = [
    "falsity_concern_mean_logit",
    "reality_testing_mean_logit",
    "validation_mystical_mean_logit",
    "misinformation_max_logit",
    "misinformation_rank_strength",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten(readout_path: Path, input_path: Path) -> pd.DataFrame:
    metadata = read_jsonl(input_path)
    rows = []
    for readout in read_jsonl(readout_path):
        meta = metadata[int(readout["row_idx"])]
        row = {
            "row_idx": int(readout["row_idx"]),
            "original_row_idx": int(meta["original_row_idx"]),
            "intervention_arm": meta["intervention_arm"],
            "source": meta["source"],
            "target_text": meta["target_text"],
            "annotation_score": meta.get("annotation_score"),
            "judge_confidence": meta.get("judge_confidence"),
            "layer": int(readout["layer"]),
        }
        for group, scores in readout["concept_scores"].items():
            row[f"{group}_mean_logit"] = scores["mean_logit"]
            row[f"{group}_max_logit"] = scores["max_logit"]
            row[f"{group}_best_rank"] = scores["best_rank"]
        misinformation = readout["token_scores"]["misinformation"]
        row["misinformation_max_logit"] = misinformation["max_logit"]
        row["misinformation_best_rank"] = misinformation["best_rank"]
        row["misinformation_rank_strength"] = -np.log10(misinformation["best_rank"])
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def paired_effects(data: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for metric in CORE_METRICS:
        for layer in sorted(data["layer"].unique()):
            wide = data[data["layer"] == layer].pivot(
                index="original_row_idx", columns="intervention_arm", values=metric
            )
            for contrast, (left, right) in CONTRASTS.items():
                paired = wide[[left, right]].dropna()
                differences = (paired[left] - paired[right]).to_numpy(dtype=float)
                low, high = bootstrap_mean_ci(differences, rng, draws)
                try:
                    p_value = float(wilcoxon(differences, alternative="two-sided").pvalue)
                except ValueError:
                    p_value = 1.0
                sd = differences.std(ddof=1)
                rows.append(
                    {
                        "metric": metric,
                        "layer": layer,
                        "contrast": contrast,
                        "n_pairs": len(differences),
                        "mean_difference": differences.mean(),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "paired_effect_dz": differences.mean() / sd if sd else np.nan,
                        "wilcoxon_p": p_value,
                    }
                )
    effects = pd.DataFrame(rows)
    effects["holm_p"] = np.nan
    for (_, _), indices in effects.groupby(["metric", "contrast"]).groups.items():
        effects.loc[indices, "holm_p"] = multipletests(
            effects.loc[indices, "wilcoxon_p"], method="holm"
        )[1]
    effects["robust"] = (
        (effects["holm_p"] < 0.05)
        & ((effects["bootstrap_ci_low"] > 0) | (effects["bootstrap_ci_high"] < 0))
    )
    return effects


def plot_effects(effects: pd.DataFrame, output: Path) -> None:
    labels = {
        "validating_minus_neutral": "Validating - neutral",
        "reality_testing_minus_neutral": "Reality-testing - neutral",
        "validating_minus_reality_testing": "Validating - reality-testing",
    }
    fig, axes = plt.subplots(len(CORE_METRICS), 1, figsize=(11, 3.4 * len(CORE_METRICS)), sharex=True)
    for ax, metric in zip(axes, CORE_METRICS, strict=True):
        subset = effects[effects["metric"] == metric]
        for contrast in CONTRASTS:
            rows = subset[subset["contrast"] == contrast].sort_values("layer")
            y = rows["mean_difference"].to_numpy()
            yerr = np.vstack(
                [
                    y - rows["bootstrap_ci_low"].to_numpy(),
                    rows["bootstrap_ci_high"].to_numpy() - y,
                ]
            )
            ax.errorbar(rows["layer"], y, yerr=yerr, marker="o", capsize=3, label=labels[contrast])
        ax.axhline(0, color="black", linewidth=1, alpha=0.6)
        ax.set_ylabel("Paired mean difference")
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Layer")
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(effects: pd.DataFrame, output: Path) -> None:
    robust = effects[effects["robust"]].sort_values(["metric", "contrast", "layer"])
    lines = [
        "# Probe-Free Context Intervention",
        "",
        "Each contrast is paired within the same flagged user message. Intervals are percentile ",
        "bootstrap 95% confidence intervals over messages. P-values are paired Wilcoxon tests ",
        "with Holm correction across tested layers within each metric and contrast.",
        "",
        f"Robust layer-level effects: {len(robust)} / {len(effects)}.",
        "",
    ]
    if robust.empty:
        lines.append("No effect met both the corrected p-value and bootstrap interval criteria.")
    else:
        lines.extend(
            [
                "| Metric | Contrast | Layer | Mean difference | 95% CI | dz | Holm p |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in robust.itertuples():
            lines.append(
                f"| {row.metric} | {row.contrast} | {row.layer} | "
                f"{row.mean_difference:.3f} | [{row.bootstrap_ci_low:.3f}, "
                f"{row.bootstrap_ci_high:.3f}] | {row.paired_effect_dz:.3f} | {row.holm_p:.3g} |"
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired J-space context interventions.")
    parser.add_argument("--readouts", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompt-variants", type=Path, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = flatten(args.readouts, args.input)
    if args.prompt_variants:
        prompts = pd.read_csv(args.prompt_variants)
        invalid_ids = set(
            (
                prompts.loc[prompts["prompt_tokens"] > args.max_prompt_tokens, "row_idx"]
                // 3
            ).astype(int)
        )
        data = data[~data["original_row_idx"].isin(invalid_ids)].copy()
    effects = paired_effects(data, args.bootstrap_draws, args.seed)
    data.to_parquet(args.out_dir / "context_intervention_readouts.parquet", index=False)
    effects.to_csv(args.out_dir / "paired_effects.csv", index=False)
    plot_effects(effects, args.out_dir / "paired_effects.png")
    write_report(effects, args.out_dir / "report.md")
    (args.out_dir / "hparams.json").write_text(
        json.dumps(
            {
                "readouts": str(args.readouts),
                "input": str(args.input),
                "prompt_variants": str(args.prompt_variants) if args.prompt_variants else None,
                "max_prompt_tokens": args.max_prompt_tokens,
                "bootstrap_draws": args.bootstrap_draws,
                "seed": args.seed,
                "multiplicity": "Holm across layers within metric and contrast",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
