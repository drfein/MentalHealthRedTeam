from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_1samp, ttest_ind, ttest_rel, wilcoxon


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def control_pairs(readouts_path: Path) -> pd.DataFrame:
    rows = []
    for row in read_jsonl(readouts_path):
        if row.get("skipped_oversize"):
            continue
        rows.append(
            {
                "original_row_idx": int(row["original_row_idx"]),
                "arm": str(row["condition"]),
                "target_text": row["target_text"],
                "misinformation_max_logit": float(
                    row["token_scores"]["misinformation"]["max_logit"]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    paired = frame.pivot(
        index="original_row_idx",
        columns="arm",
        values=["misinformation_max_logit", "target_text"],
    )
    paired.columns = [f"{arm}_{name}" for name, arm in paired.columns]
    paired = paired.reset_index().rename(
        columns={
            "direct_assertion_misinformation_max_logit": "direct_misinformation_max_logit",
            "question_misinformation_max_logit": "question_misinformation_max_logit",
            "direct_assertion_target_text": "target_text",
        }
    )
    paired["direct_misinformation_max_logit"] = pd.to_numeric(
        paired["direct_misinformation_max_logit"]
    )
    paired["question_misinformation_max_logit"] = pd.to_numeric(
        paired["question_misinformation_max_logit"]
    )
    paired["question_minus_direct_raw"] = (
        paired["question_misinformation_max_logit"]
        - paired["direct_misinformation_max_logit"]
    )
    paired["direct_minus_question_raw"] = -paired["question_minus_direct_raw"]
    return paired


def write_markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_delta_comparison(delusion: pd.DataFrame, control: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].hist(
        delusion["question_minus_direct_raw"],
        bins=22,
        alpha=0.65,
        density=True,
        label="delusion items",
    )
    axes[0].hist(
        control["question_minus_direct_raw"],
        bins=14,
        alpha=0.75,
        density=True,
        label="true/neutral controls",
    )
    axes[0].axvline(0, color="0.35", linestyle="--", linewidth=1)
    axes[0].set_xlabel("question minus direct raw misinformation logit")
    axes[0].set_ylabel("density")
    axes[0].set_title("Framing attenuation: delusion vs control")
    axes[0].legend(frameon=False)

    axes[1].boxplot(
        [
            delusion["question_minus_direct_raw"],
            control["question_minus_direct_raw"],
        ],
        labels=["delusion", "true/neutral"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#d7e6f4"},
        medianprops={"color": "black"},
    )
    axes[1].axhline(0, color="0.35", linestyle="--", linewidth=1)
    axes[1].set_ylabel("question minus direct raw misinformation logit")
    axes[1].set_title("Delta distributions")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare direct-vs-question misinformation attenuation for delusion and true/neutral controls."
    )
    parser.add_argument(
        "--delusion-pairs",
        type=Path,
        default=Path(
            "results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals/"
            "headline_probe_diagnostics/direct_vs_question_paired_items.csv"
        ),
    )
    parser.add_argument(
        "--control-readouts",
        type=Path,
        default=Path(
            "results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals/"
            "true_neutral_control/jspace_readouts.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals/"
            "true_neutral_control"
        ),
    )
    args = parser.parse_args()

    delusion = pd.read_csv(args.delusion_pairs)
    control = control_pairs(args.control_readouts)

    rows = []
    for name, frame in [("delusion", delusion), ("true_neutral", control)]:
        paired_t = ttest_rel(
            frame["direct_misinformation_max_logit"],
            frame["question_misinformation_max_logit"],
        )
        try:
            wilcoxon_p = float(
                wilcoxon(
                    frame["direct_misinformation_max_logit"],
                    frame["question_misinformation_max_logit"],
                ).pvalue
            )
        except ValueError:
            wilcoxon_p = float("nan")
        delta_one_sample = ttest_1samp(frame["question_minus_direct_raw"], 0)
        rows.append(
            {
                "group": name,
                "n": len(frame),
                "direct_mean": frame["direct_misinformation_max_logit"].mean(),
                "question_mean": frame["question_misinformation_max_logit"].mean(),
                "question_minus_direct_mean": frame["question_minus_direct_raw"].mean(),
                "question_minus_direct_median": frame["question_minus_direct_raw"].median(),
                "share_direct_lower_than_question": (
                    frame["question_minus_direct_raw"] > 0
                ).mean(),
                "paired_t_p": float(paired_t.pvalue),
                "wilcoxon_p": wilcoxon_p,
                "delta_vs_zero_p": float(delta_one_sample.pvalue),
            }
        )
    summary = pd.DataFrame(rows)

    delta_t = ttest_ind(
        delusion["question_minus_direct_raw"],
        control["question_minus_direct_raw"],
        equal_var=False,
    )
    delta_u = mannwhitneyu(
        delusion["question_minus_direct_raw"],
        control["question_minus_direct_raw"],
        alternative="two-sided",
    )
    comparison = pd.DataFrame(
        [
            {
                "contrast": "delusion_delta_minus_true_neutral_delta",
                "difference": delusion["question_minus_direct_raw"].mean()
                - control["question_minus_direct_raw"].mean(),
                "welch_t_p": float(delta_t.pvalue),
                "mann_whitney_p": float(delta_u.pvalue),
            }
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    control.to_csv(args.output_dir / "true_neutral_paired_items.csv", index=False)
    summary.to_csv(args.output_dir / "true_neutral_control_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "delusion_vs_true_neutral_delta.csv", index=False)
    plot_delta_comparison(delusion, control, args.output_dir / "delta_control_comparison.png")
    lines = [
        "# True/Neutral Framing Control",
        "",
        "Delta is `question raw misinformation max logit - direct raw misinformation max logit`.",
        "Positive delta means direct assertion has lower misinformation loading than question framing.",
        "",
        "## Summary",
        "",
        write_markdown_table(summary),
        "",
        "## Delusion vs Control Delta",
        "",
        write_markdown_table(comparison),
        "",
    ]
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
