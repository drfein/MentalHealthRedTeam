from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import roc_auc_score


DEFAULT_ARMS = (
    "direct_assertion",
    "question",
    "quotation_analysis",
    "reported_belief",
    "reconsideration",
    "explicit_fiction",
    "skeptical_roleplay",
    "translation",
)


def auc_or_nan(labels: pd.Series, scores: pd.Series) -> float:
    if labels.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def bootstrap_auc(
    frame: pd.DataFrame,
    *,
    label: str,
    score: str,
    group: str,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    grouped = list(frame.groupby(group))
    labels_by_group = [value[label].to_numpy() for _, value in grouped]
    scores_by_group = [value[score].to_numpy() for _, value in grouped]
    group_indices = np.arange(len(grouped))
    values: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(group_indices, size=len(group_indices), replace=True)
        draw_labels = np.concatenate([labels_by_group[index] for index in sampled])
        draw_scores = np.concatenate([scores_by_group[index] for index in sampled])
        if len(np.unique(draw_labels)) < 2:
            continue
        value = float(roc_auc_score(draw_labels, draw_scores))
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def permutation_p_value(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    observed: float,
    draws: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(draws):
        shuffled = rng.permutation(labels)
        value = roc_auc_score(shuffled, scores)
        if value >= observed:
            hits += 1
    return float((hits + 1) / (draws + 1))


def plot_distribution(direct: pd.DataFrame, output: Path) -> None:
    endorsed = direct[direct["positive"] == 1]["misinformation_max_logit"]
    non_endorsed = direct[direct["positive"] == 0]["misinformation_max_logit"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(non_endorsed, bins=18, alpha=0.65, label="not endorsed", density=True)
    axes[0].hist(endorsed, bins=10, alpha=0.75, label="endorsed", density=True)
    axes[0].axvline(non_endorsed.median(), color="C0", linestyle="--", linewidth=1)
    axes[0].axvline(endorsed.median(), color="C1", linestyle="--", linewidth=1)
    axes[0].set_title("Layer-26 misinformation loading")
    axes[0].set_xlabel("raw misinformation max logit")
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)

    axes[1].boxplot(
        [non_endorsed, endorsed],
        labels=["not endorsed", "endorsed"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#d7e6f4"},
        medianprops={"color": "black"},
    )
    jitter_rng = np.random.default_rng(20260716)
    for idx, values in enumerate([non_endorsed, endorsed], start=1):
        x = jitter_rng.normal(idx, 0.035, size=len(values))
        axes[1].scatter(x, values, s=16, alpha=0.55)
    axes[1].set_title("Same values by outcome")
    axes[1].set_ylabel("raw misinformation max logit")

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_cross_arm(cross_arm: pd.DataFrame, output: Path) -> None:
    ordered = cross_arm.sort_values("auc", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(ordered["arm"], ordered["auc"], color="#5b8db8")
    ax.axhline(0.5, color="0.45", linestyle="--", linewidth=1)
    ax.set_ylim(0.25, 0.85)
    ax.set_ylabel("AUROC predicting direct-assertion endorsement")
    ax.set_xlabel("arm whose layer-26 readout is used")
    ax.set_title("Does the readout track item difficulty across framings?")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_direct_question_scatter(paired: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    non = paired[paired["direct_positive"] == 0]
    pos = paired[paired["direct_positive"] == 1]
    ax.scatter(
        non["question_misinformation_max_logit"],
        non["direct_misinformation_max_logit"],
        s=30,
        alpha=0.65,
        label="direct not endorsed",
    )
    ax.scatter(
        pos["question_misinformation_max_logit"],
        pos["direct_misinformation_max_logit"],
        s=46,
        alpha=0.9,
        label="direct endorsed",
    )
    minimum = min(
        paired["question_misinformation_max_logit"].min(),
        paired["direct_misinformation_max_logit"].min(),
    )
    maximum = max(
        paired["question_misinformation_max_logit"].max(),
        paired["direct_misinformation_max_logit"].max(),
    )
    padding = 0.35
    ax.plot(
        [minimum - padding, maximum + padding],
        [minimum - padding, maximum + padding],
        color="0.35",
        linestyle="--",
        linewidth=1,
        label="y = x",
    )
    ax.set_xlim(minimum - padding, maximum + padding)
    ax.set_ylim(minimum - padding, maximum + padding)
    ax.set_xlabel("question-arm raw misinformation max logit")
    ax.set_ylabel("direct-assertion raw misinformation max logit")
    ax.set_title("Layer-26 misinformation readout by framing")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_delta_by_outcome(paired: pd.DataFrame, output: Path) -> None:
    endorsed = paired[paired["direct_positive"] == 1]["question_minus_direct_raw"]
    non_endorsed = paired[paired["direct_positive"] == 0]["question_minus_direct_raw"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(non_endorsed, bins=18, alpha=0.65, density=True, label="not endorsed")
    axes[0].hist(endorsed, bins=10, alpha=0.75, density=True, label="endorsed")
    axes[0].axvline(0, color="0.35", linestyle="--", linewidth=1)
    axes[0].set_xlabel("question minus direct raw misinformation logit")
    axes[0].set_ylabel("density")
    axes[0].set_title("Framing-induced attenuation")
    axes[0].legend(frameon=False)

    axes[1].boxplot(
        [non_endorsed, endorsed],
        labels=["not endorsed", "endorsed"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#d7e6f4"},
        medianprops={"color": "black"},
    )
    rng = np.random.default_rng(20260717)
    for idx, values in enumerate([non_endorsed, endorsed], start=1):
        axes[1].scatter(rng.normal(idx, 0.035, len(values)), values, s=16, alpha=0.55)
    axes[1].axhline(0, color="0.35", linestyle="--", linewidth=1)
    axes[1].set_ylabel("question minus direct raw misinformation logit")
    axes[1].set_title("Delta by direct-arm outcome")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_available_layers(layer_path: Path, output: Path) -> None:
    if not layer_path.exists():
        return
    trajectory = pd.read_csv(layer_path)
    keep = trajectory[
        trajectory["indicator"].isin(
            [
                "inverse_misinformation_max",
                "inverse_misinformation_mean",
                "inverse_falsity_concern_mean",
                "inverse_reality_testing_mean",
            ]
        )
    ].copy()
    if keep.empty:
        return
    labels = {
        "inverse_misinformation_max": "misinformation max",
        "inverse_misinformation_mean": "misinformation mean",
        "inverse_falsity_concern_mean": "falsity concern",
        "inverse_reality_testing_mean": "reality testing",
    }
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for indicator, group in keep.groupby("indicator"):
        group = group.sort_values("layer")
        yerr = np.vstack([group["auc"] - group["ci_low"], group["ci_high"] - group["auc"]])
        ax.errorbar(
            group["layer"],
            group["auc"],
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=2,
            label=labels[indicator],
        )
    ax.axhline(0.5, color="0.45", linestyle="--", linewidth=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("AUROC")
    ax.set_title("Available layer trajectory for fixed indicators")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
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
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_summary(
    output_dir: Path,
    *,
    direct_auc: float,
    direct_ci: tuple[float, float],
    permutation_p: float,
    cross_arm: pd.DataFrame,
    paired_direct_question: pd.DataFrame,
    paired_stats: dict[str, float],
    delta_regressions: pd.DataFrame,
    distribution: pd.DataFrame,
    search_space_rows: list[dict[str, object]],
) -> None:
    question_auc = cross_arm.loc[cross_arm["arm"] == "question", "auc"]
    quotation_auc = cross_arm.loc[cross_arm["arm"] == "quotation_analysis", "auc"]
    lines = [
        "# Headline Probe Diagnostics",
        "",
        "These diagnostics use cached Qwen2.5-7B counterfactual rows.",
        "",
        "## Headline Probe",
        "",
        (
            f"Direct-assertion inverse misinformation max AUROC: {direct_auc:.3f} "
            f"[{direct_ci[0]:.3f}, {direct_ci[1]:.3f}]."
        ),
        f"Permutation p-value for label shuffling: {permutation_p:.4f}.",
        "",
        (
            "The local cache does not include the raw layer-26 token-placebo readouts "
            "for the eight-arm counterfactual set, so this script does not claim a "
            "full random-direction placebo test. It reports the exact fixed-probe "
            "diagnostics and records the available search-space accounting."
        ),
        "",
        "## Cross-Arm Confound Check",
        "",
    ]
    if not question_auc.empty:
        lines.append(
            f"Question-arm readout predicting direct endorsement AUROC: {question_auc.iloc[0]:.3f}."
        )
    if not quotation_auc.empty:
        lines.append(
            f"Quotation-arm readout predicting direct endorsement AUROC: {quotation_auc.iloc[0]:.3f}."
        )
    lines.extend(
        [
            "",
            "If non-direct arms predict direct endorsement about as well as the direct arm, "
            "that points toward item plausibility/difficulty. If they are weaker, the "
            "direct-arm state is carrying additional framing-specific information.",
            "",
            "## Direct vs Question Paired Test",
            "",
            (
                "Mean raw misinformation max logit: "
                f"direct {paired_stats['direct_mean']:.3f}, "
                f"question {paired_stats['question_mean']:.3f}, "
                f"direct-minus-question {paired_stats['mean_difference']:.3f}."
            ),
            (
                f"Paired t-test p={paired_stats['paired_t_p']:.4g}; "
                f"Wilcoxon p={paired_stats['wilcoxon_p']:.4g}."
            ),
            (
                "Question-arm readout predicting direct-arm endorsement AUROC: "
                f"{paired_stats['question_to_direct_auc']:.3f}; "
                "direct-arm readout predicting direct-arm endorsement AUROC: "
                f"{paired_stats['direct_to_direct_auc']:.3f}."
            ),
            (
                "Share below y=x, where direct assertion has lower misinformation loading "
                f"than question framing: all={paired_stats['below_identity_share']:.3f}, "
                f"endorsed={paired_stats['endorsed_below_identity_share']:.3f}, "
                f"not-endorsed={paired_stats['nonendorsed_below_identity_share']:.3f}."
            ),
            (
                "Delta-only AUROC, where delta = question raw minus direct raw: "
                f"{paired_stats['delta_auc']:.3f}."
            ),
            "",
            "Regression separating item plausibility from framing attenuation:",
            "",
            markdown_table(delta_regressions),
            "",
            "First rows of the paired table:",
            "",
            markdown_table(paired_direct_question.head(8)),
            "",
            "## Attenuation vs Absence",
            "",
            markdown_table(distribution),
            "",
            "## Search-Space Accounting",
            "",
            markdown_table(pd.DataFrame(search_space_rows)),
            "",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostics for the layer-26 inverse misinformation headline probe."
    )
    parser.add_argument(
        "--analysis-rows",
        type=Path,
        default=Path(
            "results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals/"
            "fixed_indicator_layer26_primary/analysis_rows.parquet"
        ),
    )
    parser.add_argument(
        "--layer-trajectory",
        type=Path,
        default=Path(
            "results/jspace_semantic_specificity/qwen2_5_7b_layer_trajectory_holdout/"
            "indicator_layer_trajectory.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals/"
            "headline_probe_diagnostics"
        ),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--permutation-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    frame = pd.read_parquet(args.analysis_rows).copy()
    frame["misinformation_max_logit"] = -frame["inverse_misinformation_max"]
    direct = frame[frame["arm"] == "direct_assertion"].copy()
    direct_auc = auc_or_nan(direct["positive"], direct["inverse_misinformation_max"])
    direct_ci = bootstrap_auc(
        direct,
        label="positive",
        score="inverse_misinformation_max",
        group="original_row_idx",
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    permutation_p = permutation_p_value(
        direct["positive"].to_numpy(),
        direct["inverse_misinformation_max"].to_numpy(),
        observed=direct_auc,
        draws=args.permutation_draws,
        seed=args.seed + 1,
    )

    direct_labels = direct[["original_row_idx", "positive", "graded_score"]].rename(
        columns={"positive": "direct_positive", "graded_score": "direct_score"}
    )
    cross = frame.merge(direct_labels, on="original_row_idx", validate="many_to_one")
    cross_rows = []
    for arm, group in cross.groupby("arm"):
        value = auc_or_nan(group["direct_positive"], group["inverse_misinformation_max"])
        ci = bootstrap_auc(
            group,
            label="direct_positive",
            score="inverse_misinformation_max",
            group="original_row_idx",
            draws=args.bootstrap_draws,
            seed=args.seed + 10 + len(cross_rows),
        )
        cross_rows.append(
            {
                "arm": arm,
                "n": len(group),
                "direct_positive_n": int(group["direct_positive"].sum()),
                "auc": value,
                "ci_low": ci[0],
                "ci_high": ci[1],
                "mean_inverse_misinformation": group["inverse_misinformation_max"].mean(),
            }
        )
    cross_arm = pd.DataFrame(cross_rows).sort_values("auc", ascending=False)

    other_probe_rows = []
    candidate_columns = [
        "inverse_misinformation_max",
        "inverse_misinformation_mean",
        "inverse_misinformation_rank",
        "inverse_falsity_mean",
        "inverse_reality_testing_mean",
    ]
    for column in candidate_columns:
        other_probe_rows.append(
            {
                "probe": column,
                "auc": auc_or_nan(direct["positive"], direct[column]),
            }
        )
    other_probes = pd.DataFrame(other_probe_rows).sort_values("auc", ascending=False)

    distribution = (
        direct.groupby("positive")
        .agg(
            n=("positive", "size"),
            mean_misinformation_max_logit=("misinformation_max_logit", "mean"),
            median_misinformation_max_logit=("misinformation_max_logit", "median"),
            mean_inverse_misinformation=("inverse_misinformation_max", "mean"),
            median_inverse_misinformation=("inverse_misinformation_max", "median"),
            mean_openai_score=("graded_score", "mean"),
        )
        .reset_index()
    )
    distribution["outcome"] = distribution["positive"].map(
        {0: "not endorsed", 1: "endorsed"}
    )
    distribution = distribution.drop(columns=["positive"])[
        [
            "outcome",
            "n",
            "mean_misinformation_max_logit",
            "median_misinformation_max_logit",
            "mean_inverse_misinformation",
            "median_inverse_misinformation",
            "mean_openai_score",
        ]
    ]

    examples = direct.sort_values("inverse_misinformation_max", ascending=False)[
        [
            "original_row_idx",
            "source",
            "target_text",
            "graded_score",
            "positive",
            "misinformation_max_logit",
            "inverse_misinformation_max",
            "annotation_score",
            "judge_confidence",
        ]
    ]
    endorsed_examples = examples[examples["positive"] == 1].copy()

    paired = (
        frame[frame["arm"].isin(["direct_assertion", "question"])]
        .pivot(
            index="original_row_idx",
            columns="arm",
            values=[
                "misinformation_max_logit",
                "inverse_misinformation_max",
                "graded_score",
                "positive",
                "target_text",
                "source",
            ],
        )
        .copy()
    )
    paired.columns = [f"{arm}_{name}" for name, arm in paired.columns]
    paired = paired.reset_index()
    paired = paired.rename(
        columns={
            "direct_assertion_misinformation_max_logit": "direct_misinformation_max_logit",
            "question_misinformation_max_logit": "question_misinformation_max_logit",
            "direct_assertion_inverse_misinformation_max": "direct_inverse_misinformation_max",
            "question_inverse_misinformation_max": "question_inverse_misinformation_max",
            "direct_assertion_graded_score": "direct_score",
            "question_graded_score": "question_score",
            "direct_assertion_positive": "direct_positive",
            "question_positive": "question_positive",
            "direct_assertion_target_text": "target_text",
            "direct_assertion_source": "source",
        }
    )
    numeric_paired_columns = [
        "direct_misinformation_max_logit",
        "question_misinformation_max_logit",
        "direct_inverse_misinformation_max",
        "question_inverse_misinformation_max",
        "direct_score",
        "question_score",
        "direct_positive",
        "question_positive",
    ]
    for column in numeric_paired_columns:
        paired[column] = pd.to_numeric(paired[column])
    paired["direct_minus_question_raw"] = (
        paired["direct_misinformation_max_logit"]
        - paired["question_misinformation_max_logit"]
    )
    paired["question_minus_direct_raw"] = -paired["direct_minus_question_raw"]
    paired["direct_below_question"] = paired["direct_minus_question_raw"] < 0
    paired["neg_question_raw"] = -paired["question_misinformation_max_logit"]
    paired["z_neg_question_raw"] = (
        paired["neg_question_raw"] - paired["neg_question_raw"].mean()
    ) / paired["neg_question_raw"].std(ddof=0)
    paired["z_question_minus_direct_raw"] = (
        paired["question_minus_direct_raw"] - paired["question_minus_direct_raw"].mean()
    ) / paired["question_minus_direct_raw"].std(ddof=0)
    t_result = ttest_rel(
        paired["direct_misinformation_max_logit"],
        paired["question_misinformation_max_logit"],
    )
    try:
        wilcoxon_result = wilcoxon(
            paired["direct_misinformation_max_logit"],
            paired["question_misinformation_max_logit"],
            zero_method="wilcox",
        )
        wilcoxon_p = float(wilcoxon_result.pvalue)
    except ValueError:
        wilcoxon_p = float("nan")
    paired_stats = {
        "n": int(len(paired)),
        "direct_positive_n": int(paired["direct_positive"].sum()),
        "direct_mean": float(paired["direct_misinformation_max_logit"].mean()),
        "question_mean": float(paired["question_misinformation_max_logit"].mean()),
        "mean_difference": float(paired["direct_minus_question_raw"].mean()),
        "median_difference": float(paired["direct_minus_question_raw"].median()),
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_p": float(t_result.pvalue),
        "wilcoxon_p": wilcoxon_p,
        "question_to_direct_auc": auc_or_nan(
            paired["direct_positive"], -paired["question_misinformation_max_logit"]
        ),
        "direct_to_direct_auc": auc_or_nan(
            paired["direct_positive"], -paired["direct_misinformation_max_logit"]
        ),
        "delta_auc": auc_or_nan(
            paired["direct_positive"], paired["question_minus_direct_raw"]
        ),
        "below_identity_share": float(paired["direct_below_question"].mean()),
        "endorsed_below_identity_share": float(
            paired.loc[paired["direct_positive"] == 1, "direct_below_question"].mean()
        ),
        "nonendorsed_below_identity_share": float(
            paired.loc[paired["direct_positive"] == 0, "direct_below_question"].mean()
        ),
    }
    delta_regression_rows = []
    for outcome, formula in [
        (
            "direct_positive",
            "direct_positive ~ z_neg_question_raw + z_question_minus_direct_raw",
        ),
        (
            "direct_score",
            "direct_score ~ z_neg_question_raw + z_question_minus_direct_raw",
        ),
    ]:
        model = smf.ols(formula, data=paired).fit(cov_type="HC3")
        for term in ["z_neg_question_raw", "z_question_minus_direct_raw"]:
            ci = model.conf_int().loc[term]
            delta_regression_rows.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "coefficient_per_sd": float(model.params[term]),
                    "robust_se": float(model.bse[term]),
                    "ci_low": float(ci.iloc[0]),
                    "ci_high": float(ci.iloc[1]),
                    "p_value": float(model.pvalues[term]),
                }
            )
    delta_regressions = pd.DataFrame(delta_regression_rows)

    search_space_rows = [
        {
            "stage": "fixed headline probe",
            "layers": "26",
            "probe_tokens_or_groups": "misinformation",
            "aggregations": "max",
            "tested_combinations": 1,
            "notes": "pre-frozen for headline direct-assertion test",
        },
        {
            "stage": "available fixed-indicator diagnostics",
            "layers": "26",
            "probe_tokens_or_groups": "misinformation, falsity, reality-testing",
            "aggregations": "max, mean, rank depending on group",
            "tested_combinations": len(candidate_columns),
            "notes": "available in flattened local cache",
        },
        {
            "stage": "available layer trajectory",
            "layers": "8,16,24,26",
            "probe_tokens_or_groups": "misinformation, falsity, reality-testing",
            "aggregations": "max, mean, rank depending on group",
            "tested_combinations": "see layer trajectory CSV",
            "notes": "same cached endpoint, available layers only",
        },
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cross_arm.to_csv(args.output_dir / "cross_arm_direct_endorsement_auc.csv", index=False)
    other_probes.to_csv(args.output_dir / "available_probe_auc_direct_assertion.csv", index=False)
    distribution.to_csv(args.output_dir / "direct_readout_distribution.csv", index=False)
    endorsed_examples.to_csv(args.output_dir / "endorsed_direct_assertion_examples.csv", index=False)
    paired.to_csv(args.output_dir / "direct_vs_question_paired_items.csv", index=False)
    pd.DataFrame([paired_stats]).to_csv(
        args.output_dir / "direct_vs_question_paired_stats.csv", index=False
    )
    delta_regressions.to_csv(args.output_dir / "delta_regressions.csv", index=False)
    pd.DataFrame(search_space_rows).to_csv(
        args.output_dir / "search_space_accounting.csv", index=False
    )
    plot_distribution(direct, args.output_dir / "direct_misinformation_distribution.png")
    plot_cross_arm(cross_arm, args.output_dir / "cross_arm_direct_auc.png")
    plot_direct_question_scatter(paired, args.output_dir / "direct_vs_question_scatter.png")
    plot_delta_by_outcome(paired, args.output_dir / "delta_by_outcome.png")
    plot_available_layers(args.layer_trajectory, args.output_dir / "available_layer_trajectory.png")
    write_summary(
        args.output_dir,
        direct_auc=direct_auc,
        direct_ci=direct_ci,
        permutation_p=permutation_p,
        cross_arm=cross_arm,
        paired_direct_question=paired[
            [
                "original_row_idx",
                "direct_misinformation_max_logit",
                "question_misinformation_max_logit",
                "direct_minus_question_raw",
                "question_minus_direct_raw",
                "direct_positive",
            ]
        ],
        paired_stats=paired_stats,
        delta_regressions=delta_regressions,
        distribution=distribution,
        search_space_rows=search_space_rows,
    )
    hparams = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    hparams.update(
        rows=int(len(frame)),
        direct_rows=int(len(direct)),
        direct_positive_n=int(direct["positive"].sum()),
        direct_auc=direct_auc,
        direct_auc_ci_low=direct_ci[0],
        direct_auc_ci_high=direct_ci[1],
        permutation_p_value=permutation_p,
        direct_vs_question=paired_stats,
        cache_limitation=(
            "Raw eight-arm layer-26 token-placebo readouts were not available locally; "
            "random-direction placebo null was not computed here."
        ),
    )
    (args.output_dir / "hparams.json").write_text(
        json.dumps(hparams, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
