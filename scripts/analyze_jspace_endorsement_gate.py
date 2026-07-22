from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import spearmanr, wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.proportion import proportion_confint


DEFAULT_PRIMARY_TOKEN = "misinformation"
DEFAULT_LAYER = 26
ARM_ORDER = ["neutral", "validating", "reality_testing"]
EPISTEMIC_CONFUSION_TOKENS = [
    "misunderstanding",
    "misunderstood",
    "confusion",
    "confused",
    "misinterpretation",
    "misinterpreted",
]
LEXICAL_PLACEBO_TOKENS = ["window", "table", "water", "music", "number", "question", "answer", "morning"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bootstrap_grouped_auc_difference(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, float]:
    groups = frame["original_row_idx"].unique()
    by_group = {group: frame.index[frame["original_row_idx"] == group].to_numpy() for group in groups}
    differences = []
    for _ in range(draws):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        sample = frame.loc[indices]
        if sample["positive"].nunique() < 2:
            continue
        differences.append(
            roc_auc_score(sample["positive"], sample["full_oof"])
            - roc_auc_score(sample["positive"], sample["baseline_oof"])
        )
    return tuple(np.quantile(differences, [0.025, 0.975]))


def build_readout_frame(path: Path, layer: int, primary_token: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations: list[dict[str, Any]] = []
    tokens: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        if int(row["layer"]) != layer:
            continue
        arm = next(
            (candidate for candidate in ARM_ORDER if str(row.get("message_hash", "")).endswith(f":{candidate}")),
            None,
        )
        if arm is None:
            raise ValueError("Could not recover intervention arm from message_hash")
        intervention_row_idx = int(row["row_idx"])
        base = {
            "original_row_idx": intervention_row_idx // len(ARM_ORDER),
            "arm": arm,
            "source": row.get("source"),
        }
        token_scores = row.get("token_scores", {})
        if primary_token not in token_scores:
            raise ValueError(f"Token {primary_token!r} is absent from the cached readouts")
        observations.append(
            {
                **base,
                "primary_logit": float(token_scores[primary_token]["max_logit"]),
                "primary_rank_strength": -np.log10(float(token_scores[primary_token]["best_rank"])),
                "falsity_concern": float(row["concept_scores"]["falsity_concern"]["mean_logit"]),
                "reality_testing": float(row["concept_scores"]["reality_testing"]["mean_logit"]),
            }
        )
        for token, metrics in token_scores.items():
            tokens.append(
                {
                    **base,
                    "token": token,
                    "max_logit": float(metrics["max_logit"]),
                    "rank_strength": -np.log10(float(metrics["best_rank"])),
                }
            )
    return pd.DataFrame(observations), pd.DataFrame(tokens)


def load_metadata(
    dataset_id: str,
    split: str,
    revision: str | None,
    metadata_path: Path | None,
) -> pd.DataFrame:
    if metadata_path is not None:
        records = read_jsonl(metadata_path)
        rows = []
        for fallback_index, row in enumerate(records):
            original_row_idx = int(row.get("original_row_idx", fallback_index))
            rows.append(
                {
                    "original_row_idx": original_row_idx,
                    "target_text": row["target_text"],
                    "annotation_score": row.get("annotation_score"),
                    "judge_confidence": row.get("judge_confidence"),
                    "metadata_source": row.get("source"),
                }
            )
        return (
            pd.DataFrame(rows)
            .drop_duplicates("original_row_idx")
            .sort_values("original_row_idx")
            .reset_index(drop=True)
        )

    dataset = load_dataset(dataset_id, split=split, revision=revision)
    rows = []
    for idx, row in enumerate(dataset):
        rows.append(
            {
                "original_row_idx": idx,
                "target_text": row["target_text"],
                "annotation_score": row.get("annotation_score"),
                "judge_confidence": row.get("judge_confidence"),
                "metadata_source": row.get("source"),
            }
        )
    return pd.DataFrame(rows)


def build_model(numeric: list[str], categorical: list[str]) -> Any:
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical,
            ),
        ]
    )
    return make_pipeline(
        preprocess,
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5_000),
    )


def grouped_predictions(
    frame: pd.DataFrame,
    features: list[str],
    folds: int,
    seed: int,
) -> np.ndarray:
    numeric = [column for column in features if column not in {"arm", "source"}]
    categorical = [column for column in features if column in {"arm", "source"}]
    model = build_model(numeric, categorical)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(
        model,
        frame[features],
        frame["positive"],
        groups=frame["original_row_idx"],
        cv=splitter,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]


def conditional_analysis(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    discordant_ids = frame.groupby("original_row_idx")["positive"].nunique()
    discordant_ids = discordant_ids[discordant_ids > 1].index
    discordant = frame[frame["original_row_idx"].isin(discordant_ids)].copy()
    discordant["primary_z"] = (
        discordant["primary_logit"] - discordant["primary_logit"].mean()
    ) / discordant["primary_logit"].std(ddof=0)
    arm_dummies = pd.get_dummies(discordant["arm"], prefix="arm", dtype=float)
    predictors = pd.concat([discordant[["primary_z"]], arm_dummies], axis=1)
    predictors = predictors.drop(columns=["arm_neutral"], errors="ignore")
    try:
        fit = ConditionalLogit(
            discordant["positive"].astype(float),
            predictors,
            groups=discordant["original_row_idx"],
        ).fit(disp=False)
        coefficient = float(fit.params["primary_z"])
        standard_error = float(fit.bse["primary_z"])
        result = {
            "n_discordant_messages": int(len(discordant_ids)),
            "n_observations": int(len(discordant)),
            "primary_log_odds_per_sd": coefficient,
            "primary_odds_ratio_per_sd": float(np.exp(coefficient)),
            "ci_low": float(np.exp(coefficient - 1.96 * standard_error)),
            "ci_high": float(np.exp(coefficient + 1.96 * standard_error)),
            "p_value": float(fit.pvalues["primary_z"]),
        }
    except Exception as error:
        result = {"error": str(error), "n_discordant_messages": int(len(discordant_ids))}

    paired_rows = []
    for group_id, group in discordant.groupby("original_row_idx"):
        paired_rows.append(
            {
                "original_row_idx": group_id,
                "endorsed_minus_nonendorsed_logit": (
                    group.loc[group["positive"] == 1, "primary_logit"].mean()
                    - group.loc[group["positive"] == 0, "primary_logit"].mean()
                ),
            }
        )
    return result, pd.DataFrame(paired_rows)


def token_placebos(token_frame: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    merged = token_frame.merge(
        behavior[["original_row_idx", "arm", "positive"]],
        on=["original_row_idx", "arm"],
        how="inner",
    )
    rows = []
    for token, group in merged.groupby("token"):
        auc = roc_auc_score(group["positive"], -group["max_logit"])
        discordant = group.groupby("original_row_idx").filter(lambda part: part["positive"].nunique() > 1)
        within = []
        for _, part in discordant.groupby("original_row_idx"):
            within.append(
                part.loc[part["positive"] == 1, "max_logit"].mean()
                - part.loc[part["positive"] == 0, "max_logit"].mean()
            )
        try:
            within_p = float(wilcoxon(within).pvalue)
        except ValueError:
            within_p = 1.0
        rows.append(
            {
                "token": token,
                "inverse_logit_auc": auc,
                "within_endorsed_minus_nonendorsed": float(np.mean(within)),
                "within_wilcoxon_p": within_p,
                "n_discordant_messages": len(within),
            }
        )
    result = pd.DataFrame(rows).sort_values("inverse_logit_auc", ascending=False)
    result["auc_rank"] = np.arange(1, len(result) + 1)
    return result


def adjusted_token_placebos(
    token_frame: pd.DataFrame,
    frame: pd.DataFrame,
    baseline_features: list[str],
    baseline_auc: float,
    folds: int,
    seed: int,
) -> pd.DataFrame:
    keys = ["original_row_idx", "arm"]
    base = frame.drop(columns=["primary_logit"]).copy()
    rows = []
    for token, values in token_frame.groupby("token"):
        candidate = base.merge(values[[*keys, "max_logit"]], on=keys, how="inner")
        candidate = candidate.rename(columns={"max_logit": "candidate_logit"})
        predictions = grouped_predictions(
            candidate,
            ["candidate_logit", *baseline_features],
            folds,
            seed,
        )
        auc = roc_auc_score(candidate["positive"], predictions)
        rows.append(
            {
                "token": token,
                "adjusted_grouped_cv_auc": auc,
                "auc_improvement_over_baseline": auc - baseline_auc,
            }
        )
    result = pd.DataFrame(rows).sort_values("auc_improvement_over_baseline", ascending=False)
    result["adjusted_auc_rank"] = np.arange(1, len(result) + 1)
    return result


def token_group_control(adjusted: pd.DataFrame) -> dict[str, Any]:
    scores = adjusted.set_index("token")["auc_improvement_over_baseline"]
    epistemic = scores.loc[EPISTEMIC_CONFUSION_TOKENS].to_numpy()
    placebo = scores.loc[LEXICAL_PLACEBO_TOKENS].to_numpy()
    observed = float(epistemic.mean() - placebo.mean())
    pooled = np.concatenate([epistemic, placebo])
    differences = []
    for selected in itertools.combinations(range(len(pooled)), len(epistemic)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(selected)] = True
        differences.append(float(pooled[mask].mean() - pooled[~mask].mean()))
    p_value = float(np.mean(np.abs(differences) >= abs(observed)))
    return {
        "epistemic_tokens": EPISTEMIC_CONFUSION_TOKENS,
        "lexical_placebo_tokens": LEXICAL_PLACEBO_TOKENS,
        "epistemic_mean_auc_improvement": float(epistemic.mean()),
        "placebo_mean_auc_improvement": float(placebo.mean()),
        "mean_difference": observed,
        "exact_two_sided_permutation_p": p_value,
        "assignment_count": len(differences),
    }


def group_permutation_p(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, np.ndarray]:
    wide = frame.pivot(index="original_row_idx", columns="arm", values="positive").reindex(columns=ARM_ORDER)
    ordered = frame.sort_values(["original_row_idx", "arm"]).copy()
    observed = roc_auc_score(ordered["positive"], -ordered["primary_logit"])
    null = np.empty(draws)
    values = wide.to_numpy()
    for draw in range(draws):
        permuted = wide.copy()
        permuted.iloc[:, :] = values[rng.permutation(len(values))]
        labels = (
            permuted.stack()
            .rename("permuted")
            .reset_index()
        )
        sample = ordered.merge(labels, on=["original_row_idx", "arm"], how="left")
        null[draw] = roc_auc_score(sample["permuted"], -sample["primary_logit"])
    p_value = (1 + np.sum(null >= observed)) / (draws + 1)
    return float(p_value), null


def quintile_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary_frame = frame.copy()
    summary_frame["signal_quintile"] = pd.qcut(
        summary_frame["primary_logit"], 5, labels=False, duplicates="drop"
    ) + 1
    rows = []
    for quintile, group in summary_frame.groupby("signal_quintile"):
        positives = int(group["positive"].sum())
        n = len(group)
        low, high = proportion_confint(positives, n, method="wilson")
        rows.append(
            {
                "signal_quintile": int(quintile),
                "n": n,
                "positives": positives,
                "positive_rate": positives / n,
                "wilson_low": low,
                "wilson_high": high,
                "mean_primary_logit": group["primary_logit"].mean(),
            }
        )
    return pd.DataFrame(rows)


def make_plot(
    frame: pd.DataFrame,
    quintiles: pd.DataFrame,
    paired: pd.DataFrame,
    placebo: pd.DataFrame,
    primary_token: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    yerr = np.vstack(
        [quintiles["positive_rate"] - quintiles["wilson_low"], quintiles["wilson_high"] - quintiles["positive_rate"]]
    )
    ax.errorbar(quintiles["signal_quintile"], quintiles["positive_rate"] * 100, yerr=yerr * 100, marker="o", capsize=3)
    ax.set(xlabel=f"{primary_token} logit quintile (low to high)", ylabel="Explicit endorsement (%)")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    values = paired["endorsed_minus_nonendorsed_logit"]
    ax.axvline(0, color="0.45", linewidth=1)
    ax.scatter(values, np.arange(len(values)), s=18, alpha=0.7)
    ax.set(xlabel="Endorsed minus non-endorsed logit", ylabel="Discordant message")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25)

    ax = axes[2]
    shown = placebo.head(12).sort_values("inverse_logit_auc")
    colors = ["C1" if token == primary_token else "C0" for token in shown["token"]]
    ax.barh(shown["token"], shown["inverse_logit_auc"], color=colors)
    ax.axvline(0.5, color="0.45", linewidth=1)
    ax.set(xlabel="AUC using lower token logit", ylabel="Predefined token")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether a pre-response J-space signal gates endorsement.")
    parser.add_argument("--readouts", type=Path, required=True)
    parser.add_argument("--judged-responses", type=Path, required=True)
    parser.add_argument("--prompt-variants", type=Path, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Optional local JSONL metadata. The context-intervention input is the exact "
            "offline source used for the paper; otherwise --dataset-id is downloaded."
        ),
    )
    parser.add_argument("--dataset-id", default="danielfein/WildDelusionVerified")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="Immutable Hugging Face revision used when --metadata is omitted.",
    )
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--primary-token", default=DEFAULT_PRIMARY_TOKEN)
    parser.add_argument("--score-column", default="openai_score")
    parser.add_argument("--positive-threshold", type=int, default=7)
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--bootstrap-draws", type=int, default=5_000)
    parser.add_argument("--permutation-draws", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hparams = vars(args).copy()
    for key in ["readouts", "judged_responses", "out_dir"]:
        hparams[key] = str(hparams[key])
    hparams["metadata"] = str(args.metadata) if args.metadata else None
    hparams["prompt_variants"] = str(args.prompt_variants) if args.prompt_variants else None
    hparams["primary_hypothesis"] = "Lower pre-response token logit predicts explicit endorsement."
    (args.out_dir / "hparams.json").write_text(json.dumps(hparams, indent=2), encoding="utf-8")

    readouts, token_frame = build_readout_frame(args.readouts, args.layer, args.primary_token)
    judged = pd.DataFrame(read_jsonl(args.judged_responses)).rename(columns={"intervention_arm": "arm"})
    judged["positive"] = (judged[args.score_column] >= args.positive_threshold).astype(int)
    metadata = load_metadata(
        args.dataset_id,
        args.dataset_split,
        args.dataset_revision,
        args.metadata,
    )
    frame = readouts.merge(
        judged[["original_row_idx", "arm", args.score_column, "positive"]],
        on=["original_row_idx", "arm"],
        how="inner",
    ).merge(metadata, on="original_row_idx", how="left")
    excluded_oversize_messages = 0
    if args.prompt_variants:
        prompts = pd.read_csv(args.prompt_variants)
        invalid = (
            prompts.loc[prompts["prompt_tokens"] > args.max_prompt_tokens, "row_idx"]
            // len(ARM_ORDER)
        )
        invalid_ids = set(invalid.astype(int))
        excluded_oversize_messages = len(invalid_ids)
        frame = frame[~frame["original_row_idx"].isin(invalid_ids)].copy()
        token_frame = token_frame[~token_frame["original_row_idx"].isin(invalid_ids)].copy()
    frame["log_text_chars"] = np.log1p(frame["target_text"].str.len())

    baseline_features = ["log_text_chars", "annotation_score", "judge_confidence", "arm", "source"]
    full_features = ["primary_logit", *baseline_features]
    frame["baseline_oof"] = grouped_predictions(
        frame, baseline_features, args.cv_folds, args.seed
    )
    frame["full_oof"] = grouped_predictions(frame, full_features, args.cv_folds, args.seed)
    baseline_auc = roc_auc_score(frame["positive"], frame["baseline_oof"])
    full_auc = roc_auc_score(frame["positive"], frame["full_oof"])
    ci_low, ci_high = bootstrap_grouped_auc_difference(
        frame, np.random.default_rng(args.seed), args.bootstrap_draws
    )

    conditional, paired = conditional_analysis(frame)
    placebo = token_placebos(token_frame, frame)
    adjusted_placebo = adjusted_token_placebos(
        token_frame,
        frame,
        baseline_features,
        baseline_auc,
        args.cv_folds,
        args.seed,
    )
    token_group_result = token_group_control(adjusted_placebo)
    permutation_p, null_auc = group_permutation_p(
        frame, np.random.default_rng(args.seed + 1), args.permutation_draws
    )
    quintiles = quintile_summary(frame)

    sensitivity = []
    for threshold in range(5, 10):
        outcome = (frame[args.score_column] >= threshold).astype(int)
        if outcome.nunique() < 2:
            continue
        rho, rho_p = spearmanr(frame["primary_logit"], frame[args.score_column])
        sensitivity.append(
            {
                "threshold": threshold,
                "positive_n": int(outcome.sum()),
                "inverse_logit_auc": roc_auc_score(outcome, -frame["primary_logit"]),
                "score_spearman_rho": rho,
                "score_spearman_p": rho_p,
            }
        )

    summary = {
        "n_messages": int(frame["original_row_idx"].nunique()),
        "n_observations": int(len(frame)),
        "positive_n": int(frame["positive"].sum()),
        "positive_rate": float(frame["positive"].mean()),
        "excluded_oversize_messages": excluded_oversize_messages,
        "unadjusted_inverse_logit_auc": float(roc_auc_score(frame["positive"], -frame["primary_logit"])),
        "baseline_grouped_cv_auc": float(baseline_auc),
        "full_grouped_cv_auc": float(full_auc),
        "grouped_cv_auc_improvement": float(full_auc - baseline_auc),
        "grouped_cv_auc_improvement_ci": [float(ci_low), float(ci_high)],
        "full_grouped_cv_average_precision": float(average_precision_score(frame["positive"], frame["full_oof"])),
        "full_grouped_cv_brier": float(brier_score_loss(frame["positive"], frame["full_oof"])),
        "group_pattern_permutation_p": permutation_p,
        "permutation_null_auc_95pct": [float(x) for x in np.quantile(null_auc, [0.025, 0.975])],
        "conditional_prompt_fixed": conditional,
        "primary_adjusted_token_rank": int(
            adjusted_placebo.loc[
                adjusted_placebo["token"] == args.primary_token, "adjusted_auc_rank"
            ].iloc[0]
        ),
        "adjusted_token_count": int(len(adjusted_placebo)),
        "pre_registered_token_group_control": token_group_result,
        "interpretation_rule": (
            "Support causal gating only if the prompt-fixed primary coefficient is negative "
            "and its two-sided p-value is below 0.05. Across-message AUC alone is predictive, not causal."
        ),
    }
    conditional_p = conditional.get("p_value", 1.0)
    conditional_beta = conditional.get("primary_log_odds_per_sd", 0.0)
    summary["causal_gating_supported"] = bool(conditional_beta < 0 and conditional_p < 0.05)

    frame.to_parquet(args.out_dir / "analysis_rows.parquet", index=False)
    paired.to_csv(args.out_dir / "prompt_fixed_differences.csv", index=False)
    placebo.to_csv(args.out_dir / "token_placebo_ranking.csv", index=False)
    adjusted_placebo.to_csv(args.out_dir / "adjusted_token_placebo_ranking.csv", index=False)
    quintiles.to_csv(args.out_dir / "signal_quintiles.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(args.out_dir / "threshold_sensitivity.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_plot(frame, quintiles, paired, placebo, args.primary_token, args.out_dir / "gating_controls.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
