from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.random_projection import GaussianRandomProjection


SEMANTIC_TOKENS = (
    "misinformation",
    "misunderstanding",
    "confusion",
    "misinterpretation",
    "false",
    "mistaken",
    "impossible",
    "hallucination",
)
PLACEBO_TOKENS = (
    "window",
    "table",
    "water",
    "music",
    "number",
    "question",
    "answer",
    "morning",
)
METRICS = ("max_logit", "mean_logit", "rank_strength")
BASELINE_NUMERIC = ("annotation_score", "judge_confidence", "log_text_chars")
BASELINE_CATEGORICAL = ("arm", "source")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_feature_rows(path: Path, layer: int) -> pd.DataFrame:
    rows = []
    for readout in read_jsonl(path):
        if int(readout["layer"]) != layer:
            continue
        row: dict[str, Any] = {}
        if "row_idx" in readout:
            row["row_idx"] = int(readout["row_idx"])
        else:
            row["original_row_idx"] = int(readout["original_row_idx"])
            row["intervention_arm"] = str(readout["condition"])
        for token, values in readout["token_scores"].items():
            row[f"{token}__max_logit"] = float(values["max_logit"])
            row[f"{token}__mean_logit"] = float(values["mean_logit"])
            row[f"{token}__rank_strength"] = -np.log10(float(values["best_rank"]))
        rows.append(row)
    frame = pd.DataFrame(rows)
    key_columns = (
        ["row_idx"]
        if "row_idx" in frame
        else ["original_row_idx", "intervention_arm"]
    )
    if frame.empty or frame.duplicated(key_columns).any():
        raise ValueError("Expected one readout per row at the requested layer")
    return frame


class ResidualizedSemantic(BaseEstimator, TransformerMixin):
    def __init__(self, semantic_count: int, alpha: float = 10.0):
        self.semantic_count = semantic_count
        self.alpha = alpha

    def fit(self, values: np.ndarray, y: np.ndarray | None = None) -> "ResidualizedSemantic":
        values = np.asarray(values, dtype=float)
        self.imputer_ = SimpleImputer(strategy="median").fit(values)
        values = self.imputer_.transform(values)
        semantic = values[:, : self.semantic_count]
        nuisance = values[:, self.semantic_count :]
        self.semantic_scaler_ = StandardScaler().fit(semantic)
        self.nuisance_scaler_ = StandardScaler().fit(nuisance)
        semantic = self.semantic_scaler_.transform(semantic)
        nuisance = self.nuisance_scaler_.transform(nuisance)
        self.regression_ = Ridge(alpha=self.alpha).fit(nuisance, semantic)
        residual = semantic - self.regression_.predict(nuisance)
        self.residual_scaler_ = StandardScaler().fit(residual)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = self.imputer_.transform(np.asarray(values, dtype=float))
        semantic = self.semantic_scaler_.transform(values[:, : self.semantic_count])
        nuisance = self.nuisance_scaler_.transform(values[:, self.semantic_count :])
        residual = semantic - self.regression_.predict(nuisance)
        return self.residual_scaler_.transform(residual)


def panel_transformer(
    panel: str,
    semantic: list[str],
    placebo: list[str],
    all_features: list[str],
    random_seed: int,
) -> tuple[Any, list[str]]:
    baseline = [
        (
            "baseline_numeric",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            list(BASELINE_NUMERIC),
        ),
        (
            "baseline_categorical",
            OneHotEncoder(handle_unknown="ignore"),
            list(BASELINE_CATEGORICAL),
        ),
    ]
    needed = list(BASELINE_NUMERIC) + list(BASELINE_CATEGORICAL)
    if panel == "baseline":
        return ColumnTransformer(baseline), needed
    if panel == "semantic":
        extra = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        return ColumnTransformer([*baseline, ("jspace", extra, semantic)]), needed + semantic
    if panel == "placebo":
        extra = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        return ColumnTransformer([*baseline, ("jspace", extra, placebo)]), needed + placebo
    if panel == "semantic_plus_placebo":
        columns = semantic + placebo
        extra = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        return ColumnTransformer([*baseline, ("jspace", extra, columns)]), needed + columns
    if panel == "semantic_residual_placebo":
        columns = semantic + placebo
        extra = ResidualizedSemantic(semantic_count=len(semantic))
        return ColumnTransformer([*baseline, ("jspace", extra, columns)]), needed + columns
    if panel == "semantic_residual_all":
        nuisance = [feature for feature in all_features if feature not in semantic]
        columns = semantic + nuisance
        extra = ResidualizedSemantic(semantic_count=len(semantic))
        return ColumnTransformer([*baseline, ("jspace", extra, columns)]), needed + columns
    if panel == "pca":
        extra = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PCA(n_components=len(semantic), random_state=random_seed),
        )
        return ColumnTransformer([*baseline, ("jspace", extra, all_features)]), needed + all_features
    if panel.startswith("random_projection"):
        extra = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            GaussianRandomProjection(n_components=len(semantic), random_state=random_seed),
        )
        return ColumnTransformer([*baseline, ("jspace", extra, all_features)]), needed + all_features
    raise ValueError(f"Unknown panel: {panel}")


def build_model(
    panel: str,
    c_value: float,
    semantic: list[str],
    placebo: list[str],
    all_features: list[str],
    random_seed: int,
) -> tuple[Any, list[str]]:
    transformer, columns = panel_transformer(
        panel, semantic, placebo, all_features, random_seed
    )
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=10_000,
        solver="lbfgs",
        random_state=random_seed,
    )
    return make_pipeline(transformer, classifier), columns


def select_c(
    train: pd.DataFrame,
    panel: str,
    c_grid: list[float],
    semantic: list[str],
    placebo: list[str],
    all_features: list[str],
    folds: int,
    split_seed: int,
    model_seed: int,
) -> float:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=split_seed)
    rows = []
    for c_value in c_grid:
        aucs = []
        for train_idx, test_idx in splitter.split(
            train, train["positive"], groups=train["original_row_idx"]
        ):
            inner_train = train.iloc[train_idx]
            inner_test = train.iloc[test_idx]
            model, columns = build_model(
                panel, c_value, semantic, placebo, all_features, model_seed
            )
            model.fit(inner_train[columns], inner_train["positive"])
            score = model.decision_function(inner_test[columns])
            aucs.append(roc_auc_score(inner_test["positive"], score))
        rows.append((c_value, float(np.mean(aucs))))
    return max(rows, key=lambda item: item[1])[0]


def grouped_nested_oof(
    frame: pd.DataFrame,
    panel: str,
    c_grid: list[float],
    semantic: list[str],
    placebo: list[str],
    all_features: list[str],
    outer_folds: int,
    inner_folds: int,
    split_seed: int,
    model_seed: int,
) -> tuple[np.ndarray, list[float]]:
    splitter = StratifiedGroupKFold(
        n_splits=outer_folds, shuffle=True, random_state=split_seed
    )
    scores = np.full(len(frame), np.nan)
    selected = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(frame, frame["positive"], groups=frame["original_row_idx"])
    ):
        train = frame.iloc[train_idx]
        test = frame.iloc[test_idx]
        inner_split_seed = split_seed + fold
        c_value = select_c(
            train,
            panel,
            c_grid,
            semantic,
            placebo,
            all_features,
            inner_folds,
            inner_split_seed,
            model_seed,
        )
        model, columns = build_model(
            panel, c_value, semantic, placebo, all_features, model_seed
        )
        model.fit(train[columns], train["positive"])
        scores[test_idx] = model.decision_function(test[columns])
        selected.append(c_value)
    if np.isnan(scores).any():
        raise RuntimeError("OOF prediction is incomplete")
    return scores, selected


def bootstrap_metrics(
    frame: pd.DataFrame,
    score_columns: list[str],
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    group_rows = {
        group_id: indices.to_numpy()
        for group_id, indices in frame.groupby("original_row_idx").groups.items()
    }
    group_ids = np.array(list(group_rows))
    estimates = {column: [] for column in score_columns}
    for _ in range(draws):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        indices = np.concatenate([group_rows[group_id] for group_id in sampled])
        outcome = frame.loc[indices, "positive"].to_numpy()
        if np.unique(outcome).size < 2:
            continue
        for column in score_columns:
            estimates[column].append(
                roc_auc_score(outcome, frame.loc[indices, column].to_numpy())
            )
    summary = []
    for column, values in estimates.items():
        values_array = np.asarray(values)
        low, high = np.quantile(values_array, [0.025, 0.975])
        summary.append({"panel": column, "ci_low": low, "ci_high": high})
    comparisons = []
    baseline = np.asarray(estimates["baseline"])
    for column in score_columns:
        if column == "baseline":
            continue
        difference = np.asarray(estimates[column]) - baseline
        low, high = np.quantile(difference, [0.025, 0.975])
        comparisons.append(
            {
                "contrast": f"{column}_minus_baseline",
                "ci_low": low,
                "ci_high": high,
                "probability_greater_than_zero": float(np.mean(difference > 0)),
            }
        )
    semantic_placebo = np.asarray(estimates["semantic"]) - np.asarray(estimates["placebo"])
    low, high = np.quantile(semantic_placebo, [0.025, 0.975])
    comparisons.append(
        {
            "contrast": "semantic_minus_placebo",
            "ci_low": low,
            "ci_high": high,
            "probability_greater_than_zero": float(np.mean(semantic_placebo > 0)),
        }
    )
    return pd.DataFrame(summary), pd.DataFrame(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test semantic specificity of same-model J-space behavior predictors."
    )
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--readouts", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--selected-original-indices", type=Path, default=None)
    parser.add_argument("--positive-threshold", type=int, default=7)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--random-projections", type=int, default=10)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    prompts = pd.DataFrame(read_jsonl(args.prompts)).reset_index(names="row_idx")
    if args.selected_original_indices is not None:
        selected = {
            int(value)
            for value in args.selected_original_indices.read_text(encoding="utf-8").splitlines()
            if value.strip()
        }
        prompts = prompts[prompts["original_row_idx"].isin(selected)].copy()
    judgments = pd.DataFrame(read_jsonl(args.judgments))
    judgments = judgments.rename(columns={"intervention_arm": "arm"})
    judgments["positive"] = (judgments["openai_score"] >= args.positive_threshold).astype(int)
    readouts = load_feature_rows(args.readouts, args.layer)
    if "row_idx" in readouts:
        frame = prompts.merge(readouts, on="row_idx", validate="one_to_one")
    else:
        frame = prompts.merge(
            readouts,
            on=["original_row_idx", "intervention_arm"],
            validate="one_to_one",
        )
    frame = frame.merge(
        judgments[
            ["original_row_idx", "arm", "target_text", "openai_score", "positive"]
        ],
        left_on=["original_row_idx", "intervention_arm", "target_text"],
        right_on=["original_row_idx", "arm", "target_text"],
        validate="one_to_one",
    )
    frame["arm"] = frame["intervention_arm"]
    frame["log_text_chars"] = np.log1p(frame["target_text"].str.len())

    all_features = sorted(column for column in frame if "__" in column)
    semantic = [
        f"{token}__{metric}"
        for token in SEMANTIC_TOKENS
        for metric in METRICS
        if f"{token}__{metric}" in frame
    ]
    placebo = [
        f"{token}__{metric}"
        for token in PLACEBO_TOKENS
        for metric in METRICS
        if f"{token}__{metric}" in frame
    ]
    if len(semantic) != len(placebo):
        raise ValueError(f"Expected capacity-matched panels, got {len(semantic)} and {len(placebo)}")

    panels = [
        "baseline",
        "semantic",
        "placebo",
        "semantic_plus_placebo",
        "semantic_residual_placebo",
        "semantic_residual_all",
        "pca",
    ] + [f"random_projection_{index:02d}" for index in range(args.random_projections)]
    selected_rows = []
    for panel_index, panel in enumerate(panels):
        score, selected_c = grouped_nested_oof(
            frame,
            panel,
            args.c_grid,
            semantic,
            placebo,
            all_features,
            args.outer_folds,
            args.inner_folds,
            args.seed,
            args.seed + panel_index,
        )
        frame[panel] = score
        selected_rows.extend(
            {"panel": panel, "outer_fold": fold, "selected_C": value}
            for fold, value in enumerate(selected_c)
        )

    score_columns = panels
    ci, comparisons = bootstrap_metrics(
        frame, score_columns, args.bootstrap_draws, args.seed
    )
    performance = []
    for panel in panels:
        auc = roc_auc_score(frame["positive"], frame[panel])
        ap = average_precision_score(frame["positive"], frame[panel])
        rho, rho_p = spearmanr(frame["openai_score"], frame[panel])
        performance.append(
            {
                "panel": panel,
                "auc": auc,
                "average_precision": ap,
                "graded_spearman_rho": rho,
                "graded_spearman_p": rho_p,
            }
        )
    performance = pd.DataFrame(performance).merge(ci, on="panel")
    baseline_auc = performance.set_index("panel").loc["baseline", "auc"]
    performance["auc_minus_baseline"] = performance["auc"] - baseline_auc
    comparisons["point_difference"] = comparisons["contrast"].map(
        {
            f"{panel}_minus_baseline": (
                performance.set_index("panel").loc[panel, "auc"] - baseline_auc
            )
            for panel in panels
            if panel != "baseline"
        }
        | {
            "semantic_minus_placebo": (
                performance.set_index("panel").loc["semantic", "auc"]
                - performance.set_index("panel").loc["placebo", "auc"]
            )
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    performance.to_csv(args.output_dir / "panel_performance.csv", index=False)
    comparisons.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(args.output_dir / "selected_regularization.csv", index=False)
    frame[
        ["original_row_idx", "arm", "source", "target_text", "openai_score", "positive", *panels]
    ].to_parquet(args.output_dir / "oof_predictions.parquet", index=False)
    random_auc = performance[
        performance["panel"].str.startswith("random_projection")
    ]["auc"]
    summary = {
        "n": len(frame),
        "prompt_groups": int(frame["original_row_idx"].nunique()),
        "positive_n": int(frame["positive"].sum()),
        "positive_rate": float(frame["positive"].mean()),
        "semantic_features": semantic,
        "placebo_features": placebo,
        "all_jspace_features": len(all_features),
        "random_projection_auc_median": float(random_auc.median()),
        "random_projection_auc_min": float(random_auc.min()),
        "random_projection_auc_max": float(random_auc.max()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "hparams.json").write_text(
        json.dumps(
            {
                "outcome": f"openai_score >= {args.positive_threshold}",
                "evaluation": "nested grouped out-of-fold prediction",
                "group": "original_row_idx",
                "outer_folds": args.outer_folds,
                "inner_folds": args.inner_folds,
                "c_grid": args.c_grid,
                "bootstrap": "prompt-group resampling with all three arms retained",
                "bootstrap_draws": args.bootstrap_draws,
                "random_projections": args.random_projections,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plotted = performance[~performance["panel"].str.startswith("random_projection")].copy()
    plotted["label"] = plotted["panel"].str.replace("_", " ").str.title()
    plotted = plotted.sort_values("auc")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y = np.arange(len(plotted))
    ax.errorbar(
        plotted["auc"],
        y,
        xerr=np.vstack([plotted["auc"] - plotted["ci_low"], plotted["ci_high"] - plotted["auc"]]),
        fmt="o",
        capsize=4,
        color="#2878B5",
    )
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.axvspan(random_auc.min(), random_auc.max(), color="#E6A23C", alpha=0.18)
    ax.axvline(random_auc.median(), color="#E6A23C", linewidth=2, label="Random-projection median")
    ax.set_yticks(y, plotted["label"])
    ax.set_xlabel("Nested grouped out-of-fold AUROC")
    ax.set_title("Semantic specificity of J-space behavior predictors")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.output_dir / "semantic_specificity.png", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
