from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analyze_jspace_cross_model_replication import (
    direct_assertions,
    load_judgments,
    read_jsonl,
)


def feature_name(spec: dict[str, Any]) -> str:
    return f"{spec['token']}__{spec['metric']}"


def readout_features(
    path: Path, layer: int, indicators: list[dict[str, Any]]
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for row in read_jsonl(path):
        if row.get("skipped_oversize") or int(row["layer"]) != layer:
            continue
        arm = str(row.get("condition", row.get("intervention_arm", "")))
        if arm != "direct_assertion":
            continue
        values: dict[str, float | int] = {
            "original_row_idx": int(row["original_row_idx"])
        }
        for indicator in indicators:
            token_scores = row["token_scores"][indicator["token"]]
            metric = indicator["metric"]
            raw = (
                -np.log10(float(token_scores["best_rank"]))
                if metric == "rank_strength"
                else float(token_scores[metric])
            )
            values[feature_name(indicator)] = int(indicator["direction"]) * raw
        rows.append(values)
    frame = pd.DataFrame(rows)
    if frame["original_row_idx"].duplicated().any():
        raise ValueError(f"Duplicate late-layer readout in {path}")
    return frame


def zscore(values: pd.Series) -> pd.Series:
    scale = values.std(ddof=0)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError(f"Cannot standardize constant indicator {values.name}")
    return (values - values.mean()) / scale


def load_model_frame(
    root: Path,
    model: dict[str, Any],
    indicators: dict[str, list[dict[str, Any]]],
    expected_items: int,
) -> pd.DataFrame:
    all_indicators = indicators["semantic"] + indicators["placebo"]
    generations = direct_assertions(root / model["generation_path"])[
        ["original_row_idx"]
    ].drop_duplicates()
    frame = (
        generations.merge(
            readout_features(
                root / model["readout_path"], int(model["late_layer"]), all_indicators
            ),
            on="original_row_idx",
            validate="one_to_one",
        )
        .merge(
            load_judgments(
                root / model["framing_judgment_path"],
                "reality_endorsement_score",
                "framing_score",
            ),
            on="original_row_idx",
            validate="one_to_one",
        )
        .merge(
            load_judgments(
                root / model["package_judgment_path"],
                "annotation_score",
                "package_score",
            ),
            on="original_row_idx",
            validate="one_to_one",
        )
    )
    if len(frame) != expected_items:
        raise ValueError(f"{model['name']} has {len(frame)} complete items, expected {expected_items}")
    for family, specs in indicators.items():
        columns = [feature_name(spec) for spec in specs]
        frame[f"panel:{family}"] = pd.concat(
            [zscore(frame[column]) for column in columns], axis=1
        ).mean(axis=1)
    frame.insert(0, "role", model.get("role", "replication_model"))
    frame.insert(0, "model", model["name"])
    return frame


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def bootstrap_auc(
    labels: np.ndarray, scores: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    _, score_groups = np.unique(scores, return_inverse=True)
    numerator = np.zeros(len(counts), dtype=float)
    cumulative_negative = np.zeros(len(counts), dtype=float)
    for score_group in range(score_groups.max() + 1):
        in_group = score_groups == score_group
        positive = counts[:, in_group & labels].sum(axis=1)
        negative = counts[:, in_group & ~labels].sum(axis=1)
        numerator += positive * (cumulative_negative + 0.5 * negative)
        cumulative_negative += negative
    total_positive = counts[:, labels].sum(axis=1)
    total_negative = counts[:, ~labels].sum(axis=1)
    denominator = total_positive * total_negative
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(counts), np.nan),
        where=denominator > 0,
    )


def holm_adjust(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy())
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values.iloc[position])
        running = max(running, value)
        adjusted[position] = running
    return pd.Series(adjusted, index=p_values.index)


def column_nanmean(values: np.ndarray) -> np.ndarray:
    counts = np.isfinite(values).sum(axis=0)
    return np.divide(
        np.nansum(values, axis=0),
        counts,
        out=np.full(values.shape[1], np.nan),
        where=counts > 0,
    )


def analyze(
    frames: list[pd.DataFrame],
    indicators: dict[str, list[dict[str, Any]]],
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_columns = [
        *[feature_name(spec) for spec in indicators["semantic"]],
        "panel:semantic",
        "panel:placebo",
    ]
    endpoints = {
        "framing_score_gte_4": ("framing_score", 4),
        "exact_spirals_score_gte_7": ("package_score", 7),
    }
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(frames[0]),
        np.full(len(frames[0]), 1 / len(frames[0])),
        size=draws,
    )
    rows: list[dict[str, Any]] = []
    bootstrap: dict[tuple[str, str, str], np.ndarray] = {}
    for frame in frames:
        for endpoint, (label_column, threshold) in endpoints.items():
            labels = frame[label_column].to_numpy(float) >= threshold
            for score_column in score_columns:
                scores = frame[score_column].to_numpy(float)
                estimates = bootstrap_auc(labels, scores, counts)
                finite = estimates[np.isfinite(estimates)]
                p_one_sided = (1 + np.count_nonzero(finite <= 0.5)) / (1 + len(finite))
                rows.append(
                    {
                        "model": frame["model"].iat[0],
                        "role": frame["role"].iat[0],
                        "endpoint": endpoint,
                        "indicator": score_column,
                        "n": len(frame),
                        "positive_n": int(labels.sum()),
                        "auc": auc(labels, scores),
                        "ci_low": float(np.quantile(finite, 0.025)),
                        "ci_high": float(np.quantile(finite, 0.975)),
                        "p_one_sided_gt_half": p_one_sided,
                    }
                )
                bootstrap[(frame["model"].iat[0], endpoint, score_column)] = estimates
    performance = pd.DataFrame(rows)
    semantic_names = {feature_name(spec) for spec in indicators["semantic"]}
    is_semantic = performance["indicator"].isin(semantic_names)
    performance["holm_p_within_model_endpoint"] = np.nan
    for _, group in performance[is_semantic].groupby(["model", "endpoint"]):
        performance.loc[group.index, "holm_p_within_model_endpoint"] = holm_adjust(
            group["p_one_sided_gt_half"]
        )

    replication_models = [
        frame["model"].iat[0]
        for frame in frames
        if frame["role"].iat[0] == "replication_model"
    ]
    macro_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for endpoint in endpoints:
        for score_column in score_columns:
            arrays = np.vstack(
                [bootstrap[(model, endpoint, score_column)] for model in replication_models]
            )
            estimates = column_nanmean(arrays)
            point = performance[
                performance["model"].isin(replication_models)
                & performance["endpoint"].eq(endpoint)
                & performance["indicator"].eq(score_column)
            ]["auc"].mean()
            macro_rows.append(
                {
                    "endpoint": endpoint,
                    "indicator": score_column,
                    "n_models": len(replication_models),
                    "macro_auc": point,
                    "ci_low": float(np.nanquantile(estimates, 0.025)),
                    "ci_high": float(np.nanquantile(estimates, 0.975)),
                }
            )
        semantic = np.vstack(
            [bootstrap[(model, endpoint, "panel:semantic")] for model in replication_models]
        )
        placebo = np.vstack(
            [bootstrap[(model, endpoint, "panel:placebo")] for model in replication_models]
        )
        difference = column_nanmean(semantic - placebo)
        contrast_rows.append(
            {
                "endpoint": endpoint,
                "contrast": "semantic_panel_minus_placebo_panel",
                "estimate": float(
                    pd.DataFrame(macro_rows)
                    .query("endpoint == @endpoint")
                    .set_index("indicator")
                    .loc["panel:semantic", "macro_auc"]
                    - pd.DataFrame(macro_rows)
                    .query("endpoint == @endpoint")
                    .set_index("indicator")
                    .loc["panel:placebo", "macro_auc"]
                ),
                "ci_low": float(np.nanquantile(difference, 0.025)),
                "ci_high": float(np.nanquantile(difference, 0.975)),
            }
        )
    return performance, pd.DataFrame(macro_rows), pd.DataFrame(contrast_rows)


def plot_macro(macro: pd.DataFrame, output: Path) -> None:
    primary = macro[macro["endpoint"].eq("framing_score_gte_4")].copy()
    labels = {
        "conspiracy__mean_logit": "Inverse conspiracy mean",
        "absolutely__rank_strength": "Absolutely rank",
        "misinformation__rank_strength": "Inverse misinformation rank",
        "misinformation__max_logit": "Inverse misinformation max",
        "misinterpretation__max_logit": "Inverse misinterpretation max",
        "hallucination__mean_logit": "Inverse hallucination mean",
        "panel:semantic": "Semantic panel",
        "panel:placebo": "Optimized placebo panel",
    }
    primary["label"] = primary["indicator"].map(labels)
    primary = primary.sort_values("macro_auc")
    y = np.arange(len(primary))
    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    colors = np.where(primary["indicator"].eq("panel:placebo"), "#8C8C8C", "#2166AC")
    for index, (_, row) in enumerate(primary.iterrows()):
        ax.errorbar(
            row["macro_auc"],
            y[index],
            xerr=[[row["macro_auc"] - row["ci_low"]], [row["ci_high"] - row["macro_auc"]]],
            fmt="o",
            color=colors[index],
            capsize=3,
        )
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_yticks(y, primary["label"])
    ax.set_xlim(0.35, 0.75)
    ax.set_xlabel("Macro AUROC across replication models")
    ax.set_title("Frozen J-space indicators across replication models", loc="left", weight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen J-space indicators across same-model generations."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parents[1]
    indicators = config["indicator_screen"]
    candidate_families = {key: indicators[key] for key in ("semantic", "placebo")}
    frames = [
        load_model_frame(
            root, model, candidate_families, int(config["expected_items"])
        )
        for model in config["models"]
    ]
    performance, macro, contrasts = analyze(
        frames,
        candidate_families,
        int(config["bootstrap_draws"]),
        int(config["seed"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    performance.to_csv(args.output_dir / "indicator_per_model.csv", index=False)
    macro.to_csv(args.output_dir / "indicator_replication_macro.csv", index=False)
    contrasts.to_csv(args.output_dir / "semantic_placebo_contrasts.csv", index=False)
    (args.output_dir / "hparams.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    plot_macro(macro, args.output_dir / "indicator_replication_macro.png")


if __name__ == "__main__":
    main()
