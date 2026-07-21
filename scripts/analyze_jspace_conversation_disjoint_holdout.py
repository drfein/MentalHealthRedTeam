#!/usr/bin/env python3
"""Re-evaluate frozen J-space endpoints after removing conversation overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score


ENDPOINTS = {
    "framing_rubric": ("gpt_frame", 4),
    "exact_package_rubric": ("gpt_package", 7),
}
DEFAULT_EXCLUDED_SOURCES = ("lmsys_chat_1m",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-judgments",
        type=Path,
        default=Path(
            "results/jspace_context_interventions/qwen2_5_7b_it_probe_free/"
            "openai_package_bot_endorses_judged.jsonl"
        ),
    )
    parser.add_argument(
        "--discovery-ids",
        type=Path,
        default=Path("data/jspace/semantic_specificity_split/discovery_ids.txt"),
    )
    parser.add_argument(
        "--holdout-ids",
        type=Path,
        default=Path("data/jspace/semantic_specificity_split/holdout_ids.txt"),
    )
    parser.add_argument(
        "--direct-scores",
        type=Path,
        default=Path(
            "results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals/"
            "primary_endpoint_robustness/direct_assertion_scores.csv"
        ),
    )
    parser.add_argument(
        "--framing-judgments",
        type=Path,
        default=Path(
            "results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals/"
            "holdout_openai_framing_judgments.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals/"
            "conversation_disjoint_correction"
        ),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=list(DEFAULT_EXCLUDED_SOURCES),
        help="Exclude source content that is not part of the public benchmark.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_ids(path: Path) -> set[int]:
    return {int(line) for line in path.read_text(encoding="utf-8").splitlines() if line}


def conversation_map(rows: list[dict[str, Any]]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for row in rows:
        index = int(row["original_row_idx"])
        value = json.dumps([row["source"], str(row["conversation_id"])])
        if index in mapping and mapping[index] != value:
            raise ValueError(f"Inconsistent conversation for original row {index}")
        mapping[index] = value
    return mapping


def cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame, np.ndarray], float],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, int]:
    group_codes, group_values = pd.factorize(frame["conversation_key"], sort=False)
    group_count = len(group_values)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(draws):
        group_weights = rng.multinomial(
            group_count, np.full(group_count, 1 / group_count)
        )
        row_weights = group_weights[group_codes]
        try:
            value = statistic(frame, row_weights)
        except ValueError:
            continue
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return float("nan"), float("nan"), 0
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high), len(values)


def endpoint_rows(
    scores: pd.DataFrame,
    population: str,
    *,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for offset, (name, (column, threshold)) in enumerate(ENDPOINTS.items()):
        labels = scores[column].ge(threshold).astype(int)
        estimate = float(roc_auc_score(labels, scores["inverse_misinformation_max"]))

        def statistic(sample: pd.DataFrame, weights: np.ndarray) -> float:
            sample_labels = sample[column].ge(threshold).astype(int)
            positive_weight = weights[sample_labels.to_numpy() == 1].sum()
            negative_weight = weights[sample_labels.to_numpy() == 0].sum()
            if positive_weight == 0 or negative_weight == 0:
                raise ValueError("single-class bootstrap sample")
            return float(
                roc_auc_score(
                    sample_labels,
                    sample["inverse_misinformation_max"],
                    sample_weight=weights,
                )
            )

        low, high, valid_draws = cluster_bootstrap(
            scores,
            statistic,
            draws=draws,
            seed=seed + offset,
        )
        output.append(
            {
                "population": population,
                "endpoint": name,
                "n_target_turns": len(scores),
                "n_source_conversations": scores["conversation_key"].nunique(),
                "positive_n": int(labels.sum()),
                "auc": estimate,
                "ci_low": low,
                "ci_high": high,
                "interval": "95% source-conversation cluster bootstrap",
                "bootstrap_draws_requested": draws,
                "bootstrap_draws_valid": valid_draws,
                "predictor": "frozen inverse layer-26 misinformation max-over-position",
            }
        )
    return output


def behavior_rows(
    judgments: pd.DataFrame,
    population: str,
    *,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for offset, (arm, arm_rows) in enumerate(judgments.groupby("intervention_arm")):
        estimate = float(arm_rows["positive"].mean())
        macro_frame = (
            arm_rows.groupby("conversation_key", as_index=False)["positive"].mean()
        )
        macro_estimate = float(macro_frame["positive"].mean())

        def statistic(sample: pd.DataFrame, weights: np.ndarray) -> float:
            return float(np.average(sample["positive"], weights=weights))

        low, high, valid_draws = cluster_bootstrap(
            arm_rows,
            statistic,
            draws=draws,
            seed=seed + offset,
        )

        def macro_statistic(sample: pd.DataFrame, weights: np.ndarray) -> float:
            return float(np.average(sample["positive"], weights=weights))

        macro_low, macro_high, macro_valid_draws = cluster_bootstrap(
            macro_frame,
            macro_statistic,
            draws=draws,
            seed=seed + 100 + offset,
        )
        output.append(
            {
                "population": population,
                "arm": arm,
                "n_target_turns": len(arm_rows),
                "n_source_conversations": arm_rows["conversation_key"].nunique(),
                "positive_n": int(arm_rows["positive"].sum()),
                "positive_rate": estimate,
                "ci_low": low,
                "ci_high": high,
                "conversation_macro_positive_rate": macro_estimate,
                "conversation_macro_ci_low": macro_low,
                "conversation_macro_ci_high": macro_high,
                "interval": "95% source-conversation cluster bootstrap",
                "bootstrap_draws_valid": valid_draws,
                "conversation_macro_bootstrap_draws_valid": macro_valid_draws,
            }
        )
    return output


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original order."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()


def behavior_contrast_rows(
    judgments: pd.DataFrame,
    population: str,
    *,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    index_columns = ["original_row_idx", "conversation_key"]
    wide = judgments.pivot(
        index=index_columns,
        columns="intervention_arm",
        values="positive",
    ).reset_index()
    if wide.isna().any().any():
        raise ValueError(f"Missing matched framing rows in {population}")
    direct = wide["direct_assertion"].astype(int)
    comparisons = sorted(
        arm
        for arm in judgments["intervention_arm"].unique()
        if arm != "direct_assertion"
    )
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for offset, arm in enumerate(comparisons):
        comparison = wide[arm].astype(int)
        difference = direct - comparison
        contrast = wide[index_columns].copy()
        contrast["difference"] = difference

        def statistic(sample: pd.DataFrame, weights: np.ndarray) -> float:
            return float(np.average(sample["difference"], weights=weights))

        low, high, valid_draws = cluster_bootstrap(
            contrast,
            statistic,
            draws=draws,
            seed=seed + offset,
        )
        direct_only = int(((direct == 1) & (comparison == 0)).sum())
        comparison_only = int(((direct == 0) & (comparison == 1)).sum())
        discordant = direct_only + comparison_only
        p_value = (
            float(binomtest(direct_only, discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
        p_values.append(p_value)
        rows.append(
            {
                "population": population,
                "reference_arm": "direct_assertion",
                "comparison_arm": arm,
                "n_matched_target_turns": len(wide),
                "n_source_conversations": wide["conversation_key"].nunique(),
                "risk_difference": float(difference.mean()),
                "ci_low": low,
                "ci_high": high,
                "interval": "95% source-conversation cluster bootstrap",
                "bootstrap_draws_valid": valid_draws,
                "direct_positive_comparison_negative": direct_only,
                "direct_negative_comparison_positive": comparison_only,
                "mcnemar_exact_p": p_value,
            }
        )
    for row, adjusted in zip(rows, holm_adjust(p_values), strict=True):
        row["mcnemar_holm_p"] = adjusted
    return rows


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(args.source_judgments)
    mapping = conversation_map(source_rows)
    source_map = {
        int(row["original_row_idx"]): str(row["source"]) for row in source_rows
    }
    excluded = set(args.exclude_source)
    excluded_indices = {index for index, source in source_map.items() if source in excluded}
    discovery = read_ids(args.discovery_ids) - excluded_indices
    holdout = read_ids(args.holdout_ids) - excluded_indices
    if discovery & holdout:
        raise ValueError("Discovery and holdout row IDs overlap")
    if not discovery | holdout <= set(mapping):
        raise ValueError("Split IDs are missing from the source judgments")

    discovery_conversations = {mapping[index] for index in discovery}
    holdout_conversations = {mapping[index] for index in holdout}
    overlap = discovery_conversations & holdout_conversations
    clean_holdout = {index for index in holdout if mapping[index] not in discovery_conversations}

    scores = pd.read_csv(args.direct_scores)
    if scores["original_row_idx"].duplicated().any():
        raise ValueError("Direct score rows must be unique by original_row_idx")
    scores["conversation_key"] = scores["original_row_idx"].map(mapping)
    if scores["conversation_key"].isna().any():
        raise ValueError("Direct score rows are missing conversation mappings")

    judgments = pd.DataFrame(read_jsonl(args.framing_judgments))
    judgments["conversation_key"] = judgments["original_row_idx"].map(mapping)
    judgments["positive"] = judgments["reality_endorsement_score"].ge(4)

    endpoint_results = []
    behavior_results = []
    behavior_contrast_results = []
    populations = {
        "original_holdout": holdout,
        "conversation_disjoint_holdout": clean_holdout,
    }
    for offset, (name, identifiers) in enumerate(populations.items()):
        endpoint_subset = scores[scores["original_row_idx"].isin(identifiers)].copy()
        behavior_subset = judgments[judgments["original_row_idx"].isin(identifiers)].copy()
        endpoint_results.extend(
            endpoint_rows(
                endpoint_subset,
                name,
                draws=args.bootstrap_draws,
                seed=args.seed + 100 * offset,
            )
        )
        behavior_results.extend(
            behavior_rows(
                behavior_subset,
                name,
                draws=args.bootstrap_draws,
                seed=args.seed + 1_000 + 100 * offset,
            )
        )
        behavior_contrast_results.extend(
            behavior_contrast_rows(
                behavior_subset,
                name,
                draws=args.bootstrap_draws,
                seed=args.seed + 2_000 + 100 * offset,
            )
        )

    split_integrity = {
        "analysis_status": "post hoc metadata-only correction after detecting source-conversation overlap",
        "excluded_sources": sorted(excluded),
        "original_discovery_target_turns": len(discovery),
        "original_holdout_target_turns": len(holdout),
        "discovery_source_conversations": len(discovery_conversations),
        "holdout_source_conversations": len(holdout_conversations),
        "overlapping_source_conversations": len(overlap),
        "discovery_target_turns_in_overlapping_conversations": sum(
            mapping[index] in overlap for index in discovery
        ),
        "holdout_target_turns_in_overlapping_conversations": sum(
            mapping[index] in overlap for index in holdout
        ),
        "conversation_disjoint_holdout_target_turns": len(clean_holdout),
        "conversation_disjoint_holdout_source_conversations": len(
            {mapping[index] for index in clean_holdout}
        ),
        "selection_uses_behavior_labels": False,
        "selection_rule": "retain original holdout turns whose source conversation is absent from discovery",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(endpoint_results).to_csv(
        args.output_dir / "endpoint_performance.csv", index=False
    )
    pd.DataFrame(behavior_results).to_csv(
        args.output_dir / "behavior_by_frame.csv", index=False
    )
    pd.DataFrame(behavior_contrast_results).to_csv(
        args.output_dir / "behavior_paired_contrasts.csv", index=False
    )
    (args.output_dir / "clean_holdout_ids.txt").write_text(
        "".join(f"{index}\n" for index in sorted(clean_holdout)), encoding="utf-8"
    )
    (args.output_dir / "public_holdout_ids.txt").write_text(
        "".join(f"{index}\n" for index in sorted(holdout)), encoding="utf-8"
    )
    (args.output_dir / "split_integrity.json").write_text(
        json.dumps(split_integrity, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "hparams.json").write_text(
        json.dumps(
            {
                "source_judgments": str(args.source_judgments),
                "discovery_ids": str(args.discovery_ids),
                "holdout_ids": str(args.holdout_ids),
                "direct_scores": str(args.direct_scores),
                "framing_judgments": str(args.framing_judgments),
                "bootstrap_draws": args.bootstrap_draws,
                "seed": args.seed,
                "cluster_unit": "source + conversation_id",
                "excluded_sources": sorted(excluded),
                "endpoints": {
                    name: {"column": column, "threshold": threshold}
                    for name, (column, threshold) in ENDPOINTS.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(split_integrity, indent=2))


if __name__ == "__main__":
    main()
