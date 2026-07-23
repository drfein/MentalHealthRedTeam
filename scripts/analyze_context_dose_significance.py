from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wild_delusion_miner.plot_style import MODEL_LABELS, MODEL_ORDER


ENDPOINT_ARMS = ("prior_0", "prior_all")


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * float(p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def cluster_inference(
    deltas: pd.DataFrame,
    *,
    bootstrap_draws: int,
    permutation_draws: int,
    seed: int,
) -> dict[str, float]:
    clusters = deltas.groupby("conversation_id")["delta"].agg(["sum", "count"])
    sums = clusters["sum"].to_numpy(dtype=float)
    counts = clusters["count"].to_numpy(dtype=float)
    observed = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)

    sampled = rng.integers(
        0,
        len(clusters),
        size=(bootstrap_draws, len(clusters)),
    )
    bootstrap = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])

    observed_sum = abs(sums.sum())
    extreme = 0
    chunk_size = 10_000
    for start in range(0, permutation_draws, chunk_size):
        draws = min(chunk_size, permutation_draws - start)
        signs = rng.choice((-1.0, 1.0), size=(draws, len(clusters)))
        permuted = np.abs(signs @ sums)
        extreme += int((permuted >= observed_sum - 1e-12).sum())
    p_value = (extreme + 1) / (permutation_draws + 1)

    return {
        "difference": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "conversations": int(len(clusters)),
    }


def paired_endpoints(judgments: pd.DataFrame) -> pd.DataFrame:
    endpoints = judgments[judgments["intervention_arm"].isin(ENDPOINT_ARMS)]
    index = ["model_id", "original_row_idx", "conversation_id"]
    paired = endpoints.pivot(
        index=index,
        columns="intervention_arm",
        values="positive",
    ).dropna(subset=list(ENDPOINT_ARMS))
    paired["delta"] = paired["prior_all"] - paired["prior_0"]
    return paired.reset_index()


def analyze(
    judgments: pd.DataFrame,
    *,
    bootstrap_draws: int,
    permutation_draws: int,
    seed: int,
) -> pd.DataFrame:
    paired = paired_endpoints(judgments)
    rows = []
    for model_index, model in enumerate(MODEL_ORDER):
        model_data = paired[paired["model_id"] == model]
        if model_data.empty:
            continue
        inference = cluster_inference(
            model_data[["conversation_id", "delta"]],
            bootstrap_draws=bootstrap_draws,
            permutation_draws=permutation_draws,
            seed=seed + model_index,
        )
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "targets": int(len(model_data)),
                "conversations": inference["conversations"],
                "target_only_rate": float(model_data["prior_0"].mean()),
                "full_context_rate": float(model_data["prior_all"].mean()),
                "difference_pp": 100 * inference["difference"],
                "ci_low_pp": 100 * inference["ci_low"],
                "ci_high_pp": 100 * inference["ci_high"],
                "p_value": inference["p_value"],
            }
        )
    results = pd.DataFrame(rows)
    results["p_holm"] = holm_adjust(results["p_value"].to_numpy())
    results["significant_holm_0_05"] = results["p_holm"] < 0.05
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test the paired full-context minus target-only endorsement difference, "
            "clustered by source conversation."
        )
    )
    parser.add_argument(
        "--judgments",
        type=Path,
        default=Path("results/context_dose_response/analysis/complete_judgments.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/context_dose_response/analysis/paired_endpoint_inference.csv"
        ),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--permutation-draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    judgments = pd.read_csv(args.judgments)
    required = {
        "model_id",
        "original_row_idx",
        "conversation_id",
        "intervention_arm",
        "positive",
    }
    missing = required - set(judgments)
    if missing:
        raise ValueError(f"Judgment table is missing columns: {sorted(missing)}")
    results = analyze(
        judgments,
        bootstrap_draws=args.bootstrap_draws,
        permutation_draws=args.permutation_draws,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    manifest = {
        "estimand": "Full-context minus target-only endorsement probability.",
        "pairing_unit": "Verified target message within model.",
        "cluster_unit": "Source conversation.",
        "confidence_interval": (
            f"Percentile cluster bootstrap with {args.bootstrap_draws} draws."
        ),
        "p_value": (
            f"Two-sided conversation-cluster sign-flip test with "
            f"{args.permutation_draws} draws."
        ),
        "multiple_testing": "Holm family-wise correction across tested models.",
        "seed": args.seed,
        "models": len(results),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
