from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wild_delusion_miner.cluster_inference import (
    holm_adjust,
    paired_cluster_inference,
)
from wild_delusion_miner.plot_style import MODEL_LABELS, MODEL_ORDER


ENDPOINT_ARMS = ("prior_0", "prior_all")


cluster_inference = paired_cluster_inference


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
