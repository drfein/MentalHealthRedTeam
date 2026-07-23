from __future__ import annotations

import numpy as np
import pandas as pd


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * float(p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def paired_cluster_inference(
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

    sampled = rng.integers(0, len(clusters), size=(bootstrap_draws, len(clusters)))
    bootstrap = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])

    observed_sum = abs(sums.sum())
    extreme = 0
    for start in range(0, permutation_draws, 10_000):
        draws = min(10_000, permutation_draws - start)
        signs = rng.choice((-1.0, 1.0), size=(draws, len(clusters)))
        extreme += int((np.abs(signs @ sums) >= observed_sum - 1e-12).sum())
    p_value = (extreme + 1) / (permutation_draws + 1)

    return {
        "difference": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "conversations": int(len(clusters)),
    }
