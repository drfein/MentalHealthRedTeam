from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


COMPARISONS = [
    ("validating", "neutral"),
    ("reality_testing", "neutral"),
    ("validating", "reality_testing"),
]

READOUT_COLUMN_ALIASES = {
    "intervention_arm": "arm",
    "misinformation_max_logit": "misinformation_logit",
    "falsity_concern_mean_logit": "falsity_concern",
    "reality_testing_mean_logit": "reality_testing",
}


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    return tuple(np.quantile(values[indices].mean(axis=1), [0.025, 0.975]))


def normalize_readout_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept both the original compact cache and the canonical explicit schema."""
    rename = {
        source: target
        for source, target in READOUT_COLUMN_ALIASES.items()
        if source in frame and target not in frame
    }
    normalized = frame.rename(columns=rename)
    required = {
        "original_row_idx",
        "arm",
        "layer",
        "misinformation_logit",
        "falsity_concern",
        "reality_testing",
    }
    if missing := required - set(normalized):
        raise ValueError(f"J-space readouts are missing columns: {sorted(missing)}")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Link J-space shifts to judged response behavior.")
    parser.add_argument("--judged-responses", type=Path, required=True)
    parser.add_argument("--jspace-readouts", type=Path, required=True)
    parser.add_argument("--prompt-variants", type=Path, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--score-column", default="annotation_score")
    parser.add_argument("--positive-threshold", type=int, default=7)
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    with args.judged_responses.open(encoding="utf-8") as handle:
        behavior = pd.DataFrame(json.loads(line) for line in handle if line.strip())
    behavior = behavior.dropna(subset=[args.score_column]).copy()
    if args.prompt_variants:
        prompts = pd.read_csv(args.prompt_variants)
        invalid_ids = set(
            (
                prompts.loc[prompts["prompt_tokens"] > args.max_prompt_tokens, "row_idx"]
                // 3
            ).astype(int)
        )
        behavior = behavior[~behavior["original_row_idx"].isin(invalid_ids)].copy()
    behavior["positive"] = (behavior[args.score_column] >= args.positive_threshold).astype(int)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    effects = []
    for metric in [args.score_column, "positive"]:
        wide = behavior.pivot(index="original_row_idx", columns="intervention_arm", values=metric)
        for left, right in COMPARISONS:
            delta = (wide[left] - wide[right]).dropna().to_numpy(dtype=float)
            low, high = bootstrap_ci(delta, rng, args.bootstrap_draws)
            try:
                p_value = float(wilcoxon(delta).pvalue)
            except ValueError:
                p_value = 1.0
            effects.append(
                {
                    "metric": metric,
                    "contrast": f"{left}_minus_{right}",
                    "n_pairs": len(delta),
                    "mean_difference": delta.mean(),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "wilcoxon_p": p_value,
                }
            )
    pd.DataFrame(effects).to_csv(args.out_dir / "behavior_paired_effects.csv", index=False)

    jspace = normalize_readout_columns(pd.read_parquet(args.jspace_readouts))
    jspace = jspace[jspace["layer"] == args.layer]
    score_wide = behavior.pivot(
        index="original_row_idx", columns="intervention_arm", values=args.score_column
    )
    links = pd.DataFrame(index=score_wide.index)
    links["behavior_delta"] = score_wide["validating"] - score_wide["reality_testing"]
    correlations = []
    for metric in ["misinformation_logit", "falsity_concern", "reality_testing"]:
        jspace_wide = jspace.pivot(index="original_row_idx", columns="arm", values=metric)
        column = f"{metric}_delta"
        links[column] = jspace_wide["validating"] - jspace_wide["reality_testing"]
        complete = links[["behavior_delta", column]].dropna()
        rho, p_value = spearmanr(complete["behavior_delta"], complete[column])
        correlations.append(
            {"metric": metric, "n": len(complete), "spearman_rho": rho, "p_value": p_value}
        )
    links.to_csv(args.out_dir / "jspace_behavior_deltas.csv")
    pd.DataFrame(correlations).to_csv(args.out_dir / "jspace_behavior_correlations.csv", index=False)

    summary = behavior.groupby("intervention_arm").agg(
        n=(args.score_column, "size"),
        mean_score=(args.score_column, "mean"),
        positive_rate=("positive", "mean"),
    )
    summary.to_csv(args.out_dir / "behavior_arm_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
