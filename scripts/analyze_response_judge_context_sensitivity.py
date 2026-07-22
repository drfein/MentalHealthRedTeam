from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cluster_bootstrap_mean(
    frame: pd.DataFrame, column: str, *, draws: int, seed: int
) -> tuple[float, float]:
    grouped = frame.groupby("conversation_id")[column].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(grouped), size=(draws, len(grouped)))
    estimates = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare target-only-context and preceding-context response judgments."
    )
    parser.add_argument(
        "--target-reply-judgments",
        type=Path,
        default=Path(
            "results/observed_assistant_responses/package_judgments_target_context.jsonl"
        ),
    )
    parser.add_argument(
        "--preceding-context-judgments",
        type=Path,
        default=Path("results/observed_all_turn_trajectories/package_judgments.jsonl"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/response_judge_context_sensitivity")
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    short = pd.DataFrame(read_jsonl(args.target_reply_judgments))
    contextual = pd.DataFrame(read_jsonl(args.preceding_context_judgments))
    contextual = contextual[contextual["follows_retained_target"].astype(bool)].copy()
    short["assistant_message_index"] = (
        short["canonical_target_message_index"].astype(int) + 1
    )
    keys = ["conversation_id", "canonical_row_idx", "assistant_message_index"]
    paired = short.merge(
        contextual,
        on=keys,
        suffixes=("_target_reply", "_preceding_context"),
        validate="one_to_one",
    )
    if len(paired) != 166:
        raise ValueError(f"Expected 166 overlapping retained-target replies, found {len(paired)}.")

    paired["positive_target_reply"] = (
        paired["annotation_score_target_reply"].astype(int) >= 7
    ).astype(int)
    paired["positive_preceding_context"] = (
        paired["annotation_score_preceding_context"].astype(int) >= 7
    ).astype(int)
    paired["positive_difference"] = (
        paired["positive_preceding_context"] - paired["positive_target_reply"]
    )
    paired["score_difference"] = (
        paired["annotation_score_preceding_context"]
        - paired["annotation_score_target_reply"]
    )
    positive_ci = cluster_bootstrap_mean(
        paired,
        "positive_difference",
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    score_ci = cluster_bootstrap_mean(
        paired, "score_difference", draws=args.bootstrap_draws, seed=args.seed + 1
    )
    gained = int(
        ((paired["positive_target_reply"] == 0) & (paired["positive_preceding_context"] == 1)).sum()
    )
    lost = int(
        ((paired["positive_target_reply"] == 1) & (paired["positive_preceding_context"] == 0)).sum()
    )
    summary = {
        "paired_replies": len(paired),
        "conversations": int(paired["conversation_id"].nunique()),
        "target_plus_reply": {
            "positives": int(paired["positive_target_reply"].sum()),
            "rate": float(paired["positive_target_reply"].mean()),
            "mean_score": float(paired["annotation_score_target_reply"].mean()),
        },
        "preceding_context_plus_target_plus_reply": {
            "positives": int(paired["positive_preceding_context"].sum()),
            "rate": float(paired["positive_preceding_context"].mean()),
            "mean_score": float(paired["annotation_score_preceding_context"].mean()),
        },
        "paired_context_effect": {
            "rate_difference": float(paired["positive_difference"].mean()),
            "rate_ci_low": positive_ci[0],
            "rate_ci_high": positive_ci[1],
            "score_difference": float(paired["score_difference"].mean()),
            "score_ci_low": score_ci[0],
            "score_ci_high": score_ci[1],
            "negative_to_positive": gained,
            "positive_to_negative": lost,
            "exact_mcnemar_p": float(binomtest(gained, gained + lost, 0.5).pvalue),
        },
        "interpretation": (
            "The 58.0% rate uses all 295 recoverable target-adjacent replies and is not the "
            "context-free comparator for the 79.5% repeated-conversation rate. On the same "
            "166 replies, supplying preceding context increases strict endorsement by the "
            "paired rate difference reported here."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.out_dir / "paired_reply_judgments.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
