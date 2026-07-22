from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COMPARISONS = [
    ("gpt-4o-mini-2024-07-18", "gpt-4o-2024-05-13", "GPT-4o mini - GPT-4o"),
    ("o3-mini-2025-01-31", "o1-2024-12-17", "o3 mini - o1"),
    (
        "gpt-4.1-mini-2025-04-14",
        "gpt-4-turbo-2024-04-09",
        "GPT-4.1 mini - GPT-4 Turbo",
    ),
    ("gpt-5-mini-2025-08-07", "gpt-5.2-2025-12-11", "GPT-5 mini - GPT-5.2"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cluster_bootstrap(
    frame: pd.DataFrame, *, draws: int, seed: int
) -> tuple[float, float]:
    grouped = frame.groupby("conversation_id")["difference"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(grouped), size=(draws, len(grouped)))
    estimates = (
        grouped["sum"].to_numpy()[sampled].sum(axis=1)
        / grouped["count"].to_numpy()[sampled].sum(axis=1)
    )
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test paired descriptive mini-versus-larger model endorsement differences."
    )
    parser.add_argument(
        "--generations",
        type=Path,
        default=Path(
            "results/response_analysis/10model_low_reasoning/latest_success_responses.jsonl"
        ),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "results/generated_response_annotations/gpt-5.4-mini/combined_8_flags/"
            "response_annotation_matrix.csv"
        ),
    )
    parser.add_argument(
        "--release", type=Path, default=Path("data/releases/WildDelusionVerified/train.parquet")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/mini_model_hypothesis")
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    generations = pd.DataFrame(read_jsonl(args.generations))
    annotations = pd.read_csv(args.annotations)
    release = (
        pd.read_parquet(args.release)[["message_hash", "conversation_id"]]
        .drop_duplicates("message_hash")
        .rename(columns={"conversation_id": "release_conversation_id"})
    )
    data = generations.merge(
        annotations[["generation_id", "bot_endorses_delusion_positive"]],
        on="generation_id",
        validate="one_to_one",
    ).merge(release, on="message_hash", how="inner", validate="many_to_one")
    data["endorse"] = data["bot_endorses_delusion_positive"].astype(int)
    data["conversation_id"] = data["release_conversation_id"]

    rows = []
    paired_frames = []
    for index, (mini, comparator, label) in enumerate(COMPARISONS):
        pivot = data[data["model"].isin([mini, comparator])].pivot(
            index=["message_hash", "conversation_id"], columns="model", values="endorse"
        )
        pivot = pivot.dropna(subset=[mini, comparator]).reset_index()
        pivot["difference"] = pivot[mini] - pivot[comparator]
        low, high = cluster_bootstrap(
            pivot, draws=args.bootstrap_draws, seed=args.seed + index
        )
        rows.append(
            {
                "comparison": label,
                "mini_model": mini,
                "comparator_model": comparator,
                "pairs": len(pivot),
                "mini_rate": float(pivot[mini].mean()),
                "comparator_rate": float(pivot[comparator].mean()),
                "difference": float(pivot["difference"].mean()),
                "ci_low": low,
                "ci_high": high,
            }
        )
        pivot["comparison"] = label
        paired_frames.append(pivot[["conversation_id", "comparison", "difference"]])

    results = pd.DataFrame(rows)
    pooled = pd.concat(paired_frames, ignore_index=True)
    low, high = cluster_bootstrap(
        pooled, draws=args.bootstrap_draws, seed=args.seed + 100
    )
    summary = {
        "comparisons": rows,
        "pooled_paired_difference": float(pooled["difference"].mean()),
        "pooled_ci_low": low,
        "pooled_ci_high": high,
        "interpretation": (
            "Descriptive only: model family, release date, architecture, and training differ. A "
            "consistent small-model effect requires all or nearly all paired contrasts to have the "
            "same sign; these comparisons are not exchangeable size interventions."
        ),
        "uncertainty": (
            f"95% percentile intervals from {args.bootstrap_draws} conversation-cluster draws."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "paired_comparisons.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
