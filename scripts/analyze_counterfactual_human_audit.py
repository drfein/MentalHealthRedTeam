from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score


SCORE_COLUMN = "human_reality_endorsement_score_0_to_5"


def cluster_bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    clusters: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float, int]:
    estimate = float(roc_auc_score(labels, scores))
    group_codes, groups = pd.factorize(clusters, sort=False)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(draws):
        group_weights = rng.multinomial(
            len(groups), np.full(len(groups), 1 / len(groups))
        )
        row_weights = group_weights[group_codes]
        if row_weights[labels].sum() and row_weights[~labels].sum():
            values.append(
                float(roc_auc_score(labels, scores, sample_weight=row_weights))
            )
    low, high = np.quantile(values, [0.025, 0.975])
    return estimate, float(low), float(high), len(values)


def load_source_maps(path: Path) -> tuple[dict[int, str], dict[int, str]]:
    conversations: dict[int, str] = {}
    sources: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["original_row_idx"])
            key = json.dumps([row["source"], str(row["conversation_id"])])
            source = str(row["source"])
            if index in conversations and conversations[index] != key:
                raise ValueError(f"Inconsistent conversation mapping for row {index}")
            if index in sources and sources[index] != source:
                raise ValueError(f"Inconsistent source mapping for row {index}")
            conversations[index] = key
            sources[index] = source
    return conversations, sources


def load_review(path: Path, rater: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"review_id", SCORE_COLUMN}
    if missing := required - set(frame):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    scores = pd.to_numeric(frame[SCORE_COLUMN], errors="coerce")
    if scores.isna().any():
        missing_ids = frame.loc[scores.isna(), "review_id"].head(8).tolist()
        raise ValueError(f"{path} has incomplete/non-numeric scores, including {missing_ids}")
    if not scores.between(0, 5).all() or not np.equal(scores, scores.astype(int)).all():
        raise ValueError(f"{path} scores must be integers from 0 through 5")
    if frame["review_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate review IDs")
    return pd.DataFrame(
        {
            "review_id": frame["review_id"].astype(str),
            f"score__{rater}": scores.astype(int),
        }
    )


def agreement_row(
    first_name: str,
    second_name: str,
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, Any]:
    first_binary = first >= 4
    second_binary = second >= 4
    return {
        "first_rater": first_name,
        "second_rater": second_name,
        "n": len(first),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(first, second, weights="quadratic")
        ),
        "primary_binary_raw_agreement": float((first_binary == second_binary).mean()),
        "first_primary_positive_n": int(first_binary.sum()),
        "second_primary_positive_n": int(second_binary.sum()),
        "primary_positive_overlap_n": int((first_binary & second_binary).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze blinded human labels for the direct-assertion J-space endpoint."
    )
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--rater-names", nargs="+", default=None)
    parser.add_argument("--judge-key", type=Path, required=True)
    parser.add_argument("--indicator-scores", type=Path, required=True)
    parser.add_argument(
        "--source-judgments",
        type=Path,
        default=Path(
            "results/jspace_context_interventions/qwen2_5_7b_it_probe_free/"
            "openai_package_bot_endorses_judged.jsonl"
        ),
    )
    parser.add_argument(
        "--conversation-disjoint-ids",
        type=Path,
        default=Path(
            "results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals/"
            "conversation_disjoint_correction/clean_holdout_ids.txt"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    rater_names = args.rater_names or [f"rater_{index + 1}" for index in range(len(args.reviews))]
    if len(rater_names) != len(args.reviews):
        parser.error("--rater-names must contain one name per --reviews file")
    if len(set(rater_names)) != len(rater_names):
        parser.error("Rater names must be unique")

    key = pd.read_csv(args.judge_key)
    required_key = {"review_id", "original_row_idx", "reality_endorsement_score"}
    if missing := required_key - set(key):
        raise ValueError(f"Judge key is missing columns: {sorted(missing)}")
    if key["review_id"].duplicated().any():
        raise ValueError("Judge key contains duplicate review IDs")
    frame = key[["review_id", "original_row_idx", "reality_endorsement_score"]].copy()
    frame["review_id"] = frame["review_id"].astype(str)
    for path, rater in zip(args.reviews, rater_names, strict=True):
        review = load_review(path, rater)
        if set(review["review_id"]) != set(frame["review_id"]):
            raise ValueError(f"{path} review IDs do not exactly match the judge key")
        frame = frame.merge(review, on="review_id", validate="one_to_one")

    indicator = pd.read_csv(args.indicator_scores)[
        ["original_row_idx", "inverse_misinformation_max"]
    ]
    if indicator["original_row_idx"].duplicated().any():
        raise ValueError("Indicator scores contain duplicate original row IDs")
    frame = frame.merge(indicator, on="original_row_idx", validate="one_to_one")
    if len(frame) != 144:
        raise ValueError(f"Expected the complete 144-row public audit, found {len(frame)} rows")
    conversation_map, source_map = load_source_maps(args.source_judgments)
    frame["conversation_key"] = frame["original_row_idx"].map(conversation_map)
    frame["source"] = frame["original_row_idx"].map(source_map)
    if frame["conversation_key"].isna().any():
        raise ValueError("Audit rows are missing source-conversation mappings")
    clean_ids = {
        int(value)
        for value in args.conversation_disjoint_ids.read_text(encoding="utf-8").splitlines()
        if value.strip()
    }
    public_frame = frame[frame["source"].ne("lmsys_chat_1m")].copy()
    populations = {
        "public_evaluation": public_frame,
        "conversation_disjoint": public_frame[
            public_frame["original_row_idx"].isin(clean_ids)
        ].copy(),
    }
    if len(public_frame) != 144 or len(populations["conversation_disjoint"]) != 67:
        raise ValueError("Expected 144 public evaluation rows and 67 disjoint rows")

    performance_rows: list[dict[str, Any]] = []
    for population_offset, (population, subset) in enumerate(populations.items()):
        scores = subset["inverse_misinformation_max"].to_numpy(float)
        clusters = subset["conversation_key"].to_numpy(str)
        for rater_offset, rater in enumerate(rater_names):
            ordinal = subset[f"score__{rater}"].to_numpy(int)
            for threshold_offset, threshold in enumerate((4, 3)):
                labels = ordinal >= threshold
                if np.unique(labels).size != 2:
                    raise ValueError(
                        f"{population}/{rater} threshold {threshold} has one outcome class"
                    )
                estimate, low, high, valid_draws = cluster_bootstrap_auc(
                    labels,
                    scores,
                    clusters,
                    draws=args.bootstrap_draws,
                    seed=(
                        args.seed
                        + 100 * population_offset
                        + 10 * rater_offset
                        + threshold_offset
                    ),
                )
                performance_rows.append(
                    {
                        "population": population,
                        "rater": rater,
                        "threshold": threshold,
                        "endpoint": (
                            "primary_4_to_5" if threshold == 4 else "sensitivity_3_to_5"
                        ),
                        "n_target_turns": len(subset),
                        "n_source_conversations": subset["conversation_key"].nunique(),
                        "positive_n": int(labels.sum()),
                        "auc": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_draws_valid": valid_draws,
                        "interval": "95% source-conversation cluster bootstrap",
                        "indicator": "frozen inverse layer-26 misinformation maximum loading",
                    }
                )

    agreement_rows = []
    for first, second in combinations(rater_names, 2):
        agreement_rows.append(
            agreement_row(
                first,
                second,
                frame[f"score__{first}"].to_numpy(int),
                frame[f"score__{second}"].to_numpy(int),
            )
        )
    for rater in rater_names:
        agreement_rows.append(
            agreement_row(
                rater,
                "gpt_5_4_mini_framing_rubric",
                frame[f"score__{rater}"].to_numpy(int),
                frame["reality_endorsement_score"].to_numpy(int),
            )
        )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    performance = pd.DataFrame(performance_rows)
    performance.to_csv(output / "human_endpoint_performance.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(output / "rater_agreement.csv", index=False)
    frame.to_csv(output / "audit_analysis_rows.csv", index=False)
    (output / "hparams.json").write_text(
        json.dumps(
            {
                "reviews": [str(path) for path in args.reviews],
                "rater_names": rater_names,
                "judge_key": str(args.judge_key),
                "indicator_scores": str(args.indicator_scores),
                "source_judgments": str(args.source_judgments),
                "conversation_disjoint_ids": str(args.conversation_disjoint_ids),
                "bootstrap_draws": args.bootstrap_draws,
                "seed": args.seed,
                "cluster_unit": "source + conversation_id",
                "excluded_sources": ["lmsys_chat_1m"],
                "primary_human_endpoint": "score >= 4",
                "sensitivity_human_endpoint": "score >= 3",
                "frozen_indicator": "inverse layer-26 misinformation max-over-position readout",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot = performance.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8.4, max(3.4, 0.55 * len(plot) + 1.8)))
    ax.errorbar(
        plot["auc"],
        y,
        xerr=np.vstack([plot["auc"] - plot["ci_low"], plot["ci_high"] - plot["auc"]]),
        fmt="o",
        capsize=4,
        color="#2A6F97",
    )
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_yticks(
        y,
        [
            f"{population}: {rater}, score >= {threshold} (n+={positive_n})"
            for population, rater, threshold, positive_n in zip(
                plot["population"],
                plot["rater"],
                plot["threshold"],
                plot["positive_n"],
                strict=True,
            )
        ],
    )
    ax.set_xlim(0.3, 1.01)
    ax.set_xlabel("AUROC of frozen inverse misinformation readout")
    ax.set_title("Human-adjudicated direct-assertion endpoint", loc="left", weight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "human_endpoint_performance.png", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
