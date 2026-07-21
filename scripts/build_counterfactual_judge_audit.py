from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a blinded audit sample for framing-aware judge calibration."
    )
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument(
        "--secondary-judgments",
        type=Path,
        default=None,
        help="Optional second rubric whose positives are also included as a census.",
    )
    parser.add_argument("--secondary-score-column", default="annotation_score")
    parser.add_argument("--secondary-positive-threshold", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-at-or-above", type=int, default=3)
    parser.add_argument("--lower-controls", type=int, default=40)
    parser.add_argument(
        "--arms",
        nargs="+",
        help="Optional intervention arms to include before sampling.",
    )
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="Exclude one or more source labels before sampling.",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include every row after arm filtering instead of score-based sampling.",
    )
    parser.add_argument(
        "--selected-original-indices",
        type=Path,
        default=None,
        help="Optional newline-delimited allowlist of original row indices.",
    )
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    judgments = pd.DataFrame(read_jsonl(args.judgments)).rename(
        columns={"intervention_arm": "arm"}
    )
    prompts = pd.DataFrame(read_jsonl(args.prompts)).rename(
        columns={"intervention_arm": "arm"}
    )
    keys = ["original_row_idx", "arm"]
    frame = judgments.merge(
        prompts[keys + ["framed_text"]], on=keys, validate="one_to_one"
    )
    if args.secondary_judgments is not None:
        secondary = pd.DataFrame(read_jsonl(args.secondary_judgments)).rename(
            columns={"intervention_arm": "arm"}
        )
        if args.secondary_score_column not in secondary:
            raise ValueError(
                f"Secondary judgments lack {args.secondary_score_column!r}"
            )
        secondary = secondary[keys + [args.secondary_score_column]].rename(
            columns={args.secondary_score_column: "secondary_judge_score"}
        )
        frame = frame.merge(secondary, on=keys, validate="one_to_one")
    if args.exclude_source:
        frame = frame[~frame["source"].isin(set(args.exclude_source))].copy()
    if args.selected_original_indices is not None:
        selected = {
            int(value)
            for value in args.selected_original_indices.read_text(
                encoding="utf-8"
            ).splitlines()
            if value.strip()
        }
        frame = frame[frame["original_row_idx"].isin(selected)].copy()
    if args.arms:
        unknown_arms = sorted(set(args.arms) - set(frame["arm"]))
        if unknown_arms:
            parser.error(f"Unknown arms: {', '.join(unknown_arms)}")
        frame = frame[frame["arm"].isin(args.arms)].copy()

    rng = np.random.default_rng(args.seed)
    if args.include_all:
        sample = frame.copy()
        sample["sample_reason"] = "complete_filtered_set"
        sample["sampling_stratum"] = "complete_filtered_set"
        sample["population_stratum_n"] = len(sample)
        sample["sample_stratum_n"] = len(sample)
        sample["analysis_weight"] = 1.0
        high_count = int(
            (sample["reality_endorsement_score"] >= args.include_at_or_above).sum()
        )
        lower_count = len(sample) - high_count
    else:
        primary_positive = frame["reality_endorsement_score"].ge(
            args.include_at_or_above
        )
        secondary_positive = pd.Series(False, index=frame.index)
        if "secondary_judge_score" in frame:
            secondary_positive = frame["secondary_judge_score"].ge(
                args.secondary_positive_threshold
            )
        high = frame[primary_positive | secondary_positive].copy()
        lower = frame[~(primary_positive | secondary_positive)].copy()
        lower["sampling_stratum"] = (
            lower["arm"].astype(str)
            + "__score_"
            + lower["reality_endorsement_score"].astype(str)
        )
        stratum_count = lower["sampling_stratum"].nunique()
        if args.lower_controls < stratum_count:
            parser.error(
                f"--lower-controls must be at least {stratum_count} to cover every stratum"
            )
        target_n = min(args.lower_controls, len(lower))
        pools: dict[str, list[int]] = {}
        for stratum, group in lower.groupby("sampling_stratum"):
            indices = group.index.to_numpy(copy=True)
            rng.shuffle(indices)
            pools[str(stratum)] = indices.tolist()
        selected_indices: list[int] = []
        strata = sorted(pools)
        while len(selected_indices) < target_n:
            rng.shuffle(strata)
            selected_this_round = 0
            for stratum in strata:
                if pools[stratum] and len(selected_indices) < target_n:
                    selected_indices.append(pools[stratum].pop())
                    selected_this_round += 1
            if selected_this_round == 0:
                break
        selected_lower = lower.loc[selected_indices].copy()
        high["sample_reason"] = "automatic_positive_union_census"
        high["sampling_stratum"] = "automatic_positive_census"
        selected_lower["sample_reason"] = "stratified_lower_control"
        sample = pd.concat([high, selected_lower], ignore_index=True)
        population_counts = (
            pd.concat(
                [
                    pd.Series(
                        {
                            "automatic_positive_census": len(high),
                        }
                    ),
                    lower.groupby("sampling_stratum").size(),
                ]
            )
            .rename("population_stratum_n")
            .to_dict()
        )
        sample_counts = sample.groupby("sampling_stratum").size().to_dict()
        sample["population_stratum_n"] = sample["sampling_stratum"].map(
            population_counts
        )
        sample["sample_stratum_n"] = sample["sampling_stratum"].map(sample_counts)
        sample["analysis_weight"] = (
            sample["population_stratum_n"] / sample["sample_stratum_n"]
        )
        high_count = len(high)
        lower_count = len(selected_lower)
    order = rng.permutation(len(sample))
    sample = sample.iloc[order].reset_index(drop=True)
    sample.insert(0, "review_id", [f"CF-{index + 1:04d}" for index in range(len(sample))])

    review = sample[
        ["review_id", "arm", "framed_text", "response"]
    ].rename(
        columns={
            "arm": "declared_frame",
            "framed_text": "framed_user_message",
            "response": "assistant_response",
        }
    )
    review["human_reality_endorsement_score_0_to_5"] = ""
    review["human_frame_adherence_score_0_to_2"] = ""
    review["human_notes"] = ""
    key_columns = [
        "review_id",
        "original_row_idx",
        "arm",
        "sample_reason",
        "sampling_stratum",
        "population_stratum_n",
        "sample_stratum_n",
        "analysis_weight",
        "reality_endorsement_score",
        "frame_adherence_score",
        "behavior_label",
        "judge_confidence",
        "judge_rationale",
    ]
    if "secondary_judge_score" in sample:
        key_columns.append("secondary_judge_score")
    key = sample[key_columns]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review.to_csv(args.output_dir / "blinded_review.csv", index=False)
    key.to_csv(args.output_dir / "judge_key.csv", index=False)
    manifest = {
        "judgments": str(args.judgments),
        "prompts": str(args.prompts),
        "secondary_judgments": (
            str(args.secondary_judgments)
            if args.secondary_judgments is not None
            else None
        ),
        "secondary_score_column": args.secondary_score_column,
        "secondary_positive_threshold": args.secondary_positive_threshold,
        "seed": args.seed,
        "arms": args.arms,
        "excluded_sources": args.exclude_source,
        "include_all": args.include_all,
        "selected_original_indices": (
            str(args.selected_original_indices)
            if args.selected_original_indices is not None
            else None
        ),
        "included_score_threshold": args.include_at_or_above,
        "automatic_positive_union_n": high_count,
        "lower_rows_included": lower_count,
        "total_review_rows": len(review),
        "blinding": "judge scores, rationales, and original row IDs omitted from blinded_review.csv",
        "analysis_weight": "population_stratum_n / sample_stratum_n; automatic positives are a census",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
