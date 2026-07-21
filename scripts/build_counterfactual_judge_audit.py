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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-at-or-above", type=int, default=3)
    parser.add_argument("--lower-controls", type=int, default=40)
    parser.add_argument(
        "--arms",
        nargs="+",
        help="Optional intervention arms to include before sampling.",
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
        high_count = int(
            (sample["reality_endorsement_score"] >= args.include_at_or_above).sum()
        )
        lower_count = len(sample) - high_count
    else:
        high = frame[
            frame["reality_endorsement_score"] >= args.include_at_or_above
        ].copy()
        lower = frame[
            frame["reality_endorsement_score"] < args.include_at_or_above
        ].copy()
        lower["sampling_stratum"] = (
            lower["arm"].astype(str)
            + "__score_"
            + lower["reality_endorsement_score"].astype(str)
        )
        weights = 1 / lower.groupby("sampling_stratum")[
            "sampling_stratum"
        ].transform("size")
        selected_lower = lower.sample(
            n=min(args.lower_controls, len(lower)),
            weights=weights,
            random_state=args.seed,
            replace=False,
        )
        high["sample_reason"] = (
            f"judge_score_at_least_{args.include_at_or_above}"
        )
        selected_lower["sample_reason"] = "stratified_lower_control"
        sample = pd.concat([high, selected_lower], ignore_index=True)
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
        "reality_endorsement_score",
        "frame_adherence_score",
        "behavior_label",
        "judge_confidence",
        "judge_rationale",
    ]
    key = sample[key_columns]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review.to_csv(args.output_dir / "blinded_review.csv", index=False)
    key.to_csv(args.output_dir / "judge_key.csv", index=False)
    manifest = {
        "judgments": str(args.judgments),
        "prompts": str(args.prompts),
        "seed": args.seed,
        "arms": args.arms,
        "include_all": args.include_all,
        "selected_original_indices": (
            str(args.selected_original_indices)
            if args.selected_original_indices is not None
            else None
        ),
        "included_score_threshold": args.include_at_or_above,
        "all_rows_at_or_above_threshold": high_count,
        "lower_rows_included": lower_count,
        "total_review_rows": len(review),
        "blinding": "judge scores, rationales, and original row IDs omitted from blinded_review.csv",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
