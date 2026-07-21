from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYZER = load_script("analyze_full_behavior_human_audit.py")
BUILDER = load_script("build_counterfactual_judge_audit.py")
ADJUDICATION_BUILDER = load_script("build_human_audit_adjudication.py")
RELEASE_ANALYZER = load_script("analyze_wilddelusion_release_human_audit.py")


def test_finite_population_interval_holds_census_fixed() -> None:
    frame = pd.DataFrame(
        {
            "arm": ["direct_assertion"] * 2 + ["reported_belief"] * 2,
            "sampling_stratum": ["direct_census"] * 2 + ["reported_census"] * 2,
            "population_stratum_n": [2] * 4,
            "sample_stratum_n": [2] * 4,
            "positive": [True, False, False, False],
        }
    )

    intervals, difference = ANALYZER.stratified_finite_population_intervals(
        frame, "positive", draws=200, seed=7
    )

    assert intervals["direct_assertion"] == (0.5, 0.5)
    assert intervals["reported_belief"] == (0.0, 0.0)
    assert difference == (0.5, 0.5)


def test_sparse_zero_strata_get_non_degenerate_uncertainty() -> None:
    frame = pd.DataFrame(
        {
            "arm": ["direct_assertion"] * 3 + ["reported_belief"] * 3,
            "sampling_stratum": ["direct_sample"] * 3 + ["reported_sample"] * 3,
            "population_stratum_n": [100] * 6,
            "sample_stratum_n": [3] * 6,
            "positive": [False] * 6,
        }
    )

    intervals, _ = ANALYZER.stratified_finite_population_intervals(
        frame, "positive", draws=2_000, seed=11
    )

    assert intervals["direct_assertion"][0] == 0.0
    assert intervals["direct_assertion"][1] > 0.0
    assert intervals["reported_belief"][1] > 0.0


def test_control_allocation_balances_arms_and_oversamples_score3() -> None:
    rows = []
    for arm in ("direct_assertion", "reported_belief"):
        rows.extend({"arm": arm, "risk_tier": "score_3"} for _ in range(10))
        rows.extend({"arm": arm, "risk_tier": "score_0_2"} for _ in range(40))
    frame = pd.DataFrame(rows)

    selected = BUILDER.select_lower_control_indices(
        frame,
        np.random.default_rng(5),
        controls_per_arm=20,
        score3_controls_per_arm=8,
    )
    sample = frame.loc[selected]

    assert sample.groupby("arm").size().to_dict() == {
        "direct_assertion": 20,
        "reported_belief": 20,
    }
    assert sample[sample["risk_tier"].eq("score_3")].groupby("arm").size().to_dict() == {
        "direct_assertion": 8,
        "reported_belief": 8,
    }


def test_response_adjudication_produces_primary_endpoint(
    tmp_path: Path, monkeypatch
) -> None:
    ids = ["D1", "D2", "R1", "R2"]
    key = pd.DataFrame(
        {
            "review_id": ids,
            "arm": ["direct_assertion"] * 2 + ["reported_belief"] * 2,
            "sample_reason": ["automatic_positive_union_census"] * 4,
            "sampling_stratum": ["direct_census"] * 2 + ["reported_census"] * 2,
            "population_stratum_n": [2] * 4,
            "sample_stratum_n": [2] * 4,
            "analysis_weight": [1.0] * 4,
            "reality_endorsement_score": [4, 0, 0, 0],
            "secondary_judge_score": [7, 0, 0, 0],
        }
    )
    key_path = tmp_path / "key.csv"
    key.to_csv(key_path, index=False)

    def review(path: Path, scores: list[int]) -> None:
        pd.DataFrame(
            {
                "review_id": ids,
                "human_reality_endorsement_score_0_to_5": scores,
            }
        ).to_csv(path, index=False)

    rater_a = tmp_path / "a.csv"
    rater_b = tmp_path / "b.csv"
    adjudicated = tmp_path / "adjudicated.csv"
    review(rater_a, [4, 0, 0, 0])
    review(rater_b, [4, 0, 4, 0])
    pd.DataFrame(
        {
            "review_id": ["R1"],
            "human_reality_endorsement_score_0_to_5": [0],
        }
    ).to_csv(adjudicated, index=False)

    output = tmp_path / "analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_full_behavior_human_audit.py",
            "--reviews",
            str(rater_a),
            str(rater_b),
            "--rater-names",
            "a",
            "b",
            "--adjudication",
            str(adjudicated),
            "--judge-key",
            str(key_path),
            "--output-dir",
            str(output),
            "--posterior-draws",
            "100",
        ],
    )
    ANALYZER.main()

    rates = pd.read_csv(output / "human_weighted_rates.csv")
    adjudicated_primary = rates[
        rates["rater"].eq("adjudicated")
        & rates["endpoint"].eq("primary_score_4_to_5")
    ].set_index("arm")
    assert adjudicated_primary.loc["direct_assertion", "weighted_positive_rate"] == 0.5
    assert adjudicated_primary.loc["reported_belief", "weighted_positive_rate"] == 0.0
    assert (
        adjudicated_primary.loc[
            "direct_assertion_minus_reported_belief", "weighted_positive_rate"
        ]
        == 0.5
    )


def test_adjudication_builder_reveals_only_disagreement_rows(
    tmp_path: Path, monkeypatch
) -> None:
    blinded = pd.DataFrame(
        {
            "review_id": ["A", "B"],
            "declared_frame": ["question", "question"],
            "framed_user_message": ["u1", "u2"],
            "assistant_response": ["a1", "a2"],
            "human_reality_endorsement_score_0_to_5": ["", ""],
            "human_frame_adherence_score_0_to_2": ["", ""],
            "human_notes": ["", ""],
        }
    )
    blinded_path = tmp_path / "blinded.csv"
    blinded.to_csv(blinded_path, index=False)
    review_a = blinded.copy()
    review_b = blinded.copy()
    review_a["human_reality_endorsement_score_0_to_5"] = [0, 4]
    review_b["human_reality_endorsement_score_0_to_5"] = [0, 3]
    review_a["human_frame_adherence_score_0_to_2"] = [2, 2]
    review_b["human_frame_adherence_score_0_to_2"] = [2, 2]
    a_path = tmp_path / "a.csv"
    b_path = tmp_path / "b.csv"
    review_a.to_csv(a_path, index=False)
    review_b.to_csv(b_path, index=False)

    output = tmp_path / "adjudication"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_human_audit_adjudication.py",
            "--kind",
            "response",
            "--reviews",
            str(a_path),
            str(b_path),
            "--blinded-review",
            str(blinded_path),
            "--output-dir",
            str(output),
        ],
    )
    ADJUDICATION_BUILDER.main()

    sheet = pd.read_csv(output / "blinded_adjudication.csv", keep_default_na=False)
    assert sheet["review_id"].tolist() == ["B"]
    assert sheet["human_reality_endorsement_score_0_to_5"].eq("").all()
    assert sheet["human_frame_adherence_score_0_to_2"].eq("").all()


def test_release_adjudication_produces_single_precision_estimate(
    tmp_path: Path, monkeypatch
) -> None:
    ids = ["WD-001", "WD-002"]
    key = tmp_path / "audit_key.csv"
    pd.DataFrame({"review_id": ids, "release_row_index": [0, 1]}).to_csv(
        key, index=False
    )

    def release_review(
        path: Path, decisions: list[str], exclusions: list[str]
    ) -> None:
        pd.DataFrame(
            {
                "review_id": ids,
                "decision": decisions,
                "exclusion_reason": exclusions,
                "confidence_1_to_5": [5, 5],
            }
        ).to_csv(path, index=False)

    rater_a = tmp_path / "release_a.csv"
    rater_b = tmp_path / "release_b.csv"
    adjudicated = tmp_path / "release_adjudicated.csv"
    release_review(rater_a, ["positive", "positive"], ["none", "none"])
    release_review(
        rater_b,
        ["positive", "negative"],
        ["none", "ordinary_plausible"],
    )
    pd.DataFrame(
        {
            "review_id": ["WD-002"],
            "decision": ["negative"],
            "exclusion_reason": ["ordinary_plausible"],
            "confidence_1_to_5": [5],
        }
    ).to_csv(adjudicated, index=False)

    output = tmp_path / "release_analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_wilddelusion_release_human_audit.py",
            "--reviews",
            str(rater_a),
            str(rater_b),
            "--rater-names",
            "a",
            "b",
            "--adjudication",
            str(adjudicated),
            "--audit-key",
            str(key),
            "--output-dir",
            str(output),
        ],
    )
    RELEASE_ANALYZER.main()

    metrics = pd.read_csv(output / "precision_metrics.csv")
    adjudicated_metric = metrics[
        metrics["rater"].eq("adjudicated")
        & metrics["metric"].eq("strict_release_precision")
    ].iloc[0]
    assert adjudicated_metric["numerator"] == 1
    assert adjudicated_metric["denominator"] == 2
    assert adjudicated_metric["estimate"] == 0.5
