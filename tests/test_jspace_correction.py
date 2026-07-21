from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_jspace_conversation_disjoint_holdout.py"
SPEC = importlib.util.spec_from_file_location("jspace_correction", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
behavior_contrast_rows = MODULE.behavior_contrast_rows
holm_adjust = MODULE.holm_adjust


def test_holm_adjust_preserves_order_and_monotonicity() -> None:
    adjusted = holm_adjust([0.04, 0.01, 0.03])
    assert adjusted == [0.06, 0.03, 0.06]


def test_behavior_contrasts_are_matched_by_target_and_clustered() -> None:
    rows = []
    outcomes = {
        "direct_assertion": [1, 1, 0, 0],
        "question": [0, 0, 0, 0],
        "reported_belief": [0, 1, 0, 0],
    }
    for arm, values in outcomes.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "original_row_idx": index,
                    "conversation_key": f"conversation-{index}",
                    "intervention_arm": arm,
                    "positive": bool(value),
                }
            )

    contrasts = behavior_contrast_rows(
        pd.DataFrame(rows),
        "test",
        draws=200,
        seed=7,
    )
    by_arm = {row["comparison_arm"]: row for row in contrasts}
    assert by_arm["question"]["risk_difference"] == 0.5
    assert by_arm["question"]["direct_positive_comparison_negative"] == 2
    assert by_arm["reported_belief"]["risk_difference"] == 0.25
    assert all(row["bootstrap_draws_valid"] == 200 for row in contrasts)
