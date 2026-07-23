from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analyze_jspace_open_mindedness_steering.py"
)
SPEC = importlib.util.spec_from_file_location("open_mindedness_steering", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_uses_paired_axis_and_random_contrasts() -> None:
    rows = []
    values = {
        "primary_down": 1,
        "baseline": 2,
        "primary_up": 4,
        "random_down": 2,
        "random_up": 3,
    }
    for item in range(4):
        for condition, sensitivity in values.items():
            rows.append(
                {
                    "original_row_idx": item,
                    "condition": condition,
                    "reality_endorsement_score": 5 - sensitivity,
                    "epistemic_sensitivity_score": sensitivity,
                    "epistemic_openness_score": 5 - sensitivity,
                    "respectful_engagement_score": 4,
                    "blanket_refusal": False,
                    "incoherent": False,
                }
            )

    result = MODULE.summarize(pd.DataFrame(rows), draws=100, seed=7)

    assert result["n_messages"] == 4
    assert (
        result["contrasts"][
            "epistemic_sensitivity_score:axis_up_minus_down"
        ]["estimate"]
        == 3
    )
    assert (
        result["contrasts"][
            "epistemic_sensitivity_score:difference_in_differences"
        ]["estimate"]
        == 2
    )


def test_merge_package_judgments_uses_item_and_condition(tmp_path: Path) -> None:
    custom = pd.DataFrame(
        [
            {"original_row_idx": 7, "condition": "baseline"},
            {"original_row_idx": 7, "condition": "primary_up"},
        ]
    )
    path = tmp_path / "package.jsonl"
    rows = [
        {"original_row_idx": 7, "condition": "primary_up", "annotation_score": 4},
        {"original_row_idx": 7, "condition": "baseline", "annotation_score": 1},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    merged = MODULE.merge_package_judgments(custom, path)

    assert merged["package_endorsement_score"].tolist() == [1, 4]
