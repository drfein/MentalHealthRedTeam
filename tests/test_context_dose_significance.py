from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "analyze_context_dose_significance.py"
)
SPEC = importlib.util.spec_from_file_location("context_dose_significance", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    raw = np.array([0.04, 0.001, 0.02])
    adjusted = MODULE.holm_adjust(raw)

    assert np.allclose(adjusted, [0.04, 0.003, 0.04])


def test_analysis_detects_consistent_clustered_increase() -> None:
    rows = []
    for conversation in range(12):
        for target in range(2):
            for arm, positive in [("prior_0", 0), ("prior_all", 1)]:
                rows.append(
                    {
                        "model_id": "gpt-4.1-mini-2025-04-14",
                        "original_row_idx": conversation * 2 + target,
                        "conversation_id": f"conversation-{conversation}",
                        "intervention_arm": arm,
                        "positive": positive,
                    }
                )

    result = MODULE.analyze(
        pd.DataFrame(rows),
        bootstrap_draws=200,
        permutation_draws=2_000,
        seed=4,
    ).iloc[0]

    assert result["difference_pp"] == 100
    assert result["ci_low_pp"] == 100
    assert result["p_value"] < 0.01
