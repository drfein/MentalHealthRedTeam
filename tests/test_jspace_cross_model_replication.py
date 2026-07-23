from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analyze_jspace_cross_model_replication.py"
)
SPEC = importlib.util.spec_from_file_location("jspace_cross_model", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
bootstrap_metrics = MODULE.bootstrap_metrics


def test_bootstrap_metrics_preserve_frozen_indicator_direction() -> None:
    frame = pd.DataFrame(
        {
            11: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            22: [0.0, 0.1, 0.2, 0.8, 0.9, 1.0],
            "framing_score": [0, 1, 2, 4, 5, 5],
        }
    )
    labels = np.array([False, False, False, True, True, True])

    result = bootstrap_metrics(
        frame,
        labels,
        mid_layer=11,
        late_layer=22,
        draws=200,
        seed=7,
    )

    assert result["late_auc"] == 1.0
    assert result["late_graded_spearman"] > 0.9
    assert result["late_auc_ci_low"] == 1.0
