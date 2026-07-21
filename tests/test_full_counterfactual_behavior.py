from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_full_counterfactual_behavior.py"
SPEC = importlib.util.spec_from_file_location("full_counterfactual_behavior", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_exact_cluster_sign_flip_preserves_conversations() -> None:
    frame = pd.DataFrame(
        {"conversation_key": ["a", "a", "b"], "difference": [1, 0, 1]}
    )
    assert MODULE.exact_cluster_sign_flip_p(frame) == 0.5


def test_exact_cluster_sign_flip_handles_zero_observed_sum() -> None:
    frame = pd.DataFrame(
        {"conversation_key": ["a", "b"], "difference": [1, -1]}
    )
    assert MODULE.exact_cluster_sign_flip_p(frame) == 1.0
