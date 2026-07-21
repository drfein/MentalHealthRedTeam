from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


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


def test_unique_string_requires_one_run_setting() -> None:
    frame = pd.DataFrame({"model_id": ["model-a", "model-a"]})
    assert MODULE.unique_string(frame, "model_id") == "model-a"

    with pytest.raises(ValueError, match="Expected one model_id"):
        MODULE.unique_string(pd.DataFrame({"model_id": ["model-a", "model-b"]}), "model_id")


def test_source_stratified_summary_preserves_source_arm_counts() -> None:
    frame = pd.DataFrame(
        {
            "source": ["a", "a", "a", "a", "b", "b"],
            "intervention_arm": [
                "direct_assertion",
                "direct_assertion",
                "question",
                "question",
                "direct_assertion",
                "question",
            ],
            "conversation_key": ["a1", "a2", "a1", "a2", "b1", "b1"],
            "positive": [1, 0, 0, 0, 1, 0],
        }
    )

    summary = MODULE.source_stratified_summary(frame)
    source_a_direct = summary[
        summary["source"].eq("a") & summary["arm"].eq("direct_assertion")
    ].iloc[0]
    assert source_a_direct["n_target_turns"] == 2
    assert source_a_direct["n_source_conversations"] == 2
    assert source_a_direct["positive_n"] == 1
    assert source_a_direct["positive_rate"] == 0.5
