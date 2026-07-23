from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from wild_delusion_miner.assistant_responses import build_post_delusion_input


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_context_dose_response_inputs.py"
SPEC = importlib.util.spec_from_file_location("context_dose_response_inputs", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_arms = MODULE.build_arms

ANALYSIS_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "analyze_context_dose_response.py"
)
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "context_dose_response_analysis", ANALYSIS_SCRIPT
)
assert ANALYSIS_SPEC and ANALYSIS_SPEC.loader
ANALYSIS_MODULE = importlib.util.module_from_spec(ANALYSIS_SPEC)
ANALYSIS_SPEC.loader.exec_module(ANALYSIS_MODULE)


def test_build_arms_counts_user_and_assistant_messages() -> None:
    row = {
        "message_hash": "target",
        "target_message_index": 8,
        "messages": [
            {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
            for index in range(9)
        ],
    }

    arms = {arm["context_arm"]: arm for arm in build_arms(row)}

    assert list(arms) == [
        "prior_0",
        "prior_1",
        "prior_2",
        "prior_4",
        "prior_8",
        "prior_all",
    ]
    assert [message["content"] for message in arms["prior_2"]["messages"]] == [
        "6",
        "7",
        "8",
    ]
    assert arms["prior_2"]["visible_prior_message_count"] == 2


def test_post_delusion_input_can_omit_system_prompt() -> None:
    row = {
        "target_message_index": 1,
        "messages": [
            {"role": "assistant", "content": "Earlier reply"},
            {"role": "user", "content": "Target"},
        ],
    }

    messages = build_post_delusion_input(row, system_prompt=None)

    assert messages == row["messages"]


def test_cluster_bootstrap_returns_one_interval_per_arm() -> None:
    rows = []
    for conversation_id, positive_arm in [("a", "prior_0"), ("b", "prior_all")]:
        for arm in ANALYSIS_MODULE.ARM_ORDER:
            rows.append(
                {
                    "conversation_id": conversation_id,
                    "intervention_arm": arm,
                    "positive": int(arm == positive_arm),
                }
            )

    low, high = ANALYSIS_MODULE.bootstrap_rates(
        pd.DataFrame(rows), draws=100, seed=7
    )

    assert low.shape == (len(ANALYSIS_MODULE.ARM_ORDER),)
    assert high.shape == (len(ANALYSIS_MODULE.ARM_ORDER),)
    assert (low <= high).all()
