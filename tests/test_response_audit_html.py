from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/make_counterfactual_audit_html.py"
SPEC = importlib.util.spec_from_file_location("counterfactual_audit_html", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_safe_rater_id_is_stable_and_nonempty() -> None:
    assert MODULE.safe_rater_id("reviewer A/1") == "reviewer_A_1"
    with pytest.raises(ValueError, match="at least one letter or digit"):
        MODULE.safe_rater_id("")


def test_template_has_required_audit_contract() -> None:
    assert "__ROWS_JSON__" in MODULE.TEMPLATE
    assert "__SEED__" in MODULE.TEMPLATE
    assert "__RATER_ID__" in MODULE.TEMPLATE
    assert "human_reality_endorsement_score_0_to_5" in MODULE.TEMPLATE
    assert "human_frame_adherence_score_0_to_2" in MODULE.TEMPLATE
    assert "analysis_weight" not in MODULE.TEMPLATE
    assert "reality_endorsement_score" not in MODULE.TEMPLATE.replace(
        "human_reality_endorsement_score_0_to_5", ""
    )
