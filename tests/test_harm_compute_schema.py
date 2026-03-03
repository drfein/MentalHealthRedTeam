from __future__ import annotations

import unittest
from unittest.mock import patch

from hr.core.types import DifferentialResult, StepScores
from hr.experiments.harm_kl.compute_drift import compute_kl_drift


class HarmComputeSchemaTests(unittest.TestCase):
    def test_compute_kl_drift_preserves_output_schema_and_metrics(self) -> None:
        pref_data = {
            "case_id": "case_1",
            "name": "Case",
            "theme": "Theme",
            "harm_type": "Type",
            "condition": "Explicit",
            "conversation": [{"user": "u0", "assistant": "a0"}],
            "user_message": "x",
            "y_pos": "pos",
            "y_neg": "neg",
            "y_refusal": "ref",
        }
        fake_result = DifferentialResult(
            baseline_log_probs={"y_pos": 10.0, "y_neg": 8.0, "y_refusal": 6.0},
            steps=[
                StepScores(
                    step=0,
                    turn=-1,
                    role="baseline",
                    baseline_log_probs={"y_pos": 10.0, "y_neg": 8.0, "y_refusal": 6.0},
                    step_log_probs={"y_pos": 10.0, "y_neg": 8.0, "y_refusal": 6.0},
                    deltas={"y_pos": 0.0, "y_neg": 0.0, "y_refusal": 0.0},
                ),
                StepScores(
                    step=1,
                    turn=0,
                    role="user",
                    baseline_log_probs={"y_pos": 10.0, "y_neg": 8.0, "y_refusal": 6.0},
                    step_log_probs={"y_pos": 7.0, "y_neg": 9.0, "y_refusal": 5.0},
                    deltas={"y_pos": -3.0, "y_neg": 1.0, "y_refusal": -1.0},
                ),
            ],
        )

        with patch("hr.experiments.harm_kl.compute_drift.compute_stepwise_differentials", return_value=fake_result):
            out = compute_kl_drift(pref_data, model=object(), tokenizer=object(), beta=0.1, system_prompt="sys")

        self.assertEqual(out["case_id"], "case_1")
        self.assertIn("per_step", out)
        self.assertEqual(out["T"], 1)

        step1 = out["per_step"][1]
        self.assertAlmostEqual(step1["delta_ypos"], -3.0)
        self.assertAlmostEqual(step1["delta_yneg"], 1.0)
        self.assertAlmostEqual(step1["delta_yrefuse"], -1.0)
        self.assertAlmostEqual(step1["harm_amplification"], 4.0)
        self.assertAlmostEqual(step1["harmful_compliance_drift"], 2.0)
        self.assertAlmostEqual(step1["safety_quality_erosion"], 2.0)
        self.assertIn("pref_prob", step1)


if __name__ == "__main__":
    unittest.main()
