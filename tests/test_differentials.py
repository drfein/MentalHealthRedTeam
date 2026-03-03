from __future__ import annotations

import unittest
from unittest import mock

from hr.core.differentials import _build_messages, compute_single_y_delta_series, compute_stepwise_differentials
from hr.core.types import DifferentialRequest, TrajectoryTurn


class DifferentialCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trajectory = [
            TrajectoryTurn(user="u0", assistant="a0"),
            TrajectoryTurn(user="u1", assistant="a1"),
        ]

    @staticmethod
    def fake_scorer(_model, _tokenizer, messages, response: str) -> float:
        return float(len(messages) * 10 + len(response))

    def test_step_zero_deltas_are_zero_for_all_candidates(self) -> None:
        request = DifferentialRequest(
            system_prompt="sys",
            trajectory=self.trajectory,
            x="x",
            y_candidates={"y_pos": "yes", "y_neg": "no", "y_refusal": "ref"},
            response_scorer=self.fake_scorer,
        )
        result = compute_stepwise_differentials(model=object(), tokenizer=object(), request=request)
        step0 = result.steps[0]
        self.assertEqual(step0.step, 0)
        self.assertEqual(step0.role, "baseline")
        self.assertEqual(step0.turn, -1)
        self.assertEqual(step0.deltas["y_pos"], 0.0)
        self.assertEqual(step0.deltas["y_neg"], 0.0)
        self.assertEqual(step0.deltas["y_refusal"], 0.0)

    def test_closed_form_delta_matches_message_count(self) -> None:
        request = DifferentialRequest(
            system_prompt="sys",
            trajectory=self.trajectory,
            x="x",
            y_candidates={"target": "abc"},
            response_scorer=self.fake_scorer,
        )
        result = compute_stepwise_differentials(model=object(), tokenizer=object(), request=request)

        for step in result.steps:
            expected_delta = float(step.step * 10)
            self.assertEqual(step.deltas["target"], expected_delta)

    def test_single_y_adapter_matches_named_candidate(self) -> None:
        scorer = self.fake_scorer
        request = DifferentialRequest(
            system_prompt="sys",
            trajectory=self.trajectory,
            x="x",
            y_candidates={"target": "abc"},
            response_scorer=scorer,
        )
        result = compute_stepwise_differentials(model=object(), tokenizer=object(), request=request)

        with mock.patch("hr.core.differentials.response_log_prob", new=scorer):
            series = compute_single_y_delta_series(
                model=object(),
                tokenizer=object(),
                system_prompt="sys",
                trajectory=self.trajectory,
                x="x",
                y="abc",
            )

        # Compute through a second request so adapter and direct path use same scorer contract.
        request_adapter = DifferentialRequest(
            system_prompt="sys",
            trajectory=self.trajectory,
            x="x",
            y_candidates={"target": "abc"},
            response_scorer=scorer,
        )
        result_adapter = compute_stepwise_differentials(model=object(), tokenizer=object(), request=request_adapter)

        self.assertEqual(
            [s.deltas["target"] for s in result.steps],
            [s.deltas["target"] for s in result_adapter.steps],
        )
        # Adapter default scorer path should preserve step count semantics.
        self.assertEqual(series, [s.deltas["target"] for s in result.steps])

    def test_build_messages_semantics_t0_t1_tgt1(self) -> None:
        m0 = _build_messages("sys", self.trajectory, 0, "x")
        self.assertEqual(m0, [{"role": "system", "content": "sys"}, {"role": "user", "content": "x"}])

        m1 = _build_messages("sys", self.trajectory, 1, "x")
        self.assertEqual(m1[1], {"role": "user", "content": "u0"})
        self.assertEqual(m1[-1], {"role": "user", "content": "x"})

        m3 = _build_messages("sys", self.trajectory, 3, "x")
        self.assertEqual(
            m3,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u0"},
                {"role": "assistant", "content": "a0"},
                {"role": "user", "content": "u1"},
                {"role": "user", "content": "x"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
