from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class TrajectoryTurn:
    user: str
    assistant: str


@dataclass
class DifferentialRequest:
    system_prompt: str
    trajectory: list[TrajectoryTurn]
    x: str
    y_candidates: dict[str, str]
    response_scorer: Callable[[object, object, list[dict[str, str]], str], float] | None = None


@dataclass(frozen=True)
class StepScores:
    step: int
    turn: int
    role: str
    baseline_log_probs: dict[str, float]
    step_log_probs: dict[str, float]
    deltas: dict[str, float]


@dataclass(frozen=True)
class DifferentialResult:
    baseline_log_probs: dict[str, float]
    steps: list[StepScores]


ROLE_BASELINE = "baseline"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


def role_for_step(step: int) -> str:
    if step == 0:
        return ROLE_BASELINE
    if step % 2 == 1:
        return ROLE_USER
    return ROLE_ASSISTANT


__all__ = [
    "DifferentialRequest",
    "DifferentialResult",
    "ROLE_ASSISTANT",
    "ROLE_BASELINE",
    "ROLE_USER",
    "StepScores",
    "TrajectoryTurn",
    "role_for_step",
]
