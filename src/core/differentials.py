from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from src.core.types import DifferentialRequest, DifferentialResult, StepScores, TrajectoryTurn, role_for_step


def _build_messages(
    system_prompt: str,
    trajectory: list[TrajectoryTurn],
    step: int,
    x: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for i in range(step):
        turn_idx = i // 2
        if turn_idx >= len(trajectory):
            break
        turn = trajectory[turn_idx]
        if i % 2 == 0:
            messages.append({"role": "user", "content": turn.user})
        else:
            messages.append({"role": "assistant", "content": turn.assistant})
    messages.append({"role": "user", "content": x})
    return messages


@torch.inference_mode()
def response_log_prob(model, tokenizer, messages: list[dict[str, str]], response: str) -> float:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    full_text = prompt_text + response

    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

    prompt_len = prompt_ids.shape[1]
    response_len = full_ids.shape[1] - prompt_len
    if response_len <= 0:
        return float("-inf")

    logits = model(full_ids).logits
    shift_logits = logits[0, prompt_len - 1 : -1]
    shift_labels = full_ids[0, prompt_len:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
    return token_log_probs.sum().item()


def _get_scorer(request: DifferentialRequest) -> Callable[[object, object, list[dict[str, str]], str], float]:
    if request.response_scorer is not None:
        return request.response_scorer
    return response_log_prob


def compute_stepwise_differentials(model, tokenizer, request: DifferentialRequest) -> DifferentialResult:
    scorer = _get_scorer(request)

    baseline_messages = _build_messages(request.system_prompt, [], 0, request.x)
    baseline_log_probs = {
        name: scorer(model, tokenizer, baseline_messages, y) for name, y in request.y_candidates.items()
    }

    total_steps = 2 * len(request.trajectory)
    steps: list[StepScores] = []
    for step in range(total_steps + 1):
        step_messages = _build_messages(request.system_prompt, request.trajectory, step, request.x)
        step_log_probs = {
            name: scorer(model, tokenizer, step_messages, y) for name, y in request.y_candidates.items()
        }
        deltas = {
            name: step_log_probs[name] - baseline_log_probs[name] for name in request.y_candidates
        }
        steps.append(
            StepScores(
                step=step,
                turn=(step - 1) // 2 if step > 0 else -1,
                role=role_for_step(step),
                baseline_log_probs=dict(baseline_log_probs),
                step_log_probs=step_log_probs,
                deltas=deltas,
            )
        )

    return DifferentialResult(
        baseline_log_probs=baseline_log_probs,
        steps=steps,
    )


def compute_single_y_delta_series(
    model,
    tokenizer,
    system_prompt: str,
    trajectory: list[TrajectoryTurn],
    x: str,
    y: str,
) -> list[float]:
    result = compute_stepwise_differentials(
        model=model,
        tokenizer=tokenizer,
        request=DifferentialRequest(
            system_prompt=system_prompt,
            trajectory=trajectory,
            x=x,
            y_candidates={"target": y},
        ),
    )
    return [step.deltas["target"] for step in result.steps]


def preference_probability(delta_pos: float, delta_neg: float, beta: float) -> float:
    logit = beta * (delta_pos - delta_neg)
    return torch.sigmoid(torch.tensor(logit)).item()


__all__ = [
    "compute_single_y_delta_series",
    "compute_stepwise_differentials",
    "preference_probability",
    "response_log_prob",
]
