from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def build_messages(
    system_prompt: str,
    trajectory: list[dict[str, str]],
    step: int,
    x: str,
) -> list[dict[str, str]]:
    """Build messages for p(y | t, x) at a given step.

    - `trajectory` contains prior user/assistant turns (t).
    - `x` is the candidate user question appended after the chosen step context.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for i in range(step):
        turn_idx = i // 2
        if turn_idx >= len(trajectory):
            break
        turn = trajectory[turn_idx]
        if i % 2 == 0:
            messages.append({"role": "user", "content": turn["user"]})
        else:
            messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": x})
    return messages


def role_for_step(step: int) -> str:
    if step == 0:
        return "baseline"
    if step % 2 == 1:
        return "user"
    return "assistant"


@torch.inference_mode()
def response_log_prob(model, tokenizer, messages: list[dict[str, str]], response: str) -> float:
    """Compute log p(response | messages) as assistant continuation tokens."""
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["input_ids"]

    response_ids = tokenizer(
        response,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids
    if response_ids.shape[1] == 0:
        return float("-inf")

    prompt_len = prompt_ids.shape[1]
    full_ids = torch.cat([prompt_ids, response_ids], dim=1).to(model.device)

    logits = model(full_ids).logits
    shift_logits = logits[0, prompt_len - 1 : -1]
    shift_labels = full_ids[0, prompt_len:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
    return token_log_probs.sum().item()


def compute_stepwise_differentials(
    model,
    tokenizer,
    system_prompt: str,
    trajectory: list[dict[str, str]],
    x: str,
    y_candidates: dict[str, str],
    response_scorer: Callable[[object, object, list[dict[str, str]], str], float] | None = None,
) -> dict:
    scorer = response_scorer or response_log_prob

    baseline_messages = build_messages(system_prompt, [], 0, x)
    baseline_log_probs = {
        name: scorer(model, tokenizer, baseline_messages, y)
        for name, y in y_candidates.items()
    }

    total_steps = 2 * len(trajectory)
    steps = []
    prev_deltas: dict[str, float] | None = None
    for step in range(total_steps + 1):
        step_messages = build_messages(system_prompt, trajectory, step, x)
        step_log_probs = {
            name: scorer(model, tokenizer, step_messages, y)
            for name, y in y_candidates.items()
        }
        deltas = {
            name: step_log_probs[name] - baseline_log_probs[name]
            for name in y_candidates
        }
        delta_step_changes = {
            name: (0.0 if prev_deltas is None else deltas[name] - prev_deltas[name])
            for name in y_candidates
        }
        steps.append(
            {
                "step": step,
                "turn": (step - 1) // 2 if step > 0 else -1,
                "role": role_for_step(step),
                "baseline_log_probs": dict(baseline_log_probs),
                "step_log_probs": step_log_probs,
                "deltas": deltas,
                "delta_step_changes": delta_step_changes,
            }
        )
        prev_deltas = deltas

    return {
        "baseline_log_probs": baseline_log_probs,
        "steps": steps,
    }


def compute_single_y_delta_series(
    model,
    tokenizer,
    system_prompt: str,
    trajectory: list[dict[str, str]],
    x: str,
    y: str,
) -> list[float]:
    result = compute_stepwise_differentials(
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        trajectory=trajectory,
        x=x,
        y_candidates={"target": y},
    )
    return [step["deltas"]["target"] for step in result["steps"]]


def preference_probability(delta_pos: float, delta_neg: float, beta: float) -> float:
    logit = beta * (delta_pos - delta_neg)
    return torch.sigmoid(torch.tensor(logit)).item()


__all__ = [
    "build_messages",
    "compute_single_y_delta_series",
    "compute_stepwise_differentials",
    "preference_probability",
    "response_log_prob",
]
