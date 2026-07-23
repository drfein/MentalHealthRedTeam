from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens
from jlens.hooks import ActivationRecorder

from run_jspace_context_ablation import DEFAULT_SYSTEM_PROMPT, token_ids_for_words


CONDITIONS = {
    "primary_down": ("primary", -1.0),
    "baseline": ("primary", 0.0),
    "primary_up": ("primary", 1.0),
    "random_down": ("random", -1.0),
    "random_up": ("random", 1.0),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, *row["messages"]],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def score_residual(
    residual: torch.Tensor,
    lens: Any,
    lens_model: Any,
    layer: int,
    positive_token_ids: list[int],
    negative_token_ids: list[int] | None = None,
) -> torch.Tensor:
    transported = lens.transport(residual.float(), layer)
    logits = lens_model.unembed(transported)
    score = logits[positive_token_ids].float().mean()
    if negative_token_ids:
        score = score - logits[negative_token_ids].float().mean()
    return score


def primary_direction(
    residual: torch.Tensor,
    lens: Any,
    lens_model: Any,
    layer: int,
    positive_token_ids: list[int],
    negative_token_ids: list[int] | None = None,
) -> tuple[torch.Tensor, float]:
    source = residual.detach().float().requires_grad_(True)
    score = score_residual(
        source,
        lens,
        lens_model,
        layer,
        positive_token_ids,
        negative_token_ids,
    )
    gradient = torch.autograd.grad(score, source)[0].detach()
    direction = gradient / gradient.norm().clamp_min(1e-12)
    return direction, float(score.detach().cpu())


def random_orthogonal_direction(primary: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidate = torch.randn(primary.shape, generator=generator, dtype=torch.float32)
    candidate = candidate.to(primary.device)
    candidate -= torch.dot(candidate, primary) * primary
    return candidate / candidate.norm().clamp_min(1e-12)


@contextmanager
def steering_hook(
    block: torch.nn.Module,
    direction: torch.Tensor,
    dose: float,
) -> Iterator[None]:
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        last = modified[:, -1, :]
        residual_norm = last.float().norm(dim=-1, keepdim=True)
        delta = dose * residual_norm * direction.to(last.dtype).unsqueeze(0)
        modified[:, -1, :] = last + delta
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def prompt_residual(lens_model: Any, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
    with ActivationRecorder(lens_model.layers, at=[layer]) as recorder:
        lens_model.forward(input_ids)
    return recorder.activations[layer][0, -1].detach()


def select_rows(
    rows: list[dict[str, Any]], selected: set[int] | None, input_arm: str
) -> list[dict[str, Any]]:
    neutral = [
        row
        for row in rows
        if row.get("counterfactual_family", row.get("intervention_arm")) == input_arm
    ]
    if selected is not None:
        neutral = [row for row in neutral if int(row["original_row_idx"]) in selected]
    return neutral


def prepare_resume(output: Path) -> set[int]:
    if not output.exists():
        return set()
    existing = read_jsonl(output)
    by_message: dict[int, list[dict[str, Any]]] = {}
    for row in existing:
        by_message.setdefault(int(row["original_row_idx"]), []).append(row)
    complete = {
        message_idx
        for message_idx, message_rows in by_message.items()
        if {row["condition"] for row in message_rows} == set(CONDITIONS)
        and len(message_rows) == len(CONDITIONS)
    }
    retained = [row for row in existing if int(row["original_row_idx"]) in complete]
    temporary = output.with_suffix(f"{output.suffix}.resume-tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return complete


def build_crossfit_cache(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    lens: Any,
    lens_model: Any,
    layer: int,
    positive_token_ids: list[int],
    negative_token_ids: list[int],
    max_input_tokens: int,
    folds: int,
    seed: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, torch.Tensor]]:
    if folds < 2:
        raise ValueError("Cross-fitting requires at least two folds")
    device = next(lens_model.layers[0].parameters()).device
    cache: dict[int, dict[str, Any]] = {}
    for row in tqdm(rows, desc="crossfit directions"):
        row_idx = int(row["original_row_idx"])
        encoded = tokenizer(
            render_prompt(tokenizer, row),
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(device)
        residual = prompt_residual(lens_model, encoded.input_ids, layer)
        local_direction, base_score = primary_direction(
            residual,
            lens,
            lens_model,
            layer,
            positive_token_ids,
            negative_token_ids,
        )
        fold = stable_seed(str(row_idx), seed + 1_003) % folds
        cache[row_idx] = {
            "encoded": {key: value.cpu() for key, value in encoded.items()},
            "residual": residual.cpu(),
            "local_direction": local_direction.cpu(),
            "base_score": base_score,
            "fold": fold,
        }

    fold_directions: dict[int, torch.Tensor] = {}
    for held_out_fold in range(folds):
        training = [
            item["local_direction"] for item in cache.values() if item["fold"] != held_out_fold
        ]
        if not training:
            raise ValueError(f"Fold {held_out_fold} has no training directions")
        direction = torch.stack(training).mean(dim=0)
        fold_directions[held_out_fold] = (direction / direction.norm().clamp_min(1e-12)).to(device)
    return cache, fold_directions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causally steer a J-space token direction during generation."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--selected-indices", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument("--lens-revision", default=None)
    parser.add_argument(
        "--lens-file",
        default="qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt",
    )
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--primary-word", default="misinformation")
    parser.add_argument(
        "--positive-words",
        nargs="+",
        default=None,
        help="Words defining the positive side of a contrastive J-space axis.",
    )
    parser.add_argument(
        "--negative-words",
        nargs="+",
        default=None,
        help="Words subtracted from --positive-words; omit for a one-sided axis.",
    )
    parser.add_argument("--input-arm", default="neutral")
    parser.add_argument("--dose", type=float, default=0.02)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--direction-mode",
        choices=["per_example", "crossfit_mean"],
        default="per_example",
    )
    parser.add_argument("--crossfit-folds", type=int, default=2)
    args = parser.parse_args()

    selected = None
    if args.selected_indices:
        selected = {
            int(line.strip())
            for line in args.selected_indices.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    rows = select_rows(read_jsonl(args.input), selected, args.input_arm)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    completed = prepare_resume(args.output) if args.resume else set()
    rows = [row for row in rows if int(row["original_row_idx"]) not in completed]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    tokenizer.truncation_side = "left"
    lens_path = hf_hub_download(
        args.lens_repo,
        args.lens_file,
        revision=args.lens_revision,
    )
    lens = jlens.JacobianLens.load(lens_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    lens_model = jlens.from_hf(hf_model, tokenizer, force_bos=False)
    lens.jacobians[args.layer] = lens.jacobians[args.layer].to(hf_model.device)
    positive_words = args.positive_words or [args.primary_word]
    negative_words = args.negative_words or []
    positive_ids = sorted(
        {
            token_id
            for ids in token_ids_for_words(tokenizer, positive_words).values()
            for token_id in ids
        }
    )
    negative_ids = sorted(
        {
            token_id
            for ids in token_ids_for_words(tokenizer, negative_words).values()
            for token_id in ids
        }
    )
    if not positive_ids:
        raise ValueError("The positive side of the steering axis has no token IDs")
    crossfit_cache: dict[int, dict[str, Any]] = {}
    fold_directions: dict[int, torch.Tensor] = {}
    if args.direction_mode == "crossfit_mean":
        crossfit_cache, fold_directions = build_crossfit_cache(
            rows,
            tokenizer,
            lens,
            lens_model,
            args.layer,
            positive_ids,
            negative_ids,
            args.max_input_tokens,
            args.crossfit_folds,
            args.seed,
        )

    hparams = vars(args).copy()
    for key, value in list(hparams.items()):
        if isinstance(value, Path):
            hparams[key] = str(value)
    hparams["conditions"] = CONDITIONS
    hparams["axis"] = {
        "positive_words": positive_words,
        "negative_words": negative_words,
        "score": "mean(positive J-space logits) - mean(negative J-space logits)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "hparams.json").write_text(
        json.dumps(hparams, indent=2), encoding="utf-8"
    )

    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for row in tqdm(rows, desc="messages"):
            row_idx = int(row["original_row_idx"])
            if args.direction_mode == "crossfit_mean":
                cached = crossfit_cache[row_idx]
                encoded = {
                    key: value.to(hf_model.device) for key, value in cached["encoded"].items()
                }
                residual = cached["residual"].to(hf_model.device)
                fold = int(cached["fold"])
                primary = fold_directions[fold]
                base_score = float(cached["base_score"])
                local_to_steering_cosine = float(
                    torch.dot(cached["local_direction"].to(primary.device), primary).cpu()
                )
            else:
                encoded = tokenizer(
                    render_prompt(tokenizer, row),
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_input_tokens,
                ).to(hf_model.device)
                residual = prompt_residual(lens_model, encoded["input_ids"], args.layer)
                primary, base_score = primary_direction(
                    residual,
                    lens,
                    lens_model,
                    args.layer,
                    positive_ids,
                    negative_ids,
                )
                fold = None
                local_to_steering_cosine = 1.0
            random_direction = random_orthogonal_direction(
                primary,
                stable_seed(str(row_idx), args.seed),
            )
            residual_norm = float(residual.float().norm().cpu())

            for condition, (direction_name, sign) in CONDITIONS.items():
                direction = primary if direction_name == "primary" else random_direction
                signed_dose = sign * args.dose
                steered = residual.float() + signed_dose * residual_norm * direction
                achieved_score = float(
                    score_residual(
                        steered,
                        lens,
                        lens_model,
                        args.layer,
                        positive_ids,
                        negative_ids,
                    )
                    .detach()
                    .cpu()
                )
                with steering_hook(lens_model.layers[args.layer], direction, signed_dose):
                    with torch.inference_mode():
                        generated = hf_model.generate(
                            **encoded,
                            do_sample=False,
                            max_new_tokens=args.max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                new_tokens = generated[0, encoded["input_ids"].shape[1] :]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                result = {
                    "original_row_idx": row_idx,
                    "condition": condition,
                    "direction": direction_name,
                    "dose": signed_dose,
                    "source": row.get("source"),
                    "conversation_id": row.get("conversation_id"),
                    "message_hash": row.get("message_hash"),
                    "target_text": row["target_text"],
                    "prompt_tokens": int(encoded["input_ids"].shape[1]),
                    "layer": args.layer,
                    "primary_word": args.primary_word,
                    "axis_positive_words": positive_words,
                    "axis_negative_words": negative_words,
                    "direction_mode": args.direction_mode,
                    "crossfit_fold": fold,
                    "local_to_steering_cosine": local_to_steering_cosine,
                    "base_primary_score": base_score,
                    "steered_primary_score": achieved_score,
                    "achieved_primary_delta": achieved_score - base_score,
                    "achieved_axis_delta": achieved_score - base_score,
                    "residual_norm": residual_norm,
                    "response": response,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
