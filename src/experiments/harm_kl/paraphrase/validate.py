"""Check whether paraphrased versions of X move together in model scoring.

This file does not create paraphrases. It reads the generated X variants from
the paraphrase data file, then compares whether those variants produce similar
score movement for y_pos, y_neg, and y_refusal across the same conversation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.reader import load_yaml
from src.core.differentials import build_messages, response_token_stats


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Series lengths must match.")
    if len(xs) < 2:
        return 1.0

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_den == 0.0 or y_den == 0.0:
        return 1.0 if xs == ys else 0.0
    return num / (x_den * y_den)


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_rank(xs), _rank(ys))


def _step_direction_agreement(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Series lengths must match.")
    if len(xs) < 2:
        return 1.0

    matches = 0
    total = 0
    for idx in range(1, len(xs)):
        x_change = xs[idx] - xs[idx - 1]
        y_change = ys[idx] - ys[idx - 1]
        if x_change == 0.0 and y_change == 0.0:
            matches += 1
        elif x_change == 0.0 or y_change == 0.0:
            pass
        elif x_change * y_change > 0.0:
            matches += 1
        total += 1
    return 1.0 if total == 0 else matches / total


def _normalize_variant(case_id: str, idx: int, raw_variant: str | dict) -> dict[str, str]:
    if isinstance(raw_variant, str):
        return {
            "variant_id": f"{case_id}_v{idx + 1}",
            "label": f"variant_{idx + 1}",
            "text": raw_variant,
        }
    if isinstance(raw_variant, dict):
        text = raw_variant.get("text", "").strip()
        if not text:
            raise ValueError(f"Variant {idx + 1} for {case_id} is missing text.")
        variant_id = raw_variant.get("variant_id") or raw_variant.get("id") or f"{case_id}_v{idx + 1}"
        label = raw_variant.get("label") or variant_id
        return {
            "variant_id": str(variant_id),
            "label": str(label),
            "text": text,
        }
    raise ValueError(f"Unsupported variant format for {case_id}: {type(raw_variant).__name__}")


def load_variant_map(path: str | Path) -> dict[str, list[dict[str, str]]]:
    with open(Path(path), "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "cases" in data:
        cases = data["cases"]
    elif isinstance(data, dict):
        cases = [{"id": case_id, "variants": variants} for case_id, variants in data.items()]
    else:
        raise ValueError("Variant file must be a mapping or an object with a 'cases' list.")

    variant_map: dict[str, list[dict[str, str]]] = {}
    for case in cases:
        case_id = case.get("id")
        if not case_id:
            raise ValueError("Each variant case entry must include 'id'.")
        raw_variants = case.get("variants", [])
        if not raw_variants:
            raise ValueError(f"Case {case_id} has no variants.")
        variant_map[case_id] = [
            _normalize_variant(str(case_id), idx, raw_variant)
            for idx, raw_variant in enumerate(raw_variants)
        ]
    return variant_map


def evaluate_variant(
    pref_data: dict,
    variant: dict[str, str],
    model,
    tokenizer,
    system_prompt: str,
    target_responses: list[str],
) -> dict:
    y_candidates = {
        "y_pos": pref_data["y_pos"],
        "y_neg": pref_data["y_neg"],
        "y_refusal": pref_data["y_refusal"],
    }
    invalid_targets = [target for target in target_responses if target not in y_candidates]
    if invalid_targets:
        raise ValueError(f"Unknown target_responses: {invalid_targets}.")

    total_steps = 2 * len(pref_data["conversation"])
    baseline_messages = build_messages(system_prompt, [], 0, variant["text"])
    baseline_stats = {
        name: response_token_stats(model, tokenizer, baseline_messages, response)
        for name, response in y_candidates.items()
    }

    per_step = []
    tracked_series = {
        target: {
            "delta_series": [],
            "perplexity_series": [],
        }
        for target in target_responses
    }
    for step in range(total_steps + 1):
        step_messages = build_messages(system_prompt, pref_data["conversation"], step, variant["text"])
        step_stats = {
            name: response_token_stats(model, tokenizer, step_messages, response)
            for name, response in y_candidates.items()
        }
        for target in target_responses:
            tracked_series[target]["delta_series"].append(
                step_stats[target]["log_prob"] - baseline_stats[target]["log_prob"]
            )
            tracked_series[target]["perplexity_series"].append(step_stats[target]["perplexity"])

        per_step.append(
            {
                "step": step,
                "tracked_targets": {
                    target: {
                        "delta_log_prob": tracked_series[target]["delta_series"][-1],
                        "perplexity": tracked_series[target]["perplexity_series"][-1],
                    }
                    for target in target_responses
                },
                "responses": {
                    name: {
                        "baseline_log_prob": baseline_stats[name]["log_prob"],
                        "baseline_perplexity": baseline_stats[name]["perplexity"],
                        "step_log_prob": step_stats[name]["log_prob"],
                        "step_perplexity": step_stats[name]["perplexity"],
                        "delta_log_prob": step_stats[name]["log_prob"] - baseline_stats[name]["log_prob"],
                    }
                    for name in y_candidates
                },
            }
        )

    return {
        "variant_id": variant["variant_id"],
        "label": variant["label"],
        "text": variant["text"],
        "tracked_targets": {
            target: {
                "baseline": baseline_stats[target],
                "delta_series": tracked_series[target]["delta_series"],
                "perplexity_series": tracked_series[target]["perplexity_series"],
            }
            for target in target_responses
        },
        "per_step": per_step,
    }


def summarize_alignment(variant_runs: list[dict], target_responses: list[str]) -> dict:
    if len(variant_runs) < 2:
        return {
            "pair_count": 0,
            "targets": {
                target: {
                    "avg_delta_pearson": 1.0,
                    "min_delta_pearson": 1.0,
                    "avg_perplexity_spearman": 1.0,
                    "min_perplexity_spearman": 1.0,
                    "avg_step_direction_agreement": 1.0,
                    "min_step_direction_agreement": 1.0,
                }
                for target in target_responses
            },
            "pairs": [],
        }

    pairs = []
    for left_idx in range(len(variant_runs)):
        for right_idx in range(left_idx + 1, len(variant_runs)):
            left = variant_runs[left_idx]
            right = variant_runs[right_idx]
            pairs.append(
                {
                    "left_variant_id": left["variant_id"],
                    "right_variant_id": right["variant_id"],
                    "targets": {
                        target: {
                            "delta_pearson": _pearson(
                                left["tracked_targets"][target]["delta_series"],
                                right["tracked_targets"][target]["delta_series"],
                            ),
                            "perplexity_spearman": _spearman(
                                left["tracked_targets"][target]["perplexity_series"],
                                right["tracked_targets"][target]["perplexity_series"],
                            ),
                            "step_direction_agreement": _step_direction_agreement(
                                left["tracked_targets"][target]["perplexity_series"],
                                right["tracked_targets"][target]["perplexity_series"],
                            ),
                        }
                        for target in target_responses
                    },
                }
            )

    target_summary = {}
    for target in target_responses:
        target_pairs = [pair["targets"][target] for pair in pairs]
        target_summary[target] = {
            "avg_delta_pearson": sum(pair["delta_pearson"] for pair in target_pairs) / len(target_pairs),
            "min_delta_pearson": min(pair["delta_pearson"] for pair in target_pairs),
            "avg_perplexity_spearman": (
                sum(pair["perplexity_spearman"] for pair in target_pairs) / len(target_pairs)
            ),
            "min_perplexity_spearman": min(pair["perplexity_spearman"] for pair in target_pairs),
            "avg_step_direction_agreement": (
                sum(pair["step_direction_agreement"] for pair in target_pairs) / len(target_pairs)
            ),
            "min_step_direction_agreement": min(
                pair["step_direction_agreement"] for pair in target_pairs
            ),
        }

    return {
        "pair_count": len(pairs),
        "targets": target_summary,
        "pairs": pairs,
    }


def run_from_config(config: dict) -> None:
    case_id = config.get("case_id")
    overwrite = bool(config.get("overwrite", False))
    prefs_dir = Path(config["preferences_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_map = load_variant_map(config["variants_path"])
    system_prompt = config["neutral_system_prompt"].strip()
    configured_targets = config.get("target_responses")
    if configured_targets is None:
        legacy_target = config.get("target_response")
        target_responses = [legacy_target] if legacy_target else ["y_pos", "y_neg", "y_refusal"]
    else:
        target_responses = list(configured_targets)

    print(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    pref_files = sorted(prefs_dir.glob("*.json"))
    if case_id:
        pref_files = [f for f in pref_files if f.stem == case_id]

    if not pref_files:
        raise FileNotFoundError(f"No preference files found in {prefs_dir}.")

    all_results = []
    for pref_file in pref_files:
        with open(pref_file, "r", encoding="utf-8") as f:
            pref_data = json.load(f)

        variants = variant_map.get(pref_data["case_id"], [])
        if not variants:
            print(f"[skip] {pref_data['case_id']} - no variants configured")
            continue
        if len(variants) < 2:
            print(f"[warn] {pref_data['case_id']} - only one variant configured; alignment will be trivial")

        out_path = output_dir / f"{pref_data['case_id']}.json"
        if out_path.exists() and not overwrite:
            print(f"[skip] {pref_data['case_id']} - already exists")
            with open(out_path, "r", encoding="utf-8") as f:
                all_results.append(json.load(f))
            continue

        print(f"[validate] {pref_data['name']} ({len(variants)} variants)")
        variant_runs = [
            evaluate_variant(
                pref_data=pref_data,
                variant=variant,
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                target_responses=target_responses,
            )
            for variant in variants
        ]
        result = {
            "case_id": pref_data["case_id"],
            "name": pref_data["name"],
            "theme": pref_data["theme"],
            "harm_type": pref_data["harm_type"],
            "target_responses": target_responses,
            "variant_count": len(variant_runs),
            "variants": variant_runs,
            "alignment": summarize_alignment(variant_runs, target_responses),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        all_results.append(result)
        print(f"  -> saved {out_path}")

    agg_path = output_dir / "all_results.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAggregated results -> {agg_path}")


def run_from_config_path(config_path: str) -> None:
    run_from_config(load_yaml(config_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate how paraphrased variants of X co-move.")
    parser.add_argument(
        "--config",
        default="configs/harm_kl/paraphrase/validate.yaml",
        help="Path to the YAML config for this step.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_from_config_path(args.config)
