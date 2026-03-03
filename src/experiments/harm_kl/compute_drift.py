"""
Per-step KL-drift computation for harm amplification measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.data_paths import load_data_paths, resolve_path_spec
from src.core.differentials import compute_stepwise_differentials, preference_probability
from src.core.types import DifferentialRequest, TrajectoryTurn


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(spec: str, data_paths: dict) -> Path:
    return resolve_path_spec(spec, data_paths)


def compute_kl_drift(pref_data: dict, model, tokenizer, beta: float, system_prompt: str) -> dict:
    conversation = [TrajectoryTurn(user=t["user"], assistant=t["assistant"]) for t in pref_data["conversation"]]
    x = pref_data["user_message"]
    y_pos = pref_data["y_pos"]
    y_neg = pref_data["y_neg"]
    y_refusal = pref_data["y_refusal"]

    result = compute_stepwise_differentials(
        model=model,
        tokenizer=tokenizer,
        request=DifferentialRequest(
            system_prompt=system_prompt,
            trajectory=conversation,
            x=x,
            y_candidates={
                "y_pos": y_pos,
                "y_neg": y_neg,
                "y_refusal": y_refusal,
            },
        ),
    )

    per_step = []
    for step in result.steps:
        delta_ypos = step.deltas["y_pos"]
        delta_yneg = step.deltas["y_neg"]
        delta_yrefuse = step.deltas["y_refusal"]
        per_step.append(
            {
                "step": step.step,
                "turn": step.turn,
                "role": step.role,
                "log_pi0_ypos": step.baseline_log_probs["y_pos"],
                "log_pi0_yneg": step.baseline_log_probs["y_neg"],
                "log_pi0_yrefuse": step.baseline_log_probs["y_refusal"],
                "log_pis_ypos": step.step_log_probs["y_pos"],
                "log_pis_yneg": step.step_log_probs["y_neg"],
                "log_pis_yrefuse": step.step_log_probs["y_refusal"],
                "delta_ypos": delta_ypos,
                "delta_yneg": delta_yneg,
                "delta_yrefuse": delta_yrefuse,
                "harm_amplification": delta_yneg - delta_ypos,
                "pref_prob": preference_probability(delta_ypos, delta_yneg, beta),
                "harmful_compliance_drift": delta_yneg - delta_yrefuse,
                "safety_quality_erosion": delta_yrefuse - delta_ypos,
            }
        )

    return {
        "case_id": pref_data["case_id"],
        "name": pref_data["name"],
        "theme": pref_data["theme"],
        "harm_type": pref_data["harm_type"],
        "condition": pref_data.get("condition", ""),
        "T": len(conversation),
        "user_message": x,
        "y_pos": y_pos,
        "y_neg": y_neg,
        "y_refusal": y_refusal,
        "per_step": per_step,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute per-turn KL-drift scores.")
    parser.add_argument("--config", required=True, help="Path to kl_drift.yaml")
    parser.add_argument("--case_id", default=None, help="Process a single case by id (optional)")
    parser.add_argument("--overwrite", action="store_true", help="Re-compute even if output exists")
    parser.add_argument(
        "--crossover",
        action="store_true",
        help=(
            "For implicit cases, substitute the explicit user_message from the "
            "matching theme. Results are written to output_dir + '_crossover'."
        ),
    )
    parser.add_argument(
        "--data-paths-config",
        default="configs/data_paths.yaml",
        help="Path to data path registry yaml.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    data_paths = load_data_paths(args.data_paths_config)

    prefs_dir = _resolve_path(config["preferences_dir"], data_paths)
    base_output = _resolve_path(config["output_dir"], data_paths)
    output_dir = Path(str(base_output) + "_crossover") if args.crossover else base_output
    output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = config["neutral_system_prompt"].strip()
    beta = float(config["beta"])

    print(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    pref_files = sorted(prefs_dir.glob("*.json"))
    if args.case_id:
        pref_files = [f for f in pref_files if f.stem == args.case_id]

    if not pref_files:
        raise FileNotFoundError(
            f"No preference files found in {prefs_dir}. "
            "Run harm seed-preferences and harm generate-conversations first."
        )

    explicit_user_message: dict[str, str] = {}
    if args.crossover:
        for pref_file in pref_files:
            with open(pref_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("condition") == "Explicit":
                explicit_user_message[data["theme"]] = data["user_message"]

    all_results = []
    for pref_file in pref_files:
        with open(pref_file, "r", encoding="utf-8") as f:
            pref_data = json.load(f)

        user_message_source = "original"
        if args.crossover and pref_data.get("condition") == "Implicit":
            theme = pref_data["theme"]
            if theme in explicit_user_message:
                pref_data = dict(pref_data)
                pref_data["user_message"] = explicit_user_message[theme]
                user_message_source = "explicit_crossover"
                print(f"  [crossover] {pref_data['name']}: using explicit user_message")
            else:
                print(f"  [crossover] WARNING: no explicit counterpart found for theme '{theme}'")

        out_path = output_dir / f"{pref_data['case_id']}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {pref_data['case_id']} - already exists")
            with open(out_path, "r", encoding="utf-8") as f:
                all_results.append(json.load(f))
            continue

        print(f"[compute] {pref_data['name']}  (T={len(pref_data['conversation'])})")
        case_result = compute_kl_drift(pref_data, model, tokenizer, beta, system_prompt)
        case_result["user_message_source"] = user_message_source

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(case_result, f, indent=2)
        all_results.append(case_result)
        print(f"  -> saved {out_path}")

    agg_path = output_dir / "all_results.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAggregated results -> {agg_path}")


if __name__ == "__main__":
    main()
