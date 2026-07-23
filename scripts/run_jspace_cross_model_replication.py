#!/usr/bin/env python3
"""Run one frozen same-model J-space replication, or aggregate completed runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/jspace_cross_model_replication.json",
    )
    parser.add_argument(
        "--model",
        help="Model name from the config. Omit with --aggregate-only.",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--concurrency", type=int, default=40)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def output_dir(model: dict[str, Any]) -> Path:
    return ROOT / Path(model["generation_path"]).parent


def run_model(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for the two outcome judges")
    matches = [model for model in config["models"] if model["name"] == args.model]
    if len(matches) != 1:
        names = ", ".join(model["name"] for model in config["models"])
        raise SystemExit(f"Unknown --model {args.model!r}; choose one of: {names}")
    model = matches[0]
    if model.get("role") == "indicator_source_model":
        raise SystemExit("The source-model run is archival; select a replication model")

    destination = output_dir(model)
    destination.mkdir(parents=True, exist_ok=True)
    prompts = str(ROOT / config["prompt_path"])
    holdout = str(ROOT / config["holdout_ids_path"])
    generations = str(ROOT / model["generation_path"])
    readouts = str(ROOT / model["readout_path"])
    framing = str(ROOT / model["framing_judgment_path"])
    package = str(ROOT / model["package_judgment_path"])
    shared_model = ["--model-id", model["model_id"], "--model-revision", model["model_revision"]]

    generation_command = [
            args.python,
            "scripts/generate_jspace_matched_responses.py",
            "--input",
            prompts,
            "--output",
            generations,
            *shared_model,
            "--intervention-arm",
            "direct_assertion",
            "--selected-original-indices",
            holdout,
            "--batch-size",
            "4",
            "--max-new-tokens",
            "192",
            "--resume",
        ]
    if model["weight_precision"] == "native MXFP4":
        generation_command.append("--quantize-4bit")
    run(generation_command)
    readout_command = [
        args.python,
        "scripts/run_jspace_semantic_counterfactuals.py",
        "--input",
        prompts,
        "--output",
        readouts,
        "--concept-spec",
        "configs/jspace_indicator_vocabulary.json",
        *shared_model,
        "--lens-repo",
        config["lens_repo"],
        "--lens-revision",
        config["lens_revision"],
        "--lens-file",
        model["lens_file"],
        "--layers",
        str(model["mid_layer"]),
        str(model["late_layer"]),
        "--max-seq-len",
        "4096",
        "--selected-original-indices",
        holdout,
        "--selected-arms",
        "direct_assertion",
        "--resume",
    ]
    if model["weight_precision"] == "native MXFP4":
        readout_command.append("--quantize-4bit")
    run(readout_command)
    common_judge = [
        "--model",
        "gpt-5.4-mini",
        "--reasoning-effort",
        "low",
        "--concurrency",
        str(args.concurrency),
        "--resume",
    ]
    run(
        [
            args.python,
            "scripts/judge_semantic_counterfactual_responses_openai.py",
            "--generations",
            generations,
            "--prompts",
            prompts,
            "--output",
            framing,
            *common_judge,
        ]
    )
    run(
        [
            args.python,
            "scripts/judge_generated_responses_with_package.py",
            "--input",
            generations,
            "--prompts",
            prompts,
            "--output",
            package,
            *common_judge,
        ]
    )


def aggregate(args: argparse.Namespace) -> None:
    for script, destination in (
        ("scripts/analyze_jspace_cross_model_replication.py", "results/jspace_cross_model_replication"),
        ("scripts/analyze_jspace_cross_model_indicator_panel.py", "results/jspace_cross_model_indicator_panel"),
    ):
        run(
            [
                args.python,
                script,
                "--config",
                str(args.config),
                "--output-dir",
                destination,
            ]
        )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not args.aggregate_only:
        if not args.model:
            raise SystemExit("--model is required unless --aggregate-only is set")
        run_model(args, config)
    aggregate(args)


if __name__ == "__main__":
    main()
