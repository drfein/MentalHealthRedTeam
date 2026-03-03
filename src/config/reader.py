from __future__ import annotations

import os
from pathlib import Path

import yaml

DEFAULT_RUN_CONFIG_PATH = "configs/harm_kl/run.yaml"


def load_yaml(path: str | Path) -> dict:
    path_obj = Path(path)
    with open(path_obj, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path_obj}, got {type(data).__name__}.")
    return data


def resolve_run_config_path() -> Path:
    configured = os.environ.get("HR_RUN_CONFIG", DEFAULT_RUN_CONFIG_PATH)
    return Path(configured)


def load_run_config() -> dict:
    config = load_yaml(resolve_run_config_path())

    run_mode = config.get("run_mode", "run-all")
    skip_crossover = bool(config.get("skip_crossover", False))

    generate_conversations = dict(config.get("generate_conversations", {}))
    generate_conversations.setdefault("config_path", "configs/harm_kl/generate_preferences.yaml")

    compute_drift = dict(config.get("compute_drift", {}))
    compute_drift.setdefault("config_path", "configs/harm_kl/kl_drift.yaml")

    return {
        "run_mode": run_mode,
        "skip_crossover": skip_crossover,
        "generate_conversations": generate_conversations,
        "compute_drift": compute_drift,
    }
