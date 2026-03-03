from __future__ import annotations

from src.config.reader import load_run_config
from src.experiments.harm_kl import compute_drift, generate_preferences


def run_generate_conversations(step_config: dict) -> None:
    config_path = step_config["config_path"]
    generate_preferences.run_from_config_path(config_path)


def run_compute_drift(step_config: dict, crossover: bool) -> None:
    config_path = step_config["config_path"]
    compute_drift.run_from_config_path(config_path, force_crossover=crossover)


def main() -> int:
    config = load_run_config()
    command = config["run_mode"]
    skip_crossover = config["skip_crossover"]

    if command == "generate-conversations":
        run_generate_conversations(config["generate_conversations"])
        return 0

    if command == "compute-drift":
        run_compute_drift(config["compute_drift"], crossover=False)
        return 0

    if command == "compute-drift-crossover":
        run_compute_drift(config["compute_drift"], crossover=True)
        return 0

    if command == "run-all":
        run_generate_conversations(config["generate_conversations"])
        run_compute_drift(config["compute_drift"], crossover=False)
        if not skip_crossover:
            run_compute_drift(config["compute_drift"], crossover=True)
        return 0

    raise ValueError(f"Unknown run_mode: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
