#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LEGACY_ENTRYPOINT_RE = re.compile(
    r"python(?:3)?\s+(?:-m\s+)?(?:experiments\.|experiments/|harm_kl/experiments/)")
ABSOLUTE_DATA_RE = re.compile(r"/nlp/scr/drfein")


EXCLUDED_CONFIGS = {
    ROOT / "configs" / "data_paths.yaml",
}


def check_slurm_wrappers() -> list[str]:
    errors: list[str] = []
    for path in sorted((ROOT / "slurm").glob("run_*.sh")):
        text = path.read_text(encoding="utf-8")
        if LEGACY_ENTRYPOINT_RE.search(text):
            errors.append(f"{path}: uses legacy experiment entrypoint instead of python -m src.run")
    return errors


def check_config_paths() -> list[str]:
    errors: list[str] = []
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        if path in EXCLUDED_CONFIGS:
            continue
        text = path.read_text(encoding="utf-8")
        if ABSOLUTE_DATA_RE.search(text):
            errors.append(f"{path}: contains absolute data path; use configs/data_paths.yaml")
    return errors


def main() -> int:
    errors = []
    errors.extend(check_slurm_wrappers())
    errors.extend(check_config_paths())

    if errors:
        print("Standardization lint failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Standardization lint passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
