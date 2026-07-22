#!/usr/bin/env python3
"""Verify that every paper experiment has code and committed-style aggregates."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/paper_experiments.json"),
    )
    parser.add_argument(
        "--require-git-tracked",
        action="store_true",
        help="Also fail when a manifest path is not tracked by git.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_tracked(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    experiments = manifest.get("experiments", [])
    if not experiments:
        raise SystemExit("Experiment manifest is empty")

    experiment_ids = [experiment["id"] for experiment in experiments]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise SystemExit("Experiment IDs must be unique")

    paths = [Path(manifest["paper"]), args.manifest]
    for experiment in experiments:
        if not experiment.get("paper_sections"):
            raise SystemExit(f"{experiment['id']} has no paper section mapping")
        if not experiment.get("code") or not experiment.get("artifacts"):
            raise SystemExit(f"{experiment['id']} must list code and artifacts")
        paths.extend(Path(path) for path in experiment["code"])
        paths.extend(Path(path) for path in experiment["artifacts"])

    missing = sorted({str(path) for path in paths if not path.is_file()})
    if missing:
        raise SystemExit("Missing manifest paths:\n" + "\n".join(missing))

    if args.require_git_tracked:
        untracked = sorted({str(path) for path in paths if not git_tracked(path)})
        if untracked:
            raise SystemExit("Untracked manifest paths:\n" + "\n".join(untracked))

    result = {
        "passed": True,
        "experiments": len(experiments),
        "unique_code_paths": len(
            {path for experiment in experiments for path in experiment["code"]}
        ),
        "unique_artifact_paths": len(
            {path for experiment in experiments for path in experiment["artifacts"]}
        ),
        "git_tracking_checked": args.require_git_tracked,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
