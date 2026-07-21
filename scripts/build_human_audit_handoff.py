#!/usr/bin/env python3
"""Build deterministic, blinded handoff archives for two independent reviewers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path


FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def csv_row_count(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("results/wilddelusion_release_human_audit"),
    )
    parser.add_argument(
        "--response-root",
        type=Path,
        default=Path(
            "results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals/"
            "full_public_behavior/human_audit"
        ),
    )
    parser.add_argument(
        "--instructions",
        type=Path,
        default=Path("paper/iclr2026/HUMAN_AUDIT_REVIEWER_INSTRUCTIONS.md"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/human_audit_handoff"),
    )
    return parser.parse_args()


def write_zip(path: Path, files: dict[str, Path]) -> None:
    path.unlink(missing_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for archive_name, source in sorted(files.items()):
            if not source.is_file():
                raise FileNotFoundError(source)
            info = zipfile.ZipInfo(archive_name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            output.writestr(info, source.read_bytes(), compresslevel=9)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archives = {}
    for rater in ("a", "b"):
        archive = args.output_dir / f"wilddelusion_reviewer_{rater}.zip"
        write_zip(
            archive,
            {
                "00_REVIEWER_INSTRUCTIONS.md": args.instructions,
                "01_USER_MESSAGE_REVIEW.html": (
                    args.release_root / f"blinded_review_rater_{rater}.html"
                ),
                "02_ASSISTANT_RESPONSE_REVIEW.html": (
                    args.response_root / f"blinded_review_rater_{rater}.html"
                ),
            },
        )
        archives[rater] = {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
    manifest = {
        "reviewers": archives,
        "release_rows_per_reviewer": csv_row_count(
            args.release_root / "blinded_review.csv"
        ),
        "response_rows_per_reviewer": csv_row_count(
            args.response_root / "blinded_review.csv"
        ),
        "contains_judge_keys": False,
        "contains_other_reviewer_labels": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
