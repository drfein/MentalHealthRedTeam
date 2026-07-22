#!/usr/bin/env python3
"""Build license-safe legacy candidates for current-pipeline re-verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from wild_delusion_miner.legacy_expansion import (
    LEGACY_COMBINED_REPO,
    LEGACY_COMBINED_REVISION,
    LEGACY_VERIFIED_REPO,
    LEGACY_VERIFIED_REVISION,
    select_legacy_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current-release",
        type=Path,
        default=Path("data/releases/WildDelusionVerified/train.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/legacy_expansion/candidates.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/legacy_expansion/candidate_manifest.json"),
    )
    parser.add_argument("--legacy-verified-parquet", type=Path)
    parser.add_argument("--legacy-combined-parquet", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def download_parquet(repo: str, revision: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            filename="data/train-00000-of-00001.parquet",
        )
    )


def main() -> None:
    args = parse_args()
    verified_path = args.legacy_verified_parquet or download_parquet(
        LEGACY_VERIFIED_REPO, LEGACY_VERIFIED_REVISION
    )
    combined_path = args.legacy_combined_parquet or download_parquet(
        LEGACY_COMBINED_REPO, LEGACY_COMBINED_REVISION
    )
    selected, manifest = select_legacy_candidates(
        pd.read_parquet(verified_path).to_dict(orient="records"),
        pd.read_parquet(combined_path).to_dict(orient="records"),
        read_jsonl(args.current_release),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
