#!/usr/bin/env python3
"""Verify the hosted combined parquet against the locally built artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="danielfein/WildDelusionCombined")
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--local",
        type=Path,
        default=Path("data/releases/WildDelusionCombined/train.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/iclr2026/artifacts/combined_release_integrity.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    hosted = Path(
        hf_hub_download(
            repo_id=args.repo,
            repo_type="dataset",
            revision=args.revision,
            filename="data/train-00000-of-00001.parquet",
            force_download=True,
        )
    )
    local_digest = sha256(args.local)
    hosted_digest = sha256(hosted)
    if local_digest != hosted_digest:
        raise ValueError("Hosted parquet does not match the local release artifact")
    frame = pd.read_parquet(hosted)
    if frame["source"].astype(str).str.contains("lmsys", case=False).any():
        raise ValueError("Hosted release contains an LMSYS-derived row")
    target_errors = 0
    for row in frame.to_dict(orient="records"):
        target = row["messages"][int(row["target_message_index"])]
        if target["role"].lower() != "user" or target["content"] != row["target_text"]:
            target_errors += 1
    if target_errors:
        raise ValueError(f"Hosted release contains {target_errors} target integrity errors")
    result = {
        "passed": True,
        "repo": args.repo,
        "revision": args.revision,
        "rows": len(frame),
        "lmsys_rows": 0,
        "target_integrity_errors": 0,
        "parquet_sha256": hosted_digest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
