from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_ids(path: Path, values: list[int]) -> None:
    path.write_text("".join(f"{value}\n" for value in sorted(values)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a fixed group split balanced by observed positive-arm count."
    )
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-column", default="openai_score")
    parser.add_argument("--positive-threshold", type=float, default=7)
    parser.add_argument("--discovery-fraction", type=float, default=2 / 3)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    if not 0 < args.discovery_fraction < 1:
        raise ValueError("discovery-fraction must be strictly between zero and one")
    frame = pd.DataFrame(read_jsonl(args.judgments))
    required = {"original_row_idx", args.score_column}
    if missing := required - set(frame):
        raise ValueError(f"Judgments are missing columns: {sorted(missing)}")
    frame["positive"] = frame[args.score_column] >= args.positive_threshold
    strata = frame.groupby("original_row_idx")["positive"].sum().rename("positive_arms")

    rng = np.random.default_rng(args.seed)
    discovery: list[int] = []
    holdout: list[int] = []
    for _, stratum in strata.groupby(strata):
        ids = stratum.index.to_numpy(dtype=int, copy=True)
        rng.shuffle(ids)
        cut = int(round(len(ids) * args.discovery_fraction))
        discovery.extend(ids[:cut].tolist())
        holdout.extend(ids[cut:].tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_ids(args.output_dir / "discovery_ids.txt", discovery)
    write_ids(args.output_dir / "holdout_ids.txt", holdout)
    manifest = {
        "judgments": str(args.judgments),
        "score_column": args.score_column,
        "positive_threshold": args.positive_threshold,
        "stratification": "number of positive randomized-context arms per original message",
        "discovery_fraction": args.discovery_fraction,
        "seed": args.seed,
        "discovery_groups": len(discovery),
        "holdout_groups": len(holdout),
        "discovery_positive_arms": int(
            frame[frame["original_row_idx"].isin(discovery)]["positive"].sum()
        ),
        "holdout_positive_arms": int(
            frame[frame["original_row_idx"].isin(holdout)]["positive"].sum()
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
