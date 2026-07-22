from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wild_delusion_miner.context_interventions import build_history_arms


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def successful_replacements(
    path: Path,
    model: str,
    replacement_suffix: str,
) -> dict[str, dict[str, Any]]:
    replacements: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("model") != model or row.get("error_type"):
            continue
        response = str(row.get("response_text") or "").strip()
        message_hash = str(row.get("message_hash") or "")
        suffix = f":{replacement_suffix}"
        if not response or not message_hash.endswith(suffix):
            continue
        original_hash = message_hash.removesuffix(suffix)
        replacements[original_hash] = row
    return replacements


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build original, assistant-removed, and GPT-5.2-substituted context arms."
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("data/releases/WildDelusionCombined/train.parquet"),
    )
    parser.add_argument(
        "--replacements",
        type=Path,
        default=Path("data/assistant_history_substitution/gpt52_replacements_k32.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/assistant_history_substitution/intervention_arms.parquet"),
    )
    parser.add_argument("--replacement-model", default="gpt-5.2-2025-12-11")
    parser.add_argument("--replacement-suffix", default="gpt52_replacement_k32")
    parser.add_argument("--max-prior-messages", type=int, default=32)
    parser.add_argument("--shuffle-seed", type=int, default=20260722)
    args = parser.parse_args()

    release = pd.read_parquet(args.release).reset_index(names="original_row_idx")
    replacements = successful_replacements(
        args.replacements,
        args.replacement_model,
        args.replacement_suffix,
    )
    rows = []
    for row in release.to_dict(orient="records"):
        replacement = replacements.get(str(row["message_hash"]))
        if replacement is None:
            continue
        rows.extend(
            build_history_arms(
                row,
                int(row["original_row_idx"]),
                str(replacement["response_text"]),
                str(replacement["generation_id"]),
                max_prior_messages=args.max_prior_messages,
            )
        )
    output = pd.DataFrame(rows)
    counts = output.groupby("original_row_idx")["context_arm"].nunique()
    if not counts.eq(3).all():
        raise ValueError("Every included target must have exactly three intervention arms.")
    order = np.random.default_rng(args.shuffle_seed).permutation(len(output))
    output = output.iloc[order].reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    manifest = {
        "release": str(args.release),
        "replacements": str(args.replacements),
        "output": str(args.output),
        "targets": int(output["original_row_idx"].nunique()),
        "rows": len(output),
        "arms": ["original", "assistant_removed", "gpt52_substituted"],
        "shuffle_seed": args.shuffle_seed,
        "max_prior_messages": args.max_prior_messages,
        "primary_contrast": "gpt52_substituted minus original",
        "secondary_contrast": "gpt52_substituted minus assistant_removed",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
