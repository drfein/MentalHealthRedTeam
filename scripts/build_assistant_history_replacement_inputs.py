from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wild_delusion_miner.context_interventions import build_replacement_input


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build causal GPT-5.2 inputs for replacing the assistant turn before each target."
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("data/releases/WildDelusionCombined/train.parquet"),
    )
    parser.add_argument("--max-prior-messages", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/assistant_history_substitution/replacement_inputs.parquet"),
    )
    args = parser.parse_args()

    release = pd.read_parquet(args.release).reset_index(names="original_row_idx")
    rows = []
    for row in release.to_dict(orient="records"):
        replacement = build_replacement_input(
            row,
            int(row["original_row_idx"]),
            max_prior_messages=args.max_prior_messages,
        )
        if replacement is not None:
            rows.append(replacement)
    output = pd.DataFrame(rows)
    if output["conversation_id"].duplicated().any():
        raise ValueError("Expected at most one eligible target per source conversation.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    manifest = {
        "release": str(args.release),
        "output": str(args.output),
        "eligible_targets": len(output),
        "conversations": int(output["conversation_id"].nunique()),
        "replacement_model": "gpt-5.2-2025-12-11",
        "max_prior_messages": args.max_prior_messages,
        "information_constraint": (
            f"Each replacement sees at most the last {args.max_prior_messages} source messages "
            "through the user turn immediately before the replaced assistant reply; the later "
            "flagged target is hidden."
        ),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
