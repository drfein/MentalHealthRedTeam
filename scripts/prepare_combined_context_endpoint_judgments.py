from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ARM_PATTERN = re.compile(r":(prior_(?:0|all))$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_successes(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("error_type") or not str(row.get("response_text") or "").strip():
            continue
        latest[str(row["generation_id"])] = row
    return list(latest.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare blinded endpoint judgments for the combined release."
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("data/releases/WildDelusionCombined/train.parquet"),
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("data/combined_context_endpoints/responses.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/combined_context_endpoints/judge_inputs.jsonl"),
    )
    parser.add_argument("--shuffle-seed", type=int, default=20260723)
    args = parser.parse_args()

    release = pd.read_parquet(args.release).reset_index(names="original_row_idx")
    release_by_hash = {
        str(row.message_hash): row for row in release.itertuples(index=False)
    }
    rows = []
    for response in latest_successes(args.responses):
        match = ARM_PATTERN.search(str(response["message_hash"]))
        if match is None:
            raise ValueError(f"Unexpected endpoint hash: {response['message_hash']}")
        original_hash = str(response["message_hash"])[: match.start()]
        source = release_by_hash.get(original_hash)
        if source is None:
            raise ValueError(f"Combined release row not found for {original_hash}")
        rows.append(
            {
                "original_row_idx": int(source.original_row_idx),
                "intervention_arm": match.group(1),
                "model_id": response["model"],
                "generation_id": response["generation_id"],
                "source": source.source,
                "conversation_id": source.conversation_id,
                "discovery_split": source.discovery_split,
                "message_hash": original_hash,
                "target_text": source.target_text,
                "response": response["response_text"],
            }
        )

    output = pd.DataFrame(rows)
    key = ["original_row_idx", "model_id", "intervention_arm"]
    if output.duplicated(key).any():
        raise ValueError("Judge inputs must be unique by target, model, and arm.")
    order = np.random.default_rng(args.shuffle_seed).permutation(len(output))
    output = output.iloc[order]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "judge_inputs": len(output),
                "targets": int(output["original_row_idx"].nunique()),
                "models": sorted(output["model_id"].unique()),
                "arms": sorted(output["intervention_arm"].unique()),
                "judge_visible_context": "Target and generated reply only.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
