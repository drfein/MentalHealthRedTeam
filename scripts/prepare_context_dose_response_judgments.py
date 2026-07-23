from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ARM_PATTERN = re.compile(r":(prior_(?:0|1|2|4|8|all))$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_successes(path: Path) -> list[dict[str, Any]]:
    attempts: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        attempts.setdefault(str(row["generation_id"]), []).append(row)
    results = []
    for rows in attempts.values():
        successes = [
            row
            for row in rows
            if not row.get("error_type") and str(row.get("response_text") or "").strip()
        ]
        if successes:
            results.append(successes[-1])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare blinded, context-free judgments for the context dose response."
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("data/releases/WildDelusionVerified/train.parquet"),
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("data/context_dose_response/responses.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/context_dose_response/judge_inputs.jsonl"),
    )
    parser.add_argument("--shuffle-seed", type=int, default=20260722)
    args = parser.parse_args()

    release = pd.read_parquet(args.release).reset_index(names="original_row_idx")
    release_by_hash = {
        str(row.message_hash): row for row in release.itertuples(index=False)
    }
    rows = []
    for response in latest_successes(args.responses):
        match = ARM_PATTERN.search(str(response["message_hash"]))
        if match is None:
            raise ValueError(f"Unexpected dose-response hash: {response['message_hash']}")
        arm = match.group(1)
        original_hash = str(response["message_hash"])[: match.start()]
        source = release_by_hash.get(original_hash)
        if source is None:
            raise ValueError(f"Release row not found for {original_hash}")
        rows.append(
            {
                "original_row_idx": int(source.original_row_idx),
                "intervention_arm": arm,
                "model_id": response["model"],
                "generation_id": response["generation_id"],
                "source": source.source,
                "conversation_id": source.conversation_id,
                "message_hash": original_hash,
                "target_text": source.target_text,
                "response": response["response_text"],
            }
        )

    output = pd.DataFrame(rows)
    key = ["original_row_idx", "model_id", "intervention_arm"]
    if output.duplicated(key).any():
        raise ValueError("Judgment inputs are not unique by target, model, and arm.")
    order = np.random.default_rng(args.shuffle_seed).permutation(len(output))
    output = output.iloc[order]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "responses": str(args.responses),
        "output": str(args.output),
        "judge_inputs": len(output),
        "targets": int(output["original_row_idx"].nunique()),
        "models": sorted(output["model_id"].unique()),
        "arms": sorted(output["intervention_arm"].unique()),
        "shuffle_seed": args.shuffle_seed,
        "judge_visible_context": "Target user message and generated assistant reply only.",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
