from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare generated discovery-route responses for exact package judging."
    )
    parser.add_argument(
        "--release", type=Path, default=Path("data/discovery_route_benchmark/historical.parquet")
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("data/discovery_route_benchmark/historical_responses.jsonl"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/discovery_route_benchmark/historical")
    )
    args = parser.parse_args()

    release = pd.read_parquet(args.release).reset_index(names="original_row_idx")
    release_by_hash = release.set_index("message_hash").to_dict(orient="index")
    attempts = read_jsonl(args.responses)
    by_generation: dict[str, list[dict[str, Any]]] = {}
    for row in attempts:
        by_generation.setdefault(str(row["generation_id"]), []).append(row)

    responses = []
    prompts = []
    errors = 0
    for generation_attempts in by_generation.values():
        successes = [
            row
            for row in generation_attempts
            if not row.get("error_type") and str(row.get("response_text") or "").strip()
        ]
        if not successes:
            errors += 1
            continue
        response = successes[-1]
        metadata = release_by_hash[str(response["message_hash"])]
        messages = [
            message
            for message in list(response["input_messages"])
            if str(message.get("role", "")).lower() != "system"
        ]
        base = {
            "generation_id": response["generation_id"],
            "original_row_idx": int(metadata["original_row_idx"]),
            "intervention_arm": response["model"],
            "model_id": response["model"],
            "source": metadata["source"],
            "conversation_id": metadata["conversation_id"],
            "message_hash": response["message_hash"],
            "target_text": metadata["target_text"],
            "response": response["response_text"],
        }
        responses.append(base)
        prompts.append({**base, "messages": messages})

    if len({row["generation_id"] for row in responses}) != len(responses):
        raise ValueError("Prepared discovery-route responses contain duplicate IDs.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("responses.jsonl", responses), ("prompts.jsonl", prompts)]:
        with (args.out_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "release": str(args.release),
        "response_attempts": len(attempts),
        "unique_generation_ids": len(by_generation),
        "prepared_responses": len(responses),
        "generation_errors": errors,
        "models": sorted({row["model_id"] for row in responses}),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
