from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from wild_delusion_miner.assistant_responses import GenerationParams, generation_id


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report cached and pending combined context endpoint work."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/combined_context_endpoints.json"),
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path("data/combined_context_endpoints/inputs.parquet"),
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("data/combined_context_endpoints/responses.jsonl"),
    )
    parser.add_argument(
        "--judgments",
        type=Path,
        default=Path("results/combined_context_endpoints/package_judgments.jsonl"),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    models = list(config["generation"]["models"])
    inputs = pd.read_parquet(args.inputs).to_dict(orient="records")
    params = GenerationParams()
    successful_ids = {
        str(row["generation_id"])
        for row in read_jsonl(args.responses)
        if not row.get("error_type") and str(row.get("response_text") or "").strip()
    }
    judged_ids = {
        str(row["generation_id"])
        for row in read_jsonl(args.judgments)
        if not row.get("judge_error")
    }

    pending = []
    requested_ids = set()
    for row in inputs:
        for model in models:
            identifier = generation_id(row, model, params.prompt_version)
            requested_ids.add(identifier)
            if identifier not in successful_ids:
                pending.append(
                    {
                        "model": model,
                        "discovery_split": str(row["discovery_split"]),
                        "context_arm": str(row["context_arm"]),
                    }
                )

    def count(field: str) -> dict[str, int]:
        return dict(sorted(Counter(row[field] for row in pending).items()))

    summary = {
        "input_targets": len(inputs) // 2,
        "input_rows": len(inputs),
        "models": len(models),
        "requested_generations": len(requested_ids),
        "cached_successes": len(requested_ids & successful_ids),
        "pending_generations": len(pending),
        "cached_judgments": len(requested_ids & judged_ids),
        "pending_by_model": count("model"),
        "pending_by_discovery_split": count("discovery_split"),
        "pending_by_context_arm": count("context_arm"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
