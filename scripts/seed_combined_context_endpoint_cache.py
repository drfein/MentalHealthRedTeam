from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ENDPOINT_PATTERN = re.compile(r":prior_(?:0|all)$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_by_key(
    rows: list[dict[str, Any]], key: str, *, require_success: bool
) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not ENDPOINT_PATTERN.search(str(row.get("message_hash", ""))):
            continue
        if require_success and (
            row.get("error_type") or not str(row.get("response_text") or "").strip()
        ):
            continue
        latest[str(row[key])] = row
    return list(latest.values())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed combined endpoint caches from the completed six-arm run."
    )
    parser.add_argument(
        "--source-responses",
        type=Path,
        default=Path("data/context_dose_response/responses.jsonl"),
    )
    parser.add_argument(
        "--source-judgments",
        type=Path,
        default=Path("results/context_dose_response/package_judgments.jsonl"),
    )
    parser.add_argument(
        "--output-responses",
        type=Path,
        default=Path("data/combined_context_endpoints/responses.jsonl"),
    )
    parser.add_argument(
        "--output-judgments",
        type=Path,
        default=Path("results/combined_context_endpoints/package_judgments.jsonl"),
    )
    args = parser.parse_args()

    responses = latest_by_key(
        read_jsonl(args.source_responses), "generation_id", require_success=True
    )
    judgments = latest_by_key(
        read_jsonl(args.source_judgments), "generation_id", require_success=False
    )
    judgments = [row for row in judgments if not row.get("judge_error")]
    write_jsonl(args.output_responses, responses)
    write_jsonl(args.output_judgments, judgments)
    summary = {
        "seeded_responses": len(responses),
        "seeded_judgments": len(judgments),
        "source_responses": str(args.source_responses),
        "source_judgments": str(args.source_judgments),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
