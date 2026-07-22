from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from llm_delusions_annotations.annotation_prompts import (
    ANNOTATION_SYSTEM_PROMPT,
    build_prompt,
    load_annotations,
)
from openai import AsyncOpenAI


ANNOTATION_ID = "user-endorses-delusion"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score user turns with the pinned user-endorses-delusion package prompt."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    prompts = {row["generation_id"]: row for row in read_jsonl(args.prompts)}
    prior: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        prior = read_jsonl(args.output)
        if args.retry_errors:
            prior = [row for row in prior if "judge_error" not in row]
            with args.output.open("w", encoding="utf-8") as handle:
                for row in prior:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        completed = {row["generation_id"] for row in prior}
        rows = [row for row in rows if row["generation_id"] not in completed]

    annotation = next(item for item in load_annotations() if item["id"] == ANNOTATION_ID)
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)

    async def judge(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        package_prompt = build_prompt(
            annotation,
            row["user_text"],
            role="user",
            context_messages=prompts[row["generation_id"]]["messages"],
        )
        for attempt in range(5):
            try:
                async with semaphore:
                    response = await client.responses.create(
                        model=args.model,
                        input=[
                            {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                            {"role": "user", "content": package_prompt},
                        ],
                        reasoning={"effort": args.reasoning_effort},
                        text={"format": {"type": "json_object"}},
                        max_output_tokens=args.max_output_tokens,
                    )
                label = json.loads(response.output_text)
                return index, {
                    **row,
                    "package_annotation_id": ANNOTATION_ID,
                    "judge_model": args.model,
                    "judge_reasoning_effort": args.reasoning_effort,
                    "annotation_score": int(label["score"]),
                    "annotation_rationale": label["rationale"],
                    "annotation_quotes": label["quotes"],
                    "judge_input_tokens": response.usage.input_tokens,
                    "judge_output_tokens": response.usage.output_tokens,
                }
            except Exception as error:
                if attempt == 4:
                    return index, {**row, "judge_error": repr(error)}
                await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    judged = await asyncio.gather(*(judge(index, row) for index, row in enumerate(rows)))
    judged.sort(key=lambda item: item[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for _, row in judged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    errors = sum("judge_error" in row for _, row in judged)
    print(f"Wrote {len(judged)} rows ({errors} errors) to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
