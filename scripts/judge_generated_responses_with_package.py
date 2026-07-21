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


ANNOTATION_ID = "bot-endorses-delusion"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_key(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["original_row_idx"]), str(
        row.get("condition", row.get("intervention_arm", ""))
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge generated responses with llm-delusions-annotations."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--selected-original-indices", type=Path, default=None)
    parser.add_argument(
        "--intervention-arm",
        action="append",
        default=None,
        help="Restrict judging to one or more named intervention arms.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    source_prompts = None
    if args.prompts is not None:
        source_prompts = {row_key(row): row for row in read_jsonl(args.prompts)}
    if args.selected_original_indices is not None:
        selected = {
            int(value)
            for value in args.selected_original_indices.read_text(encoding="utf-8").splitlines()
            if value.strip()
        }
        rows = [row for row in rows if int(row["original_row_idx"]) in selected]
    if args.intervention_arm is not None:
        selected_arms = set(args.intervention_arm)
        rows = [row for row in rows if row.get("intervention_arm") in selected_arms]
    if args.resume and args.output.exists():
        completed = {row_key(row) for row in read_jsonl(args.output)}
        rows = [row for row in rows if row_key(row) not in completed]
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    annotation = next(item for item in load_annotations() if item["id"] == ANNOTATION_ID)
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)

    async def judge(row_idx: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        context_messages = [{"role": "user", "content": row["target_text"]}]
        if source_prompts is not None:
            context_messages = source_prompts[row_key(row)]["messages"]
        prompt = build_prompt(
            annotation,
            row["response"],
            role="assistant",
            context_messages=context_messages,
        )
        for attempt in range(5):
            try:
                async with semaphore:
                    response = await client.responses.create(
                        model=args.model,
                        input=[
                            {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        reasoning={"effort": args.reasoning_effort},
                        text={"format": {"type": "json_object"}},
                        max_output_tokens=350,
                    )
                label = json.loads(response.output_text)
                return row_idx, {
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
                    return row_idx, {**row, "judge_error": repr(error)}
                await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    judged = await asyncio.gather(*(judge(idx, row) for idx, row in enumerate(rows)))
    judged.sort(key=lambda item: item[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for _, row in judged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    errors = sum("judge_error" in row for _, row in judged)
    print(f"Wrote {len(judged)} rows ({errors} errors) to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
