from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from judge_semantic_counterfactual_responses import RUBRIC, parse, row_key


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the framing-aware counterfactual judge with an OpenAI model."
    )
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--selected-original-indices", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    prompts = {row_key(row): row for row in read_jsonl(args.prompts)}
    rows = read_jsonl(args.generations)
    if args.selected_original_indices is not None:
        selected = {
            int(value)
            for value in args.selected_original_indices.read_text(encoding="utf-8").splitlines()
            if value.strip()
        }
        rows = [row for row in rows if int(row["original_row_idx"]) in selected]
    if args.resume and args.output.exists():
        completed = {row_key(row) for row in read_jsonl(args.output)}
        rows = [row for row in rows if row_key(row) not in completed]
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)

    async def judge(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        source = prompts[row_key(row)]
        payload = {
            "framing": row["intervention_arm"],
            "user_message": source["framed_text"],
            "assistant_response": row["response"],
        }
        for attempt in range(5):
            try:
                async with semaphore:
                    response = await client.responses.create(
                        model=args.model,
                        input=[
                            {"role": "system", "content": RUBRIC},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                        ],
                        reasoning={"effort": args.reasoning_effort},
                        text={"format": {"type": "json_object"}},
                        max_output_tokens=250,
                    )
                label = parse(response.output_text)
                return index, {
                    **row,
                    "judge_model": args.model,
                    "judge_reasoning_effort": args.reasoning_effort,
                    "raw_judge_output": response.output_text,
                    "reality_endorsement_score": int(label["reality_endorsement_score"]),
                    "frame_adherence_score": int(label["frame_adherence_score"]),
                    "behavior_label": str(label["behavior_label"]),
                    "judge_confidence": float(label.get("confidence", 0.0)),
                    "judge_rationale": str(label.get("rationale", "")),
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
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for _, row in judged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    errors = sum("judge_error" in row for _, row in judged)
    print(f"Wrote {len(judged)} rows ({errors} errors) to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
