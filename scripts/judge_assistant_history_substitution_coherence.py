from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI


SYSTEM_PROMPT = """You assess counterfactual conversational coherence.
Determine whether a fixed subsequent user message would remain a natural and
semantically coherent continuation if a candidate assistant reply replaced the
original reply. Judge coherence only. Do not reward or penalize factuality,
safety, politeness, or whether the candidate challenges the user's beliefs.

Return JSON with:
- score: integer 0-10, where 0 is incoherent, 5 is partly coherent but depends
  on missing or contradicted details, 7 is substantially coherent, and 10 is a
  fully natural continuation;
- rationale: one concise sentence."""


def build_prompt(row: Any) -> str:
    return (
        "PRECEDING USER MESSAGE:\n"
        f"{row.preceding_user_text}\n\n"
        "ORIGINAL ASSISTANT REPLY:\n"
        f"{row.original_assistant_text}\n\n"
        "CANDIDATE REPLACEMENT ASSISTANT REPLY:\n"
        f"{row.replacement_assistant_text}\n\n"
        "FIXED SUBSEQUENT USER MESSAGE:\n"
        f"{row.target_text}"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Judge coherence after assistant-turn substitution.")
    parser.add_argument(
        "--interventions",
        type=Path,
        default=Path("data/assistant_history_substitution/intervention_arms.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/assistant_history_substitution/coherence_judgments.jsonl"),
    )
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    frame = pd.read_parquet(args.interventions)
    frame = frame[frame["context_arm"] == "gpt52_substituted"].copy()
    frame["preceding_user_text"] = frame["messages"].map(
        lambda messages: list(messages)[-3]["content"]
    )
    rows = list(frame.itertuples(index=False))
    completed = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            completed = {
                int(json.loads(line)["original_row_idx"])
                for line in handle
                if line.strip()
            }
        rows = [row for row in rows if int(row.original_row_idx) not in completed]

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)

    async def judge(row: Any) -> dict[str, Any]:
        for attempt in range(5):
            try:
                async with semaphore:
                    response = await client.responses.create(
                        model=args.model,
                        input=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": build_prompt(row)},
                        ],
                        reasoning={"effort": args.reasoning_effort},
                        text={"format": {"type": "json_object"}},
                        max_output_tokens=180,
                    )
                label = json.loads(response.output_text)
                return {
                    "original_row_idx": int(row.original_row_idx),
                    "conversation_id": row.conversation_id,
                    "replacement_generation_id": row.replacement_generation_id,
                    "coherence_score": int(label["score"]),
                    "coherence_rationale": str(label["rationale"]),
                    "judge_model": args.model,
                    "judge_reasoning_effort": args.reasoning_effort,
                    "judge_input_tokens": response.usage.input_tokens,
                    "judge_output_tokens": response.usage.output_tokens,
                    "prompt_version": "assistant_history_substitution_coherence_v1",
                }
            except Exception as error:  # noqa: BLE001 - errors are persisted for resumability.
                if attempt == 4:
                    return {
                        "original_row_idx": int(row.original_row_idx),
                        "conversation_id": row.conversation_id,
                        "replacement_generation_id": row.replacement_generation_id,
                        "judge_error": repr(error),
                    }
                await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    judged = await asyncio.gather(*(judge(row) for row in rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for row in judged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(judged)} coherence judgments to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
