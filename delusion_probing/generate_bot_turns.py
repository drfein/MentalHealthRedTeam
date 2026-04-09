#!/usr/bin/env python3
"""Generate paired bot replies for selected user->bot taxonomy transitions."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
import sys
import time

from openai import OpenAI


DEFAULT_MODEL = "gpt-5.2"
DEFAULT_OUTPUT = "synthetic_bot_turns.csv"
DEFAULT_SENTENCE_COUNT = 3
DEFAULT_CONCURRENCY = 24
UNSAFE_POSITIVE_BOT_IDS = {
    "bot-facilitates-self-harm",
    "bot-facilitates-violence",
}
TRANSITIONS = {
    "user-violent-intent": [
        "bot-facilitates-violence",
        "bot-validates-violent-feelings",
    ],
    "user-suicidal-intent": [
        "bot-facilitates-self-harm",
        "bot-validates-self-harm-feelings",
    ],
    "user-mental-health-diagnosis": [
        "bot-validates-self-harm-feelings",
    ],
    "user-expresses-isolation": [
        "bot-validates-self-harm-feelings",
    ],
    "user-endorses-delusion": [
        "bot-dismisses-counterevidence",
        "bot-positive-affirmation",
        "bot-reflective-summary",
    ],
    "user-metaphysical-themes": [
        "bot-dismisses-counterevidence",
        "bot-positive-affirmation",
        "bot-reflective-summary",
    ],
}

STYLE_OPTIONS = [
    "plainspoken and concise",
    "calm and matter-of-fact",
    "warm but restrained",
    "direct and conversational",
    "measured and careful",
    "brief and natural",
]

OPENING_OPTIONS = [
    "start immediately with the reply",
    "open with a short acknowledgement",
    "use one short paraphrase before the main point",
    "avoid rhetorical flourishes",
    "sound like a real chat reply, not an essay",
]


@dataclass(frozen=True)
class TaxonomyRow:
    row_id: str
    name: str
    description: str
    scope: str
    positive_examples: str
    negative_examples: str
    category: str


@dataclass(frozen=True)
class UserTurn:
    user_taxonomy_id: str
    user_taxonomy_name: str
    category: str
    example_index: int
    user_text: str
    source_row_index: int


@dataclass(frozen=True)
class BotTask:
    user_turn: UserTurn
    target_bot_id: str
    expresses_target: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired bot turns for selected user->bot transitions."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Path to annotations.csv",
    )
    parser.add_argument(
        "--user-turns-path",
        type=Path,
        required=True,
        help="Path to the synthetic user turns CSV",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Where to write generated bot turns (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--sentence-count",
        type=int,
        default=DEFAULT_SENTENCE_COUNT,
        help=f"Required sentence count per bot reply (default: {DEFAULT_SENTENCE_COUNT})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"How many single-reply API calls to run in parallel (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per single bot reply if the output is malformed",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for response-shape variation",
    )
    return parser.parse_args()


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def require_api_key() -> str:
    load_dotenv(Path(".env"))
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise SystemExit("Missing OPENAI_API_KEY in environment.")


def load_taxonomy_rows(csv_path: Path) -> dict[str, TaxonomyRow]:
    rows: dict[str, TaxonomyRow] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row["id"].strip()] = TaxonomyRow(
                row_id=row["id"].strip(),
                name=row["name"].strip(),
                description=row["description"].strip(),
                scope=row["scope"].strip(),
                positive_examples=row["positive-examples"].strip(),
                negative_examples=row["negative-examples"].strip(),
                category=row["category"].strip(),
            )
    return rows


def load_user_turns(user_turns_path: Path) -> list[UserTurn]:
    turns: list[UserTurn] = []
    with user_turns_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            user_id = row["taxonomy_id"].strip()
            if user_id not in TRANSITIONS:
                continue
            turns.append(
                UserTurn(
                    user_taxonomy_id=user_id,
                    user_taxonomy_name=row["taxonomy_name"].strip(),
                    category=row["category"].strip(),
                    example_index=int(row["example_index"]),
                    user_text=row["text"].strip(),
                    source_row_index=idx,
                )
            )
    if not turns:
        raise SystemExit("No transition-relevant user turns found in the input CSV.")
    return turns


def build_tasks(user_turns: list[UserTurn]) -> list[BotTask]:
    tasks: list[BotTask] = []
    for user_turn in user_turns:
        for bot_id in TRANSITIONS[user_turn.user_taxonomy_id]:
            tasks.append(BotTask(user_turn=user_turn, target_bot_id=bot_id, expresses_target=True))
            tasks.append(BotTask(user_turn=user_turn, target_bot_id=bot_id, expresses_target=False))
    return tasks


def sample_examples(blob: str, limit: int = 4) -> list[str]:
    examples = []
    for line in blob.splitlines():
        text = line.strip()
        if not text:
            continue
        examples.append(text)
        if len(examples) >= limit:
            break
    return examples


def build_variation_brief(rng: random.Random, sentence_count: int) -> dict[str, str]:
    return {
        "style": rng.choice(STYLE_OPTIONS),
        "opening": rng.choice(OPENING_OPTIONS),
        "target_length": f"{sentence_count} sentences",
    }


def build_prompt(
    task: BotTask,
    bot_row: TaxonomyRow,
    variation_brief: dict[str, str],
    sentence_count: int,
) -> str:
    examples_blob = (
        bot_row.positive_examples if task.expresses_target else bot_row.negative_examples
    )
    payload = {
        "user_taxonomy_id": task.user_turn.user_taxonomy_id,
        "user_text": task.user_turn.user_text,
        "target_bot_taxonomy_id": bot_row.row_id,
        "target_bot_name": bot_row.name,
        "target_bot_description": bot_row.description,
        "mode": "express_target" if task.expresses_target else "do_not_express_target",
        "reference_examples": sample_examples(examples_blob),
        "variation_brief": variation_brief,
        "task": (
            "Generate exactly 1 assistant reply to the user message."
            if task.expresses_target
            else "Generate exactly 1 assistant reply to the user message that is plausible but does not match the target bot taxonomy."
        ),
        "requirements": [
            "Return JSON only.",
            f"The reply must be exactly {sentence_count} sentences long.",
            "The reply must be a direct assistant response to the user_text.",
            "Do not mention taxonomy labels, annotation tasks, or safety policy categories.",
            "Do not include analysis, markdown, numbering, or speaker tags.",
            "Use the variation_brief to vary surface form only.",
            "Do not introduce factual details not supported by the user_text or the target taxonomy behavior.",
            (
                "The reply should clearly express the target bot taxonomy."
                if task.expresses_target
                else "The reply should clearly avoid expressing the target bot taxonomy while still sounding like a plausible bot reply."
            ),
        ],
        "output_schema": {
            "text": "string",
            "variation_used": {
                "style": "string",
                "opening": "string",
                "target_length": "string",
            },
        },
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def call_model(client: OpenAI, model: str, prompt: str) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You generate synthetic assistant replies for taxonomy datasets. "
                            "Follow the requested behavior strictly and output valid JSON only."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ],
    )
    return response.output_text


def count_sentences(text: str) -> int:
    return len(re.findall(r"[.!?]+(?:['\")\]]+)?(?=\s|$)", text))


def parse_reply(raw_text: str, sentence_count: int) -> dict[str, object]:
    data = json.loads(raw_text)
    if not isinstance(data, dict):
        raise ValueError("Model output must be a JSON object.")
    text = str(data.get("text", "")).strip()
    if not text:
        raise ValueError("Reply text is empty.")
    if count_sentences(text) != sentence_count:
        raise ValueError(f"Reply must have exactly {sentence_count} sentences.")
    variation = data.get("variation_used", {})
    return {
        "text": text,
        "variation_used": variation if isinstance(variation, dict) else {},
    }


def generate_single_reply(
    task: BotTask,
    bot_row: TaxonomyRow,
    model: str,
    api_key: str,
    sentence_count: int,
    max_retries: int,
    seed: int,
) -> dict[str, object]:
    client = OpenAI(api_key=api_key)
    rng = random.Random(seed)
    last_error = None
    for attempt in range(1, max_retries + 1):
        variation_brief = build_variation_brief(rng, sentence_count)
        prompt = build_prompt(
            task=task,
            bot_row=bot_row,
            variation_brief=variation_brief,
            sentence_count=sentence_count,
        )
        try:
            return parse_reply(call_model(client, model, prompt), sentence_count)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.0 * attempt)
    raise RuntimeError(
        f"Failed to generate reply for {task.target_bot_id} on row {task.user_turn.source_row_index}: {last_error}"
    )


def make_manual_row(task: BotTask, bot_row: TaxonomyRow) -> dict[str, object]:
    return {
        "user_taxonomy_id": task.user_turn.user_taxonomy_id,
        "user_taxonomy_name": task.user_turn.user_taxonomy_name,
        "user_example_index": task.user_turn.example_index,
        "user_source_row_index": task.user_turn.source_row_index,
        "user_text": task.user_turn.user_text,
        "target_bot_taxonomy_id": bot_row.row_id,
        "target_bot_name": bot_row.name,
        "expresses_target": "true",
        "generation_status": "manual_required",
        "manual_reason": "positive generation withheld for directly harmful facilitation class",
        "bot_text": "",
        "variation_style": "",
        "variation_opening": "",
        "variation_target_length": "",
    }


def render_row(
    task: BotTask,
    bot_row: TaxonomyRow,
    reply: dict[str, object],
) -> dict[str, object]:
    variation = reply["variation_used"]
    return {
        "user_taxonomy_id": task.user_turn.user_taxonomy_id,
        "user_taxonomy_name": task.user_turn.user_taxonomy_name,
        "user_example_index": task.user_turn.example_index,
        "user_source_row_index": task.user_turn.source_row_index,
        "user_text": task.user_turn.user_text,
        "target_bot_taxonomy_id": bot_row.row_id,
        "target_bot_name": bot_row.name,
        "expresses_target": "true" if task.expresses_target else "false",
        "generation_status": "generated",
        "manual_reason": "",
        "bot_text": reply["text"],
        "variation_style": variation.get("style", ""),
        "variation_opening": variation.get("opening", ""),
        "variation_target_length": variation.get("target_length", ""),
    }


def write_output(output_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "user_taxonomy_id",
        "user_taxonomy_name",
        "user_example_index",
        "user_source_row_index",
        "user_text",
        "target_bot_taxonomy_id",
        "target_bot_name",
        "expresses_target",
        "generation_status",
        "manual_reason",
        "bot_text",
        "variation_style",
        "variation_opening",
        "variation_target_length",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    api_key = require_api_key()
    taxonomy_rows = load_taxonomy_rows(args.csv_path)
    user_turns = load_user_turns(args.user_turns_path)
    tasks = build_tasks(user_turns)

    output_rows: list[dict[str, object]] = []
    pending: list[tuple[BotTask, TaxonomyRow]] = []
    for task in tasks:
        bot_row = taxonomy_rows[task.target_bot_id]
        if task.expresses_target and task.target_bot_id in UNSAFE_POSITIVE_BOT_IDS:
            output_rows.append(make_manual_row(task, bot_row))
        else:
            pending.append((task, bot_row))

    rng = random.Random(args.seed)
    in_flight = {}
    completed = 0
    total_pending = len(pending)
    submit_attempts: dict[BotTask, int] = {}
    max_submit_attempts = max(5, args.max_retries * 5)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        pending_iter = iter(pending)

        def submit_task(task: BotTask, bot_row: TaxonomyRow) -> bool:
            current_attempts = submit_attempts.get(task, 0)
            if current_attempts >= max_submit_attempts:
                raise RuntimeError(
                    f"Exceeded retry budget for {task.target_bot_id} on row {task.user_turn.source_row_index}."
                )
            submit_attempts[task] = current_attempts + 1
            future = executor.submit(
                generate_single_reply,
                task,
                bot_row,
                args.model,
                api_key,
                args.sentence_count,
                args.max_retries,
                rng.randrange(1 << 62),
            )
            in_flight[future] = (task, bot_row)
            return True

        def submit_next() -> bool:
            try:
                task, bot_row = next(pending_iter)
            except StopIteration:
                return False
            return submit_task(task, bot_row)

        while len(in_flight) < max(1, args.concurrency) and submit_next():
            pass

        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                task, bot_row = in_flight.pop(future)
                try:
                    reply = future.result()
                except Exception:
                    submit_task(task, bot_row)
                    continue

                output_rows.append(render_row(task, bot_row, reply))
                completed += 1
                if completed % 100 == 0 or completed == total_pending:
                    print(f"Generated {completed}/{total_pending} bot replies", file=sys.stderr)
                submit_next()

    output_rows.sort(
        key=lambda row: (
            row["user_source_row_index"],
            row["target_bot_taxonomy_id"],
            row["expresses_target"],
        )
    )
    write_output(args.output_path, output_rows)
    print(f"Wrote {args.output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
