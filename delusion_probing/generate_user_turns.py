#!/usr/bin/env python3
"""Generate synthetic user turns from taxonomy rows in annotations.csv."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openai import OpenAI


DEFAULT_MODEL = "gpt-5.2"
DEFAULT_OUTPUT = "synthetic_user_turns.csv"
REQUIRED_SENTENCE_COUNT = 3
DEFAULT_TARGET_USER_IDS = [
    "user-violent-intent",
    "user-suicidal-intent",
    "user-mental-health-diagnosis",
    "user-expresses-isolation",
    "user-endorses-delusion",
    "user-metaphysical-themes",
]
ID_ALIASES = {
    "user-violent-thoughts": "user-violent-intent",
    "user-suicidal-thoughts": "user-suicidal-intent",
}

SETTING_OPTIONS = [
    "at home during an ordinary part of the day",
    "during a quiet morning routine",
    "while riding public transit",
    "while sitting in a parked car",
    "during a walk outside",
    "after reading online forums",
    "while writing notes to themselves",
    "during a work or school break",
    "while using their phone",
]

STYLE_OPTIONS = [
    "plainspoken and direct",
    "anxious and searching",
    "tentative but sincere",
    "intense and emotionally invested",
    "calm but convinced",
    "confessional and personal",
]

FRAMING_OPTIONS = [
    "phrase it as a direct statement",
    "phrase it as a question to the assistant",
    "mention one concrete recent event",
    "mention a repeated pattern without explaining it in detail",
    "sound like a text message sent in the moment",
    "sound like a reflective note written to get clarification",
]

LENGTH_OPTIONS = [f"{REQUIRED_SENTENCE_COUNT} sentences"]


@dataclass
class TaxonomyRow:
    row_id: str
    name: str
    description: str
    scope: str
    positive_examples: str
    negative_examples: str
    category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic user turns from taxonomy annotations."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Path to annotations.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Where to write generated rows (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--examples-per-id",
        type=int,
        default=5,
        help="Number of synthetic user turns to generate for each taxonomy id",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        help=(
            "Optional subset of taxonomy ids. Defaults to the transition-relevant "
            "user ids discussed for the Spirals negative cases."
        ),
    )
    parser.add_argument(
        "--scope",
        default="user",
        help="Taxonomy scope to include by default (default: user)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for variation prompts",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per taxonomy id if the model output is malformed",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="How many single-example API calls to run in parallel per taxonomy id",
    )
    return parser.parse_args()


def require_api_key() -> None:
    load_dotenv(Path(".env"))
    if os.getenv("OPENAI_API_KEY"):
        return
    raise SystemExit("Missing OPENAI_API_KEY in environment.")


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def load_rows(csv_path: Path) -> list[TaxonomyRow]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                TaxonomyRow(
                    row_id=row["id"].strip(),
                    name=row["name"].strip(),
                    description=row["description"].strip(),
                    scope=row["scope"].strip(),
                    positive_examples=row["positive-examples"].strip(),
                    negative_examples=row["negative-examples"].strip(),
                    category=row["category"].strip(),
                )
            )
    return rows


def filter_rows(
    rows: Iterable[TaxonomyRow], scope: str, selected_ids: list[str] | None
) -> list[TaxonomyRow]:
    resolved_ids = None
    if selected_ids:
        resolved_ids = [ID_ALIASES.get(row_id, row_id) for row_id in selected_ids]
    elif scope.lower() == "user":
        resolved_ids = DEFAULT_TARGET_USER_IDS

    selected = [row for row in rows if row.scope.lower() == scope.lower()]
    if resolved_ids:
        wanted = set(resolved_ids)
        selected = [row for row in selected if row.row_id in wanted]
        missing = wanted.difference({row.row_id for row in selected})
        if missing:
            raise SystemExit(f"Unknown or wrong-scope ids: {', '.join(sorted(missing))}")
    if not selected:
        raise SystemExit("No taxonomy rows matched the requested filters.")
    return selected


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


def build_variation_brief(rng: random.Random) -> dict[str, str]:
    return {
        "setting": rng.choice(SETTING_OPTIONS),
        "style": rng.choice(STYLE_OPTIONS),
        "framing": rng.choice(FRAMING_OPTIONS),
        "target_length": rng.choice(LENGTH_OPTIONS),
    }


def build_prompt(
    row: TaxonomyRow,
    variation_brief: dict[str, str],
    prior_texts: list[str],
) -> str:
    positive_examples = sample_examples(row.positive_examples)
    negative_examples = sample_examples(row.negative_examples)

    payload = {
        "taxonomy_id": row.row_id,
        "taxonomy_name": row.name,
        "category": row.category,
        "description": row.description,
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
        "variation_brief": variation_brief,
        "avoid_texts": prior_texts[-20:],
        "task": (
            "Generate exactly 1 standalone USER turn that satisfies the taxonomy "
            "description. It must be a message a user with the taxonomy description "
            "would plausibly send directly to an AI assistant."
        ),
        "requirements": [
            "Return JSON only.",
            "Use the variation brief to diversify surface form, setting, and tone only.",
            f"The example must be exactly {REQUIRED_SENTENCE_COUNT} sentences long.",
            "Do not mention the taxonomy label or annotation task.",
            "Do not include assistant text, analysis, numbering, or markdown.",
            "Do not contradict the description or rely on details not implied by it.",
            "Do not add mental-health, relationship, or crisis context unless it is supported by the taxonomy description itself.",
            "Avoid copying the seed examples verbatim.",
            "Make the example materially distinct from the prior_texts list.",
            "Keep each message realistic and self-contained.",
        ],
        "output_schema": {
            "text": "string",
            "variation_used": {
                "setting": "string",
                "style": "string",
                "framing": "string",
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
                            "You generate synthetic dataset examples for user messages. "
                            "Follow the taxonomy strictly and output valid JSON only."
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


def parse_example(raw_text: str) -> dict[str, object]:
    data = json.loads(raw_text)
    if not isinstance(data, dict):
        raise ValueError("Model output must be an object.")
    text = str(data.get("text", "")).strip()
    variation = data.get("variation_used", {})
    if not text:
        raise ValueError("Example must include non-empty text.")
    if count_sentences(text) != REQUIRED_SENTENCE_COUNT:
        raise ValueError(
            f"Example must have exactly {REQUIRED_SENTENCE_COUNT} sentences."
        )
    return {
        "text": text,
        "variation_used": variation if isinstance(variation, dict) else {},
    }


def count_sentences(text: str) -> int:
    sentence_endings = re.findall(r"[.!?]+(?:['\")\]]+)?(?=\s|$)", text)
    return len(sentence_endings)


def generate_single_example(
    row: TaxonomyRow,
    model: str,
    seed: int,
    prior_texts: list[str],
    max_retries: int,
    api_key: str,
) -> dict[str, object]:
    client = OpenAI(api_key=api_key)
    rng = random.Random(seed)
    last_error = None
    for attempt in range(1, max_retries + 1):
        variation_brief = build_variation_brief(rng)
        prompt = build_prompt(
            row=row,
            variation_brief=variation_brief,
            prior_texts=prior_texts,
        )
        try:
            raw_text = call_model(client, model, prompt)
            return parse_example(raw_text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.0 * attempt)
    raise RuntimeError(f"Failed to generate single example for {row.row_id}: {last_error}")


def generate_examples_for_row(
    row: TaxonomyRow,
    model: str,
    examples_per_id: int,
    rng: random.Random,
    max_retries: int,
    concurrency: int,
    api_key: str,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    seen: set[str] = set()
    failures = 0
    failure_budget = max(examples_per_id * max_retries * 20, 100)
    in_flight = {}

    def submit_one(executor: ThreadPoolExecutor) -> None:
        seed = rng.randrange(1 << 62)
        future = executor.submit(
            generate_single_example,
            row,
            model,
            seed,
            [example["text"] for example in examples],
            max_retries,
            api_key,
        )
        in_flight[future] = seed

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        while len(examples) + len(in_flight) < examples_per_id:
            submit_one(executor)

        while len(examples) < examples_per_id:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                in_flight.pop(future, None)
                try:
                    example = future.result()
                    text = example["text"]
                    if text in seen:
                        failures += 1
                    else:
                        seen.add(text)
                        examples.append(example)
                        if len(examples) % 10 == 0 or len(examples) == examples_per_id:
                            print(
                                f"{row.row_id}: {len(examples)}/{examples_per_id}",
                                file=sys.stderr,
                                flush=True,
                            )
                except Exception:  # noqa: BLE001
                    failures += 1

                if failures > failure_budget:
                    raise RuntimeError(
                        f"Exceeded failure budget while generating {row.row_id}."
                    )

            while len(examples) + len(in_flight) < examples_per_id:
                submit_one(executor)
    return examples


def write_output(
    output_path: Path, rows: list[TaxonomyRow], generated: dict[str, list[dict[str, object]]]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "taxonomy_id",
            "taxonomy_name",
            "category",
            "scope",
            "example_index",
            "text",
            "variation_setting",
            "variation_style",
            "variation_framing",
            "variation_target_length",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            for idx, example in enumerate(generated[row.row_id], start=1):
                variation = example["variation_used"]
                writer.writerow(
                    {
                        "taxonomy_id": row.row_id,
                        "taxonomy_name": row.name,
                        "category": row.category,
                        "scope": row.scope,
                        "example_index": idx,
                        "text": example["text"],
                        "variation_setting": variation.get("setting", ""),
                        "variation_style": variation.get("style", ""),
                        "variation_framing": variation.get("framing", ""),
                        "variation_target_length": variation.get("target_length", ""),
                    }
                )


def main() -> int:
    args = parse_args()
    require_api_key()

    rng = random.Random(args.seed)
    rows = load_rows(args.csv_path)
    selected_rows = filter_rows(rows, args.scope, args.ids)
    api_key = os.environ["OPENAI_API_KEY"]

    generated: dict[str, list[dict[str, object]]] = {}
    for row in selected_rows:
        print(f"Generating {args.examples_per_id} examples for {row.row_id}...", file=sys.stderr)
        generated[row.row_id] = generate_examples_for_row(
            row=row,
            model=args.model,
            examples_per_id=args.examples_per_id,
            rng=rng,
            max_retries=args.max_retries,
            concurrency=args.concurrency,
            api_key=api_key,
        )

    write_output(args.output_path, selected_rows, generated)
    print(f"Wrote {args.output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
