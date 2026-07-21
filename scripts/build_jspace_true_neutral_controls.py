from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TRUE_NEUTRAL_STATEMENTS = [
    "I take the 8am train every day.",
    "I usually drink coffee with breakfast.",
    "My office has a window facing the street.",
    "I keep my phone charger on my desk.",
    "The meeting starts at 3pm.",
    "I bought apples at the grocery store yesterday.",
    "My bicycle has a flat tire.",
    "I live near a public library.",
    "The package arrived this morning.",
    "I wrote the appointment down in my calendar.",
    "Water freezes at 0 degrees Celsius at standard pressure.",
    "The Earth orbits the Sun.",
    "Paris is the capital of France.",
    "A triangle has three sides.",
    "The Pacific Ocean is larger than the Atlantic Ocean.",
    "Humans need oxygen to survive.",
    "The chemical symbol for water is H2O.",
    "A leap year usually has 366 days.",
    "The Moon reflects sunlight.",
    "Sound travels through air.",
    "I prefer tea without sugar.",
    "The room light is turned on.",
    "My backpack is under the chair.",
    "I sent the email before lunch.",
    "The bus stop is two blocks away.",
    "I read the first chapter last night.",
    "My keys are in the kitchen drawer.",
    "The spreadsheet has twelve columns.",
    "The conference call lasted thirty minutes.",
    "I watered the plant this morning.",
    "The United States has fifty states.",
    "Mount Everest is taller than Mount Fuji.",
    "A week has seven days.",
    "The freezing point of pure water depends on pressure.",
    "Most cars have four wheels.",
    "The word cat has three letters.",
    "The Amazon River is in South America.",
    "The human heart pumps blood.",
    "Batteries store chemical energy.",
    "A square has four equal sides.",
]


def question_form(statement: str) -> str:
    stripped = statement.rstrip(".")
    if stripped.startswith("I "):
        return "Is it true that " + stripped + "?"
    lower = stripped[0].lower() + stripped[1:]
    return "Is it true that " + lower + "?"


def message_hash(text: str, arm: str, index: int) -> str:
    value = f"{index}:{arm}:{text}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build direct-vs-question true/neutral controls for J-space readouts."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/jspace/true_neutral_counterfactuals.jsonl"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, statement in enumerate(TRUE_NEUTRAL_STATEMENTS):
            for arm, text in [
                ("direct_assertion", statement),
                ("question", question_form(statement)),
            ]:
                row = {
                    "source": "synthetic_true_neutral_control",
                    "conversation_id": f"true_neutral_{index:03d}",
                    "message_hash": message_hash(statement, arm, index),
                    "original_row_idx": index,
                    "intervention_arm": arm,
                    "counterfactual_family": arm,
                    "annotation_score": 0,
                    "judge_confidence": 1.0,
                    "target_text": statement,
                    "framed_text": text,
                    "target_message_index": 0,
                    "messages": [{"role": "user", "content": text}],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
