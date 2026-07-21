#!/usr/bin/env python3
"""Summarize the released WildDelusion rows without exposing message content."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/releases/WildDelusionVerified/train.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/wilddelusion_dataset_characterization"),
    )
    return parser.parse_args()


def load_positive_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    positives = [row for row in rows if row.get("judge_label") == "positive"]
    if not positives:
        raise ValueError(f"No strict verifier-positive rows found in {path}")
    return positives


def target_text(row: dict[str, Any]) -> str:
    messages = row["messages"]
    index = int(row["target_message_index"])
    if not 0 <= index < len(messages):
        raise ValueError(f"Invalid target index {index} for {row.get('stable_key')}")
    return str(messages[index].get("content", ""))


def quantile_summary(series: pd.Series) -> dict[str, float]:
    return {
        "median": float(series.median()),
        "q1": float(series.quantile(0.25)),
        "q3": float(series.quantile(0.75)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def format_iqr(values: dict[str, float], digits: int = 0) -> str:
    template = f"{{:.{digits}f}} [{{:.{digits}f}}, {{:.{digits}f}}]"
    return template.format(values["median"], values["q1"], values["q3"])


def main() -> None:
    args = parse_args()
    rows = load_positive_rows(args.input)
    records = [
        {
            "source": row["source"],
            "conversation_key": json.dumps(
                [row["source"], row["split"], str(row["conversation_id"])]
            ),
            "stored_context_messages": len(row["messages"]),
            "target_message_index": int(row["target_message_index"]),
            "target_words": len(target_text(row).split()),
            "annotation_score": int(row["annotation_score"]),
            "verifier_confidence": float(row["judge_confidence"]),
        }
        for row in rows
    ]
    frame = pd.DataFrame.from_records(records)

    source_counts = Counter(frame["source"])
    conversation_counts = (
        frame.drop_duplicates("conversation_key")["source"].value_counts().to_dict()
    )
    turns_per_conversation = frame.groupby("conversation_key").size()
    score_counts = Counter(int(value) for value in frame["annotation_score"])
    summary = {
        "input": str(args.input),
        "release_rule": "judge_label == positive",
        "rows": len(frame),
        "distinct_source_conversations": int(frame["conversation_key"].nunique()),
        "source_counts": dict(sorted(source_counts.items())),
        "source_conversation_counts": {
            str(key): int(value) for key, value in sorted(conversation_counts.items())
        },
        "annotation_score_counts": {
            str(key): value for key, value in sorted(score_counts.items())
        },
        "stored_context_messages": quantile_summary(frame["stored_context_messages"]),
        "target_turns_per_source_conversation": quantile_summary(turns_per_conversation),
        "target_message_index": quantile_summary(frame["target_message_index"]),
        "target_words": quantile_summary(frame["target_words"]),
        "verifier_confidence": quantile_summary(frame["verifier_confidence"]),
    }

    source_labels = [
        ("sharechat_chatgpt", "ShareChat--ChatGPT"),
        ("wildchat_full", "WildChat"),
        ("sharechat_grok", "ShareChat--Grok"),
        ("lmsys_chat_1m", "LMSYS-Chat-1M"),
    ]
    table_rows = [
        ("Released target turns", str(len(frame))),
        ("Distinct source conversations", str(frame["conversation_key"].nunique())),
    ]
    table_rows.extend(
        (
            f"{label} target turns",
            f"{source_counts[key]} ({100 * source_counts[key] / len(frame):.1f}\\%)",
        )
        for key, label in source_labels
        if source_counts[key]
    )
    table_rows.extend(
        [
            ("Stored context messages", format_iqr(summary["stored_context_messages"])),
            (
                "Target turns per source conversation",
                format_iqr(summary["target_turns_per_source_conversation"]),
            ),
            ("Target-turn index (zero-based)", format_iqr(summary["target_message_index"])),
            ("Target-message words", format_iqr(summary["target_words"])),
            ("Verifier confidence", format_iqr(summary["verifier_confidence"], 2)),
            (
                "Package score 7 / 8 / 9 / 10",
                " / ".join(str(score_counts[score]) for score in range(7, 11)),
            ),
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(table_rows, columns=["property", "value"]).to_csv(
        args.output_dir / "paper_table.csv", index=False
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
