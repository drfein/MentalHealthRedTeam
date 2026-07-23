from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PRIOR_COUNTS = (0, 1, 2, 4, 8)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        role = str(message.get("role", "")).lower()
        if role in {"human", "user"}:
            role = "user"
        elif role in {"assistant", "llm"}:
            role = "assistant"
        else:
            continue
        content = str(message.get("content", "") or "").strip()
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def build_arms(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = normalize_messages(list(row["messages"]))
    target_index = int(row["target_message_index"])
    if target_index >= len(messages):
        raise ValueError("Role normalization changed target indexing.")
    if messages[target_index]["role"] != "user":
        raise ValueError("The retained target must be a user message.")

    arms: list[tuple[str, list[dict[str, str]]]] = []
    for prior_count in PRIOR_COUNTS:
        start = target_index - prior_count
        visible = messages[start : target_index + 1]
        arms.append((f"prior_{prior_count}", visible))
    arms.append(("prior_all", messages[: target_index + 1]))

    output = []
    for arm, visible in arms:
        arm_row = dict(row)
        arm_row.update(
            {
                "context_arm": arm,
                "full_target_message_index": target_index,
                "full_prior_message_count": target_index,
                "visible_prior_message_count": len(visible) - 1,
                "messages": visible,
                "target_message_index": len(visible) - 1,
                "message_hash": f"{row['message_hash']}:{arm}",
            }
        )
        output.append(arm_row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a matched 0/1/2/4/8/all source-message context sweep."
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("data/releases/WildDelusionVerified/train.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/context_dose_response/inputs.parquet"),
    )
    parser.add_argument("--minimum-prior-messages", type=int, default=8)
    args = parser.parse_args()

    release = pd.read_parquet(args.release).copy()
    release = release[
        release["target_message_index"].astype(int) >= args.minimum_prior_messages
    ].copy()
    rows = [
        arm
        for row in release.to_dict(orient="records")
        for arm in build_arms(row)
    ]
    output = pd.DataFrame(rows)
    counts = output.groupby("message_hash")["context_arm"].nunique()
    if not counts.eq(1).all():
        raise ValueError("Dose-response message hashes must be unique.")
    arm_counts = output.groupby("context_arm").size()
    if arm_counts.nunique() != 1:
        raise ValueError("Every context arm must contain the same targets.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    manifest = {
        "release": str(args.release),
        "output": str(args.output),
        "targets": len(release),
        "conversations": int(release["conversation_id"].nunique()),
        "rows": len(output),
        "arms": [f"prior_{count}" for count in PRIOR_COUNTS] + ["prior_all"],
        "minimum_prior_messages": args.minimum_prior_messages,
        "counting_rule": "Both user and assistant source messages count as prior messages.",
        "system_prompt": None,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
