from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    result = []
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
            result.append({"role": role, "content": content})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build last-k-context generation inputs for a dose-response ablation."
    )
    parser.add_argument(
        "--release", type=Path, default=Path("data/releases/WildDelusionVerified/train.parquet")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/context_ablation/truncation_sweep.parquet")
    )
    parser.add_argument("--keep-prior", type=int, nargs="+", default=[1, 3, 7])
    parser.add_argument(
        "--minimum-full-prior",
        type=int,
        default=13,
        help="Restrict to targets with enough history for a meaningful dose-response curve.",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.release).copy()
    frame = frame[frame["target_message_index"].astype(int) >= args.minimum_full_prior].copy()
    output_rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        full_messages = normalize_messages(list(row["messages"]))
        target_index = int(row["target_message_index"])
        if target_index >= len(full_messages):
            raise ValueError("Role normalization changed target indexing.")
        for keep in sorted(set(args.keep_prior)):
            start = max(0, target_index - keep)
            messages = full_messages[start : target_index + 1]
            if not messages or messages[-1]["role"] != "user":
                raise ValueError("Truncated input does not end at the target user turn.")
            output = dict(row)
            output.update(
                {
                    "context_arm": f"last_{keep}",
                    "full_target_message_index": target_index,
                    "full_prior_message_count": target_index,
                    "visible_prior_message_count": len(messages) - 1,
                    "messages": messages,
                    "target_message_index": len(messages) - 1,
                    "message_hash": f"{row['message_hash']}:last_{keep}",
                }
            )
            output_rows.append(output)

    output_frame = pd.DataFrame(output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_parquet(args.output, index=False)
    manifest = {
        "release": str(args.release),
        "output": str(args.output),
        "targets": int(frame["message_hash"].nunique()),
        "rows": len(output_frame),
        "keep_prior_messages": sorted(set(args.keep_prior)),
        "minimum_full_prior_messages": args.minimum_full_prior,
        "intervention": (
            "Keep only the last k source messages before the verified target. The target, system "
            "instruction, model snapshot, and generation parameters remain fixed."
        ),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
