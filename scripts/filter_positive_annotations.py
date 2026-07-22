#!/usr/bin/env python3
"""Retain successfully annotated rows meeting an explicit score cutoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        row
        for row in rows
        if row.get("annotation_error") is None
        and row.get("annotation_score") is not None
        and int(row["annotation_score"]) >= args.cutoff
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"input_rows": len(rows), "cutoff": args.cutoff, "selected": len(selected)}))


if __name__ == "__main__":
    main()
