#!/usr/bin/env python3
"""Render the paper's full-benchmark matched-framing figure from aggregates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze_full_counterfactual_behavior import plot_main_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_main_figure(
        pd.read_csv(args.results_dir / "behavior_by_frame.csv"),
        pd.read_csv(args.results_dir / "paired_frame_contrasts.csv"),
        args.output,
        package_behavior=pd.read_csv(
            args.results_dir / "package_behavior_by_frame.csv"
        ),
        package_contrasts=pd.read_csv(
            args.results_dir / "package_paired_frame_contrasts.csv"
        ),
    )


if __name__ == "__main__":
    main()
