#!/usr/bin/env python3
"""Plot strict delusion-endorsement rates against model release dates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
import matplotlib.dates as mdates
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wild_delusion_miner.plot_style import (  # noqa: E402
    MODEL_COLORS,
    apply_paper_style,
)


ENDORSEMENT_FLAG = "bot_endorses_delusion"
EVENT_COLORS = {
    "provider": "#4477AA",
    "public": "#6B7280",
    "regulatory": "#AA3377",
    "research": "#228833",
}


def load_timeline(
    rates_path: Path,
    route_rates_path: Path,
    config_path: Path,
) -> tuple[pd.DataFrame, list[dict[str, str]], dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rates = pd.read_csv(rates_path)
    rates = rates[
        (rates["flag"] == ENDORSEMENT_FLAG) & (rates["model"] != "all")
    ].copy()
    expected = set(config["models"])
    observed = set(rates["model"])
    if observed != expected:
        raise ValueError(
            f"Model mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    route_rates = pd.read_csv(route_rates_path)
    route_observed = set(route_rates["model"])
    if route_observed != expected:
        raise ValueError(
            f"Route model mismatch: missing={sorted(expected - route_observed)}, "
            f"unexpected={sorted(route_observed - expected)}"
        )
    route_rates["historical_positive"] = (
        route_rates["historical_n"] * route_rates["historical_rate"]
    ).round().astype(int)
    rates = rates.merge(
        route_rates[["model", "historical_n", "historical_positive"]],
        on="model",
        validate="one_to_one",
    )
    rates = rates.rename(
        columns={"positive": "primary_positive", "total": "primary_n"}
    )
    rates["positive"] = rates["primary_positive"] + rates["historical_positive"]
    rates["total"] = rates["primary_n"] + rates["historical_n"]
    rates["rate"] = rates["positive"] / rates["total"]
    intervals = [
        wilson_interval(int(row.positive), int(row.total))
        for row in rates.itertuples(index=False)
    ]
    rates["wilson_lower"] = [interval[0] for interval in intervals]
    rates["wilson_upper"] = [interval[1] for interval in intervals]
    rates["model_label"] = rates["model"].map(
        lambda model: config["models"][model]["label"]
    )
    rates["release_date"] = pd.to_datetime(
        rates["model"].map(
            lambda model: config["models"][model]["release_date"]
        )
    )
    rates = rates.sort_values("release_date").reset_index(drop=True)
    return rates, config["events"], config


def wilson_interval(
    positive: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total == 0:
        return math.nan, math.nan
    proportion = positive / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - half_width, center + half_width


def plot_timeline(
    rates: pd.DataFrame,
    events: list[dict[str, str]],
    output_dir: Path,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(13.2, 6.5))
    x = rates["release_date"]
    y = rates["rate"] * 100
    yerr = [
        (rates["rate"] - rates["wilson_lower"]) * 100,
        (rates["wilson_upper"] - rates["rate"]) * 100,
    ]

    ax.plot(x, y, color="#374151", linewidth=1.6, zorder=2)
    for row, lower, upper in zip(
        rates.itertuples(index=False),
        yerr[0],
        yerr[1],
        strict=True,
    ):
        color = MODEL_COLORS.get(row.model, "#374151")
        ax.errorbar(
            row.release_date,
            row.rate * 100,
            yerr=[[lower], [upper]],
            fmt="o",
            markersize=8,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            zorder=3,
        )

    label_offsets = {
        "gpt-3.5-turbo-0125": (8, 8),
        "gpt-4-turbo-2024-04-09": (8, 8),
        "gpt-4o-2024-05-13": (8, 8),
        "gpt-4o-mini-2024-07-18": (8, -17),
        "o1-2024-12-17": (-8, 9),
        "o3-mini-2025-01-31": (8, 8),
        "gpt-4.1-mini-2025-04-14": (-8, 9),
        "gpt-5-mini-2025-08-07": (8, 12),
        "gpt-5.2-2025-12-11": (-8, 12),
        "gpt-5.5-2026-04-23": (-8, 9),
    }
    for row in rates.itertuples(index=False):
        dx, dy = label_offsets[row.model]
        ax.annotate(
            f"{row.model_label}\n{row.release_date:%b %-d, %Y}",
            (row.release_date, row.rate * 100),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            fontsize=9.5,
            color="#1F2937",
            zorder=4,
        )

    for event in events:
        date = pd.Timestamp(event["date"])
        color = EVENT_COLORS[event["category"]]
        ax.axvline(
            date,
            color=color,
            linestyle=(0, (4, 3)),
            linewidth=1.3,
            alpha=0.9,
            zorder=1,
        )
        ax.annotate(
            event["short_label"],
            xy=(date, event["label_y"]),
            xytext=(4, 0),
            textcoords="offset points",
            rotation=90,
            ha="left",
            va="top",
            fontsize=9,
            color=color,
        )

    ax.set_xlim(pd.Timestamp("2023-02-01"), pd.Timestamp("2026-05-15"))
    ax.set_ylim(0, 26)
    ax.set_xlabel("Model release date")
    ax.set_ylabel("Delusion endorsement rate (%)")
    ax.xaxis.set_major_locator(
        mdates.MonthLocator(bymonth=[1, 4, 7, 10], bymonthday=1)
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "model_endorsement_timeline.png", dpi=300)
    fig.savefig(output_dir / "model_endorsement_timeline.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rates",
        type=Path,
        default=Path(
            "paper/iclr2026/artifacts/multimodel/spirals_taxonomy_by_model.csv"
        ),
    )
    parser.add_argument(
        "--route-rates",
        type=Path,
        default=Path(
            "paper/iclr2026/artifacts/discovery_route_benchmark/"
            "endorsement_by_discovery_route.csv"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_endorsement_timeline.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/model_endorsement_timeline"),
    )
    args = parser.parse_args()

    rates, events, config = load_timeline(
        args.rates,
        args.route_rates,
        args.config,
    )
    plot_timeline(rates, events, args.output_dir)
    rates.to_csv(args.output_dir / "model_endorsement_timeline.csv", index=False)
    (args.output_dir / "hparams.json").write_text(
        json.dumps(
            {
                "rates": str(args.rates),
                "route_rates": str(args.route_rates),
                "config": str(args.config),
                "endpoint": ENDORSEMENT_FLAG,
                "positive_cutoff": 7,
                "judge": "GPT-5.4-mini with the pinned SPIRALS package prompt",
                "cohort": (
                    "All usable primary and historical discovery-route responses"
                ),
                "model_date_basis": config["model_date_basis"],
                "interval": "Pointwise 95% Wilson score interval",
                "events": config["events"],
                "interpretation": (
                    "Descriptive model-snapshot timeline; release date does not isolate "
                    "architecture, scale, training, system behavior, or policy changes."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
