from __future__ import annotations

from collections.abc import Mapping


MODEL_ORDER = [
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
    "gpt-4.1-mini-2025-04-14",
    "gpt-5-mini-2025-08-07",
    "gpt-5.2-2025-12-11",
    "gpt-5.5-2026-04-23",
]

MODEL_LABELS: Mapping[str, str] = {
    "gpt-3.5-turbo-0125": "GPT-3.5",
    "gpt-4-turbo-2024-04-09": "GPT-4 Turbo",
    "gpt-4o-2024-05-13": "GPT-4o",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "o1-2024-12-17": "o1",
    "o3-mini-2025-01-31": "o3 mini",
    "gpt-4.1-mini-2025-04-14": "GPT-4.1 mini",
    "gpt-5-mini-2025-08-07": "GPT-5 mini",
    "gpt-5.2-2025-12-11": "GPT-5.2",
    "gpt-5.5-2026-04-23": "GPT-5.5",
}

# Stable, colorblind-conscious identities used across all model-comparison figures.
MODEL_COLORS: Mapping[str, str] = {
    "gpt-3.5-turbo-0125": "#6B7280",
    "gpt-4-turbo-2024-04-09": "#0072B2",
    "gpt-4o-2024-05-13": "#009E73",
    "gpt-4o-mini-2024-07-18": "#56B4E9",
    "o1-2024-12-17": "#8C564B",
    "o3-mini-2025-01-31": "#E69F00",
    "gpt-4.1-mini-2025-04-14": "#D55E00",
    "gpt-5-mini-2025-08-07": "#CC79A7",
    "gpt-5.2-2025-12-11": "#332288",
    "gpt-5.5-2026-04-23": "#117733",
}


def apply_paper_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#374151",
            "axes.labelcolor": "#1F2937",
            "axes.titlecolor": "#111827",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "legend.fontsize": 9,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
