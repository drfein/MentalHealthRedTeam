from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def direct_assertions(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(read_jsonl(path))
    arm = "intervention_arm" if "intervention_arm" in frame else "condition"
    if arm in frame:
        frame = frame[frame[arm].eq("direct_assertion")].copy()
    return frame


def load_readouts(path: Path, layers: list[int]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for row in read_jsonl(path):
        if row.get("skipped_oversize") or int(row["layer"]) not in layers:
            continue
        condition = str(row.get("condition", row.get("intervention_arm", "")))
        if condition != "direct_assertion":
            continue
        rows.append(
            {
                "original_row_idx": int(row["original_row_idx"]),
                "layer": int(row["layer"]),
                "inverse_misinformation_max": -float(
                    row["token_scores"]["misinformation"]["max_logit"]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["original_row_idx", "layer"]).any():
        raise ValueError(f"Duplicate item-layer readout in {path}")
    return frame.pivot(
        index="original_row_idx",
        columns="layer",
        values="inverse_misinformation_max",
    ).reset_index()


def load_judgments(path: Path, score: str, output: str) -> pd.DataFrame:
    frame = direct_assertions(path)
    if "judge_error" in frame:
        frame = frame[frame["judge_error"].isna()].copy()
    result = frame[["original_row_idx", score]].rename(columns={score: output})
    if result["original_row_idx"].duplicated().any():
        raise ValueError(f"Duplicate item judgment in {path}")
    return result


def auc_or_nan(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else np.nan


def bootstrap_metrics(
    frame: pd.DataFrame,
    labels: np.ndarray,
    mid_layer: int,
    late_layer: int,
    draws: int,
    seed: int,
) -> dict[str, float]:
    mid = frame[mid_layer].to_numpy(float)
    late = frame[late_layer].to_numpy(float)
    graded = frame["framing_score"].to_numpy(float)
    estimates = {
        "mid_auc": auc_or_nan(labels, mid),
        "late_auc": auc_or_nan(labels, late),
        "late_minus_mid_auc": auc_or_nan(labels, late) - auc_or_nan(labels, mid),
        "late_graded_spearman": float(spearmanr(late, graded).statistic),
    }
    rng = np.random.default_rng(seed)
    sampled: dict[str, list[float]] = {key: [] for key in estimates}
    for _ in range(draws):
        index = rng.integers(0, len(frame), len(frame))
        y = labels[index]
        if np.unique(y).size != 2:
            continue
        mid_auc = auc_or_nan(y, mid[index])
        late_auc = auc_or_nan(y, late[index])
        values = {
            "mid_auc": mid_auc,
            "late_auc": late_auc,
            "late_minus_mid_auc": late_auc - mid_auc,
            "late_graded_spearman": float(
                spearmanr(late[index], graded[index]).statistic
            ),
        }
        for key, value in values.items():
            if np.isfinite(value):
                sampled[key].append(value)
    result: dict[str, float] = {}
    for key, estimate in estimates.items():
        low, high = np.quantile(sampled[key], [0.025, 0.975])
        result[key] = estimate
        result[f"{key}_ci_low"] = float(low)
        result[f"{key}_ci_high"] = float(high)
    return result


def analyze_model(
    root: Path,
    spec: dict[str, Any],
    expected_items: int,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    generations = direct_assertions(root / spec["generation_path"])[
        ["original_row_idx", "target_text", "response"]
    ]
    if generations["original_row_idx"].duplicated().any():
        raise ValueError(f"Duplicate generation for {spec['name']}")
    mid, late = int(spec["mid_layer"]), int(spec["late_layer"])
    frame = (
        generations.merge(
            load_readouts(root / spec["readout_path"], [mid, late]),
            on="original_row_idx",
            validate="one_to_one",
        )
        .merge(
            load_judgments(
                root / spec["framing_judgment_path"],
                "reality_endorsement_score",
                "framing_score",
            ),
            on="original_row_idx",
            validate="one_to_one",
        )
        .merge(
            load_judgments(
                root / spec["package_judgment_path"],
                "annotation_score",
                "package_score",
            ),
            on="original_row_idx",
            validate="one_to_one",
        )
    )
    if len(frame) != expected_items:
        raise ValueError(
            f"{spec['name']} merged {len(frame)} items; expected {expected_items}"
        )
    frame.insert(0, "model", spec["name"])
    endpoints = {
        "framing_score_gte_4": frame["framing_score"].to_numpy(float) >= 4,
        "exact_spirals_score_gte_7": frame["package_score"].to_numpy(float) >= 7,
    }
    rows: list[dict[str, Any]] = []
    for offset, (endpoint, labels) in enumerate(endpoints.items()):
        rows.append(
            {
                "model": spec["name"],
                "role": spec.get("role", "replication_model"),
                "model_id": spec["model_id"],
                "weight_precision": spec["weight_precision"],
                "endpoint": endpoint,
                "n": len(frame),
                "positive_n": int(labels.sum()),
                "mid_layer": mid,
                "late_layer": late,
                **bootstrap_metrics(
                    frame, labels, mid, late, draws, seed + 100 * offset
                ),
            }
        )
    return frame, rows


def plot_performance(performance: pd.DataFrame, output: Path) -> None:
    labels = {
        "framing_score_gte_4": "Framing-aware",
        "exact_spirals_score_gte_7": "Exact SPIRALS",
    }
    models = list(dict.fromkeys(performance["model"]))
    fig, ax = plt.subplots(figsize=(8.2, max(3.2, 1.1 * len(models) + 1.5)))
    colors = {"framing_score_gte_4": "#2166AC", "exact_spirals_score_gte_7": "#B2182B"}
    for endpoint, offset in [("framing_score_gte_4", -0.13), ("exact_spirals_score_gte_7", 0.13)]:
        part = performance[performance["endpoint"].eq(endpoint)].set_index("model").loc[models]
        y = np.arange(len(models)) + offset
        ax.errorbar(
            part["late_auc"],
            y,
            xerr=np.vstack(
                [
                    part["late_auc"] - part["late_auc_ci_low"],
                    part["late_auc_ci_high"] - part["late_auc"],
                ]
            ),
            fmt="o",
            capsize=3,
            color=colors[endpoint],
            label=labels[endpoint],
        )
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(models)), models)
    ax.set_xlim(0.25, 1.0)
    ax.set_xlabel("Late-layer AUROC of inverse misinformation readout")
    ax.set_title("Pre-specified J-space indicator across open models", loc="left", weight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen cross-model J-space replication."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parents[1]
    performance_rows: list[dict[str, Any]] = []
    analysis_frames: list[pd.DataFrame] = []
    for index, spec in enumerate(config["models"]):
        frame, rows = analyze_model(
            root,
            spec,
            int(config["expected_items"]),
            int(config["bootstrap_draws"]),
            int(config["seed"]) + 1000 * index,
        )
        analysis_frames.append(frame)
        performance_rows.extend(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    performance = pd.DataFrame(performance_rows)
    performance.to_csv(args.output_dir / "performance.csv", index=False)
    pd.concat(analysis_frames, ignore_index=True).to_parquet(
        args.output_dir / "analysis_rows.parquet", index=False
    )
    (args.output_dir / "hparams.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    plot_performance(performance, args.output_dir / "cross_model_auc.png")


if __name__ == "__main__":
    main()
