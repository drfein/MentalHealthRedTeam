from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wild_delusion_miner.plot_style import MODEL_COLORS, apply_paper_style  # noqa: E402


ARM_ORDER = ["target_only", "last_1", "last_3", "last_7", "full_context"]
ARM_LABELS = ["0", "1", "3", "7", "All"]
MODEL_LABELS = {
    "gpt-4.1-mini-2025-04-14": "GPT-4.1 mini",
    "gpt-5.2-2025-12-11": "GPT-5.2",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bootstrap_rates(
    frame: pd.DataFrame, *, draws: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    conversations = frame["conversation_id"].drop_duplicates().to_numpy()
    grouped = {
        conversation_id: frame[frame["conversation_id"] == conversation_id]
        for conversation_id in conversations
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty((draws, len(ARM_ORDER)))
    for draw in range(draws):
        sampled = rng.choice(conversations, size=len(conversations), replace=True)
        sample = pd.concat([grouped[item] for item in sampled], ignore_index=True)
        estimates[draw] = sample.groupby("arm")["endorse"].mean().reindex(ARM_ORDER)
    return np.quantile(estimates, 0.025, axis=0), np.quantile(estimates, 0.975, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze last-k context dose-response results.")
    parser.add_argument(
        "--intermediate-judgments",
        type=Path,
        default=Path("results/context_ablation/truncation_sweep_package_judgments.jsonl"),
    )
    parser.add_argument(
        "--paired-endpoints",
        type=Path,
        default=Path("results/context_ablation/paired_scores.csv"),
    )
    parser.add_argument(
        "--release", type=Path, default=Path("data/releases/WildDelusionVerified/train.parquet")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/context_ablation/truncation_sweep")
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    intermediate = pd.DataFrame(read_jsonl(args.intermediate_judgments))
    if intermediate.get("judge_error", pd.Series(dtype=object)).notna().any():
        raise ValueError("Truncation judgments contain errors.")
    intermediate["endorse"] = (intermediate["annotation_score"].astype(int) >= 7).astype(int)
    intermediate["model"] = intermediate["model_id"]
    intermediate["arm"] = intermediate["intervention_arm"]

    endpoints = pd.read_csv(args.paired_endpoints)
    release = pd.read_parquet(args.release).reset_index(names="original_row_idx")[
        ["original_row_idx", "target_message_index"]
    ]
    endpoints = endpoints.merge(release, on="original_row_idx", validate="many_to_one")
    endpoints = endpoints[endpoints["target_message_index_y"] >= 13].copy()
    target = endpoints.rename(columns={"target_only_endorse": "endorse"})
    target["arm"] = "target_only"
    full = endpoints.rename(columns={"full_context_endorse": "endorse"})
    full["arm"] = "full_context"
    columns = ["original_row_idx", "model", "conversation_id", "arm", "endorse"]
    data = pd.concat(
        [target[columns], intermediate[columns], full[columns]], ignore_index=True
    )
    data = data[data["model"].isin(MODEL_LABELS)].copy()

    rows = []
    for model_index, (model, model_label) in enumerate(MODEL_LABELS.items()):
        model_data = data[data["model"] == model]
        counts = model_data.groupby("original_row_idx")["arm"].nunique()
        complete = set(counts[counts == len(ARM_ORDER)].index)
        model_data = model_data[model_data["original_row_idx"].isin(complete)].copy()
        rates = model_data.groupby("arm")["endorse"].mean().reindex(ARM_ORDER)
        low, high = bootstrap_rates(
            model_data, draws=args.bootstrap_draws, seed=args.seed + model_index
        )
        for arm_index, arm in enumerate(ARM_ORDER):
            rows.append(
                {
                    "model": model,
                    "model_label": model_label,
                    "arm": arm,
                    "prior_messages": ARM_LABELS[arm_index],
                    "targets": len(complete),
                    "endorsement_rate": float(rates[arm]),
                    "ci_low": float(low[arm_index]),
                    "ci_high": float(high[arm_index]),
                }
            )
    results = pd.DataFrame(rows)
    summary = {
        "models": list(MODEL_LABELS),
        "arms": ARM_ORDER,
        "minimum_available_prior_messages": 13,
        "results": rows,
        "uncertainty": (
            f"95% percentile intervals from {args.bootstrap_draws} source-conversation-cluster "
            "bootstrap draws; each model is analyzed separately."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "context_truncation_rates.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    x = np.arange(len(ARM_ORDER))
    for model, label in MODEL_LABELS.items():
        group = results[results["model"] == model].set_index("arm").loc[ARM_ORDER]
        ax.errorbar(
            x,
            group["endorsement_rate"] * 100,
            yerr=np.vstack(
                [
                    (group["endorsement_rate"] - group["ci_low"]) * 100,
                    (group["ci_high"] - group["endorsement_rate"]) * 100,
                ]
            ),
            marker="o",
            capsize=3,
            color=MODEL_COLORS[model],
            label=label,
        )
    ax.set_xticks(x, ARM_LABELS)
    ax.set_xlabel("Prior source messages retained")
    ax.set_ylabel("SPIRALS endorsement rate (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("Context Dose-Response on Targets With 13+ Prior Messages")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(args.out_dir / "context_truncation.png", dpi=240, bbox_inches="tight")
    fig.savefig(args.out_dir / "context_truncation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
