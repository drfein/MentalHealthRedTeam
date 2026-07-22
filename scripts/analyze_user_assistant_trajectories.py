from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BIN_LABELS = ["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fixed_effect_trend(frame: pd.DataFrame, outcome: str) -> dict[str, float | int]:
    fitted = smf.ols(
        f"{outcome} ~ conversation_progress + C(conversation_id)", data=frame
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": frame["conversation_id"], "use_correction": True},
        use_t=True,
    )
    low, high = fitted.conf_int().loc["conversation_progress"]
    return {
        "coefficient": float(fitted.params["conversation_progress"]),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(fitted.pvalues["conversation_progress"]),
        "rows": len(frame),
        "conversations": int(frame["conversation_id"].nunique()),
    }


def adjusted_assistant_trend(frame: pd.DataFrame) -> dict[str, float | int]:
    fitted = smf.ols(
        "assistant_positive ~ conversation_progress + user_positive + C(conversation_id)",
        data=frame,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": frame["conversation_id"], "use_correction": True},
        use_t=True,
    )
    output: dict[str, float | int] = {
        "rows": len(frame),
        "conversations": int(frame["conversation_id"].nunique()),
    }
    for coefficient in ["conversation_progress", "user_positive"]:
        low, high = fitted.conf_int().loc[coefficient]
        prefix = "progress" if coefficient == "conversation_progress" else "user_positive"
        output[f"{prefix}_coefficient"] = float(fitted.params[coefficient])
        output[f"{prefix}_ci_low"] = float(low)
        output[f"{prefix}_ci_high"] = float(high)
        output[f"{prefix}_p_value"] = float(fitted.pvalues[coefficient])
    return output


def trajectory_summary(
    frame: pd.DataFrame, column: str, *, draws: int, seed: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    macro = (
        frame.groupby(["conversation_id", "progress_bin"], observed=True)[column]
        .mean()
        .unstack()
        .reindex(columns=BIN_LABELS)
    )
    values = macro.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(macro), size=(draws, len(macro)))
    boot = np.nanmean(values[sampled], axis=1)
    rows = []
    for index, label in enumerate(BIN_LABELS):
        rows.append(
            {
                "progress_bin": label,
                "rate": float(np.nanmean(values[:, index])),
                "ci_low": float(np.nanquantile(boot[:, index], 0.025)),
                "ci_high": float(np.nanquantile(boot[:, index], 0.975)),
                "conversations": int(np.isfinite(values[:, index]).sum()),
            }
        )
    endpoint = values[:, -1] - values[:, 0]
    endpoint = endpoint[np.isfinite(endpoint)]
    endpoint_boot = endpoint[
        rng.integers(0, len(endpoint), size=(draws, len(endpoint)))
    ].mean(axis=1)
    endpoint_summary = {
        "last_minus_first_quintile": float(endpoint.mean()),
        "ci_low": float(np.quantile(endpoint_boot, 0.025)),
        "ci_high": float(np.quantile(endpoint_boot, 0.975)),
    }
    return pd.DataFrame(rows), endpoint_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare user endorsement and assistant endorsement trajectories."
    )
    parser.add_argument(
        "--user-judgments",
        type=Path,
        default=Path("results/observed_all_turn_user_trajectories/package_judgments.jsonl"),
    )
    parser.add_argument(
        "--assistant-judgments",
        type=Path,
        default=Path("results/observed_all_turn_trajectories/package_judgments.jsonl"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/user_assistant_trajectories")
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    user = pd.DataFrame(read_jsonl(args.user_judgments))
    assistant = pd.DataFrame(read_jsonl(args.assistant_judgments))
    attempted_user_turns = len(user)
    user_errors = int(user.get("judge_error", pd.Series(dtype=object)).notna().sum())
    if "judge_error" in user:
        user = user[user["judge_error"].isna()].copy()
    keys = ["conversation_id", "assistant_message_index"]
    paired = user.merge(
        assistant[
            keys + ["annotation_score", "follows_retained_target", "response"]
        ],
        on=keys,
        suffixes=("_user", "_assistant"),
        validate="one_to_one",
    )
    if len(paired) != attempted_user_turns - user_errors:
        raise ValueError("Successful user judgments did not align one-to-one with assistant turns.")
    paired["user_positive"] = (paired["annotation_score_user"].astype(int) >= 7).astype(int)
    paired["assistant_positive"] = (
        paired["annotation_score_assistant"].astype(int) >= 7
    ).astype(int)
    paired["assistant_minus_user"] = (
        paired["assistant_positive"] - paired["user_positive"]
    )
    paired["progress_bin"] = pd.cut(
        paired["conversation_progress"],
        [-1e-9, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=BIN_LABELS,
        include_lowest=True,
    )

    user_bins, user_endpoint = trajectory_summary(
        paired, "user_positive", draws=args.bootstrap_draws, seed=args.seed
    )
    assistant_bins, assistant_endpoint = trajectory_summary(
        paired, "assistant_positive", draws=args.bootstrap_draws, seed=args.seed + 1
    )
    _, gap_endpoint = trajectory_summary(
        paired, "assistant_minus_user", draws=args.bootstrap_draws, seed=args.seed + 2
    )
    user_bins["series"] = "User endorses delusion"
    assistant_bins["series"] = "Assistant endorses delusion"
    bins = pd.concat([user_bins, assistant_bins], ignore_index=True)
    summary = {
        "attempted_user_turns": attempted_user_turns,
        "user_judge_errors": user_errors,
        "aligned_turn_pairs": len(paired),
        "conversations": int(paired["conversation_id"].nunique()),
        "user_positive_rate": float(paired["user_positive"].mean()),
        "assistant_positive_rate": float(paired["assistant_positive"].mean()),
        "user_binary_fixed_effect_trend": fixed_effect_trend(paired, "user_positive"),
        "user_score_fixed_effect_trend": fixed_effect_trend(paired, "annotation_score_user"),
        "assistant_binary_fixed_effect_trend": fixed_effect_trend(
            paired, "assistant_positive"
        ),
        "assistant_trend_adjusted_for_user_positive": adjusted_assistant_trend(paired),
        "user_endpoint_change": user_endpoint,
        "assistant_endpoint_change": assistant_endpoint,
        "assistant_minus_user_endpoint_change": gap_endpoint,
        "uncertainty": (
            "Plot and endpoint intervals use 10000 conversation-cluster bootstrap draws; "
            "regressions use conversation-clustered small-sample-corrected t inference."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.out_dir / "aligned_user_assistant_turns.csv", index=False)
    bins.to_csv(args.out_dir / "trajectory_bins.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    colors = ["#C8553D", "#176B87"]
    for color, (series, group) in zip(colors, bins.groupby("series", sort=False), strict=True):
        group = group.set_index("progress_bin").loc[BIN_LABELS]
        x = np.arange(len(group))
        ax.errorbar(
            x,
            group["rate"] * 100,
            yerr=np.vstack(
                [
                    (group["rate"] - group["ci_low"]) * 100,
                    (group["ci_high"] - group["rate"]) * 100,
                ]
            ),
            marker="o",
            capsize=3,
            color=color,
            label=series,
        )
    ax.set_xticks(np.arange(len(BIN_LABELS)), BIN_LABELS)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Normalized conversation progress")
    ax.set_ylabel("Conversation-macro SPIRALS-positive rate (%)")
    ax.set_title("User and Assistant Endorsement Across Production Conversations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(args.out_dir / "user_assistant_trajectories.png", dpi=240, bbox_inches="tight")
    fig.savefig(args.out_dir / "user_assistant_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
