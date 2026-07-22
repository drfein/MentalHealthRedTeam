from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten_readouts(path: Path, metadata: list[dict[str, Any]], layer: int, token: str) -> pd.DataFrame:
    rows = []
    for readout in read_jsonl(path):
        if int(readout["layer"]) != layer:
            continue
        idx = int(readout["row_idx"])
        meta = metadata[idx]
        target = meta["messages"][int(meta["target_message_index"])]
        token_scores = readout.get("token_scores", {})
        if token not in token_scores:
            raise ValueError(f"Token {token!r} is absent from readouts")
        rows.append(
            {
                "row_idx": idx,
                "source": meta.get("source"),
                "target_text": target["content"],
                "annotation_score": meta.get("annotation_score"),
                "judge_confidence": meta.get("judge_confidence"),
                "user_endorses_delusion": int(bool(meta.get("judge_potential_user_endorses_delusion"))),
                "exclusion": meta.get("judge_exclusion"),
                "primary_logit": float(token_scores[token]["max_logit"]),
                "primary_rank_strength": -np.log10(float(token_scores[token]["best_rank"])),
                "falsity_concern": float(readout["concept_scores"]["falsity_concern"]["mean_logit"]),
                "epistemic_confusion": float(
                    readout["concept_scores"].get("epistemic_confusion", {}).get("mean_logit", np.nan)
                ),
            }
        )
    return pd.DataFrame(rows)


def classifier(numeric: list[str], categorical: list[str]) -> Any:
    transform = ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            (
                "categorical",
                make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")),
                categorical,
            ),
        ]
    )
    return make_pipeline(transform, LogisticRegression(C=1.0, class_weight="balanced", max_iter=5_000))


def oof(frame: pd.DataFrame, features: list[str], folds: int, seed: int) -> np.ndarray:
    numeric = [column for column in features if column != "source"]
    categorical = [column for column in features if column == "source"]
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(
        classifier(numeric, categorical),
        frame[features],
        frame["user_endorses_delusion"],
        cv=split,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]


def bootstrap_auc_delta(frame: pd.DataFrame, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
        if sample["user_endorses_delusion"].nunique() < 2:
            continue
        values.append(
            roc_auc_score(sample["user_endorses_delusion"], sample["full_oof"])
            - roc_auc_score(sample["user_endorses_delusion"], sample["baseline_oof"])
        )
    return tuple(np.quantile(values, [0.025, 0.975]))


def exclusion_summary(frame: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    positive = frame.loc[frame["user_endorses_delusion"] == 1, "primary_logit"].to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for exclusion, group in frame[frame["user_endorses_delusion"] == 0].groupby("exclusion"):
        negative = group["primary_logit"].to_numpy()
        difference = positive.mean() - negative.mean()
        boot = np.empty(draws)
        for draw in range(draws):
            boot[draw] = (
                rng.choice(positive, len(positive), replace=True).mean()
                - rng.choice(negative, len(negative), replace=True).mean()
            )
        rows.append(
            {
                "exclusion": exclusion,
                "negative_n": len(negative),
                "positive_minus_negative_mean_logit": difference,
                "bootstrap_ci_low": np.quantile(boot, 0.025),
                "bootstrap_ci_high": np.quantile(boot, 0.975),
                "mannwhitney_p": mannwhitneyu(positive, negative, alternative="two-sided").pvalue,
            }
        )
    return pd.DataFrame(rows).sort_values("negative_n", ascending=False)


def plot(frame: pd.DataFrame, exclusions: pd.DataFrame, token: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    groups = [
        frame.loc[frame["user_endorses_delusion"] == label, "primary_logit"]
        for label in [0, 1]
    ]
    ax.violinplot(groups, positions=[0, 1], showmeans=True, showextrema=False)
    ax.set_xticks([0, 1], ["Contextual exclusion", "Verified delusion"])
    ax.set_ylabel(f"{token} J-space logit")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    shown = exclusions.sort_values("positive_minus_negative_mean_logit")
    x = shown["positive_minus_negative_mean_logit"].to_numpy()
    xerr = np.vstack([x - shown["bootstrap_ci_low"], shown["bootstrap_ci_high"] - x])
    ax.errorbar(x, np.arange(len(shown)), xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0, color="0.45", linewidth=1)
    ax.set_yticks(np.arange(len(shown)), shown["exclusion"])
    ax.set_xlabel("Verified minus exclusion mean logit")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate verified delusions against contextual hard negatives.")
    parser.add_argument("--readouts", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--prompt-variants", type=Path, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--primary-token", default="misunderstanding")
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--bootstrap-draws", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_jsonl(args.metadata)
    frame = flatten_readouts(args.readouts, metadata, args.layer, args.primary_token)
    excluded_oversize = 0
    if args.prompt_variants:
        prompts = pd.read_csv(args.prompt_variants)
        invalid_ids = set(
            prompts.loc[prompts["prompt_tokens"] > args.max_prompt_tokens, "row_idx"].astype(int)
        )
        excluded_oversize = len(invalid_ids)
        frame = frame[~frame["row_idx"].isin(invalid_ids)].copy()
    frame["log_text_chars"] = np.log1p(frame["target_text"].str.len())
    baseline = ["log_text_chars", "annotation_score", "judge_confidence", "source"]
    full = ["primary_logit", *baseline]
    frame["baseline_oof"] = oof(frame, baseline, args.cv_folds, args.seed)
    frame["full_oof"] = oof(frame, full, args.cv_folds, args.seed)
    baseline_auc = roc_auc_score(frame["user_endorses_delusion"], frame["baseline_oof"])
    full_auc = roc_auc_score(frame["user_endorses_delusion"], frame["full_oof"])
    ci = bootstrap_auc_delta(frame, args.bootstrap_draws, args.seed + 1)
    exclusions = exclusion_summary(frame, args.bootstrap_draws, args.seed + 2)
    summary = {
        "n": len(frame),
        "verified_n": int(frame["user_endorses_delusion"].sum()),
        "hard_negative_n": int((1 - frame["user_endorses_delusion"]).sum()),
        "excluded_oversize_n": excluded_oversize,
        "unadjusted_primary_auc": roc_auc_score(
            frame["user_endorses_delusion"], frame["primary_logit"]
        ),
        "baseline_oof_auc": baseline_auc,
        "full_oof_auc": full_auc,
        "oof_auc_improvement": full_auc - baseline_auc,
        "oof_auc_improvement_ci": list(ci),
        "full_oof_average_precision": average_precision_score(
            frame["user_endorses_delusion"], frame["full_oof"]
        ),
        "control_definition": (
            "Hard negatives passed the message-level delusion judge but were excluded by the full-conversation "
            "verifier as roleplay, fiction, joking, translation, third-party quotation, ordinary plausible content, "
            "or insufficient context."
        ),
    }
    hparams = vars(args).copy()
    for key in ["readouts", "metadata", "out_dir"]:
        hparams[key] = str(hparams[key])
    hparams["prompt_variants"] = str(args.prompt_variants) if args.prompt_variants else None
    (args.out_dir / "hparams.json").write_text(json.dumps(hparams, indent=2), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    frame.to_parquet(args.out_dir / "analysis_rows.parquet", index=False)
    exclusions.to_csv(args.out_dir / "exclusion_effects.csv", index=False)
    plot(frame, exclusions, args.primary_token, args.out_dir / "hard_negative_control.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
