#!/usr/bin/env python3
"""Discovery/test screening of complete J-space vocabularies across models."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


WORD_PATTERN = re.compile(r"^[A-Za-z][A-Za-z'-]{2,30}$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_word(decoded: str, is_special: bool = False) -> str | None:
    if is_special:
        return None
    word = decoded.strip().lower()
    if not WORD_PATTERN.fullmatch(word):
        return None
    return word


def auc_columns(scores: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    outcome = np.asarray(outcome, dtype=bool)
    positive_n = int(outcome.sum())
    negative_n = int((~outcome).sum())
    if not positive_n or not negative_n:
        return np.full(scores.shape[1], np.nan)
    ranks = rankdata(scores, axis=0, method="average")
    positive_rank_sum = ranks[outcome].sum(axis=0)
    u_statistic = positive_rank_sum - positive_n * (positive_n + 1) / 2
    return u_statistic / (positive_n * negative_n)


def split_items(
    item_ids: np.ndarray,
    outcome_matrix: np.ndarray,
    discovery_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    endorsement_count = outcome_matrix.sum(axis=0)
    rng = np.random.default_rng(seed)
    discovery: list[int] = []
    test: list[int] = []
    for count in np.unique(endorsement_count):
        indices = np.flatnonzero(endorsement_count == count)
        rng.shuffle(indices)
        cut = int(round(len(indices) * discovery_fraction))
        discovery.extend(indices[:cut].tolist())
        test.extend(indices[cut:].tolist())
    return (
        item_ids[np.array(sorted(discovery))],
        item_ids[np.array(sorted(test))],
    )


def load_model(spec: dict[str, Any], positive_threshold: int) -> dict[str, Any]:
    cache_dir = Path(spec["cache_dir"])
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if not np.lib.format.open_memmap(
        cache_dir / "completed.uint8.npy", mode="r"
    ).all():
        raise ValueError(f"Incomplete cache: {cache_dir}")
    items = pd.DataFrame(read_jsonl(cache_dir / "items.jsonl"))
    tokens = pd.DataFrame(read_jsonl(cache_dir / "tokens.jsonl"))
    logits = np.lib.format.open_memmap(cache_dir / "logits.float16.npy", mode="r")
    judgments = pd.DataFrame(read_jsonl(Path(spec["judgments"])))
    judgments = judgments[
        judgments["intervention_arm"].eq(spec.get("arm", "direct_assertion"))
    ].copy()
    judgments["positive"] = (
        judgments["reality_endorsement_score"].astype(float) >= positive_threshold
    )
    judgments = judgments[["original_row_idx", "positive"]].drop_duplicates()
    if judgments["original_row_idx"].duplicated().any():
        raise ValueError(f"Duplicate judgments for {spec['name']}")
    items = items.reset_index(names="matrix_row").merge(
        judgments,
        on="original_row_idx",
        validate="one_to_one",
    )
    return {
        "name": spec["name"],
        "layer": int(manifest["layer"]),
        "items": items,
        "tokens": tokens,
        "logits": logits,
    }


def screen_model(
    model: dict[str, Any],
    discovery_ids: set[int],
    chunk_size: int,
) -> pd.DataFrame:
    rows = model["items"]
    selected = rows["original_row_idx"].isin(discovery_ids).to_numpy()
    matrix_rows = rows.loc[selected, "matrix_row"].to_numpy(dtype=int)
    outcome = rows.loc[selected, "positive"].to_numpy(dtype=bool)
    logits = model["logits"]
    auc = np.empty(logits.shape[1], dtype=np.float32)
    for start in range(0, logits.shape[1], chunk_size):
        stop = min(start + chunk_size, logits.shape[1])
        auc[start:stop] = auc_columns(
            np.asarray(logits[matrix_rows, start:stop], dtype=np.float32),
            outcome,
        )
    tokens = model["tokens"].copy()
    tokens["raw_auc"] = auc
    tokens["direction"] = np.where(tokens["raw_auc"] >= 0.5, 1, -1)
    tokens["discovery_auc"] = np.maximum(tokens["raw_auc"], 1 - tokens["raw_auc"])
    tokens["word"] = [
        canonical_word(decoded, bool(special))
        for decoded, special in zip(tokens["decoded"], tokens["is_special"], strict=True)
    ]
    interpretable = tokens.dropna(subset=["word"]).sort_values(
        "discovery_auc", ascending=False
    )
    return interpretable.drop_duplicates("word", keep="first").reset_index(drop=True)


def candidate_scores(
    models: list[dict[str, Any]],
    screens: dict[str, pd.DataFrame],
    item_ids: set[int],
    words: list[str],
    directions: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model_scores: list[np.ndarray] = []
    model_outcomes: list[np.ndarray] = []
    included_models: list[str] = []
    for model in models:
        lookup = screens[model["name"]].set_index("word")
        available = [word for word in words if word in lookup.index]
        if len(available) != len(words):
            raise ValueError(f"Missing candidate token in {model['name']}")
        rows = model["items"]
        selected = rows["original_row_idx"].isin(item_ids).to_numpy()
        ordered = rows.loc[selected].sort_values("original_row_idx")
        token_ids = lookup.loc[words, "token_id"].to_numpy(dtype=int)
        matrix_rows = ordered["matrix_row"].to_numpy(dtype=int)
        values = np.asarray(
            model["logits"][np.ix_(matrix_rows, token_ids)],
            dtype=np.float32,
        )
        values *= np.array([directions[word] for word in words])[None, :]
        model_scores.append(values)
        model_outcomes.append(ordered["positive"].to_numpy(dtype=bool))
        included_models.append(model["name"])
    return np.stack(model_scores), np.stack(model_outcomes), included_models


def evaluate_model_candidates(
    model: dict[str, Any],
    screen: pd.DataFrame,
    item_ids: set[int],
    top_k: int,
) -> pd.DataFrame:
    selected = screen.head(top_k).copy()
    words = selected["word"].tolist()
    directions = dict(zip(selected["word"], selected["direction"], strict=True))
    scores, outcomes, _ = candidate_scores(
        [model],
        {model["name"]: screen},
        item_ids,
        words,
        directions,
    )
    auc = auc_columns(scores[0], outcomes[0])
    selected["model"] = model["name"]
    selected["test_auc"] = auc
    selected["test_n"] = scores.shape[1]
    selected["test_positive_n"] = int(outcomes[0].sum())
    return selected[
        [
            "model",
            "word",
            "token_id",
            "decoded",
            "direction",
            "discovery_auc",
            "raw_auc",
            "test_auc",
            "test_n",
            "test_positive_n",
        ]
    ]


def bootstrap_macro_auc(
    scores: np.ndarray,
    outcomes: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    model_n, item_n, candidate_n = scores.shape
    estimates = np.empty((draws, candidate_n), dtype=np.float32)
    for draw in range(draws):
        sampled = rng.integers(0, item_n, size=item_n)
        model_auc = [
            auc_columns(scores[model_idx, sampled], outcomes[model_idx, sampled])
            for model_idx in range(model_n)
        ]
        estimates[draw] = np.nanmean(np.stack(model_auc), axis=0)
    return np.nanquantile(estimates, 0.025, axis=0), np.nanquantile(
        estimates, 0.975, axis=0
    )


def permutation_p_values(
    scores: np.ndarray,
    outcomes: np.ndarray,
    observed: np.ndarray,
    draws: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    null = np.empty((draws, scores.shape[2]), dtype=np.float32)
    for draw in range(draws):
        aucs = []
        for model_idx in range(scores.shape[0]):
            shuffled = rng.permutation(outcomes[model_idx])
            aucs.append(auc_columns(scores[model_idx], shuffled))
        null[draw] = np.nanmean(np.stack(aucs), axis=0)
    return (1 + (null >= observed[None, :]).sum(axis=0)) / (draws + 1)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values without an optional statsmodels dependency."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_sorted = np.empty_like(ranked)
    total = len(ranked)
    running_max = 0.0
    for rank, value in enumerate(ranked):
        adjusted = min(1.0, (total - rank) * value)
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = running_max
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def plot_model_holdouts(per_model_holdout: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    top = (
        per_model_holdout.sort_values(["model", "test_auc"], ascending=[True, False])
        .groupby("model", group_keys=False)
        .head(8)
        .copy()
    )
    model_names = top["model"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        len(model_names),
        1,
        figsize=(8.5, max(2.2, 1.7 * len(model_names))),
        sharex=True,
    )
    if len(model_names) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, model_names, strict=True):
        frame = top[top["model"].eq(model_name)].sort_values("test_auc")
        y = np.arange(len(frame))
        ax.scatter(frame["test_auc"], y, color="#286A8B", s=24)
        labels = [
            f"{row.word} ({'higher' if row.direction > 0 else 'lower'})"
            for row in frame.itertuples()
        ]
        ax.set_yticks(y, labels)
        ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(model_name, loc="left", fontsize=10)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("AUROC on untouched test items")
    axes[-1].set_xlim(0.25, 0.9)
    fig.suptitle("Per-model frozen full-vocabulary J-space indicators", y=0.995)
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positive-threshold", type=int, default=4)
    parser.add_argument("--discovery-fraction", type=float, default=0.5)
    parser.add_argument("--per-model-top-k", type=int, default=500)
    parser.add_argument("--per-model-holdout-top-k", type=int, default=40)
    parser.add_argument("--final-top-k", type=int, default=30)
    parser.add_argument("--minimum-models", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--permutation-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    models = [load_model(spec, args.positive_threshold) for spec in config["models"]]
    common_ids = sorted(
        set.intersection(
            *(set(model["items"]["original_row_idx"]) for model in models)
        )
    )
    outcome_matrix = np.stack(
        [
            model["items"]
            .set_index("original_row_idx")
            .loc[common_ids, "positive"]
            .to_numpy(dtype=int)
            for model in models
        ]
    )
    discovery_ids_array, test_ids_array = split_items(
        np.array(common_ids),
        outcome_matrix,
        args.discovery_fraction,
        args.seed,
    )
    discovery_ids = set(discovery_ids_array.tolist())
    test_ids = set(test_ids_array.tolist())

    screens = {
        model["name"]: screen_model(model, discovery_ids, args.chunk_size)
        for model in models
    }
    per_model_holdout = pd.concat(
        [
            evaluate_model_candidates(
                model,
                screens[model["name"]],
                test_ids,
                args.per_model_holdout_top_k,
            )
            for model in models
        ],
        ignore_index=True,
    )
    candidate_models: dict[str, set[str]] = {}
    for model_name, screen in screens.items():
        for word in screen.head(args.per_model_top_k)["word"]:
            candidate_models.setdefault(word, set()).add(model_name)
    candidates = sorted(
        word
        for word, names in candidate_models.items()
        if len(names) >= args.minimum_models
        and all(word in set(screens[model["name"]]["word"]) for model in models)
    )
    if not candidates:
        raise ValueError("No cross-model candidate words survived")

    discovery_raw = []
    for word in candidates:
        raw_aucs = [
            float(
                screens[model["name"]]
                .set_index("word")
                .loc[word, "raw_auc"]
            )
            for model in models
        ]
        raw_macro = float(np.mean(raw_aucs))
        direction = 1 if raw_macro >= 0.5 else -1
        discovery_raw.append(
            {
                "word": word,
                "direction": direction,
                "discovery_macro_auc": max(raw_macro, 1 - raw_macro),
                "models_selecting_word": len(candidate_models[word]),
            }
        )
    discovery = pd.DataFrame(discovery_raw).sort_values(
        "discovery_macro_auc", ascending=False
    )
    selected = discovery.head(args.final_top_k).copy()
    words = selected["word"].tolist()
    directions = dict(zip(selected["word"], selected["direction"], strict=True))

    test_scores, test_outcomes, included_models = candidate_scores(
        models,
        screens,
        test_ids,
        words,
        directions,
    )
    per_model_rows = []
    for model_idx, model_name in enumerate(included_models):
        auc = auc_columns(test_scores[model_idx], test_outcomes[model_idx])
        for word, value in zip(words, auc, strict=True):
            per_model_rows.append(
                {
                    "model": model_name,
                    "word": word,
                    "direction": directions[word],
                    "test_auc": value,
                    "test_n": test_scores.shape[1],
                    "test_positive_n": int(test_outcomes[model_idx].sum()),
                }
            )
    per_model = pd.DataFrame(per_model_rows)
    observed = (
        per_model.groupby("word")["test_auc"].mean().reindex(words).to_numpy()
    )
    low, high = bootstrap_macro_auc(
        test_scores,
        test_outcomes,
        args.bootstrap_draws,
        args.seed + 1,
    )
    p_values = permutation_p_values(
        test_scores,
        test_outcomes,
        observed,
        args.permutation_draws,
        args.seed + 2,
    )
    holm = holm_adjust(p_values)
    holdout = selected.copy()
    holdout["test_macro_auc"] = observed
    holdout["test_ci_low"] = low
    holdout["test_ci_high"] = high
    holdout["permutation_p_one_sided"] = p_values
    holdout["holm_p"] = holm

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, screen in screens.items():
        safe_name = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
        screen.to_csv(
            args.output_dir / f"{safe_name}_full_vocab_discovery.csv",
            index=False,
        )
    discovery.to_csv(args.output_dir / "cross_model_discovery.csv", index=False)
    holdout.to_csv(args.output_dir / "frozen_words_holdout.csv", index=False)
    per_model.to_csv(args.output_dir / "frozen_words_per_model.csv", index=False)
    per_model_holdout.to_csv(
        args.output_dir / "per_model_frozen_words_holdout.csv",
        index=False,
    )
    hparams = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "models": [model["name"] for model in models],
        "layers": {model["name"]: model["layer"] for model in models},
        "n_common_items": len(common_ids),
        "n_discovery_items": len(discovery_ids),
        "n_test_items": len(test_ids),
        "discovery_ids": sorted(discovery_ids),
        "test_ids": sorted(test_ids),
        "selection_note": (
            "Every vocabulary token was screened on discovery items only. "
            "Token strings, direction, and final candidates were frozen before test evaluation."
        ),
    }
    (args.output_dir / "hparams.json").write_text(
        json.dumps(hparams, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_frame = holdout.head(20).sort_values("test_macro_auc")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    y = np.arange(len(plot_frame))
    means = plot_frame["test_macro_auc"].to_numpy()
    ax.errorbar(
        means,
        y,
        xerr=np.vstack(
            [
                means - plot_frame["test_ci_low"].to_numpy(),
                plot_frame["test_ci_high"].to_numpy() - means,
            ]
        ),
        fmt="o",
        color="#286A8B",
        capsize=3,
    )
    labels = [
        f"{row.word} ({'higher' if row.direction > 0 else 'lower'})"
        for row in plot_frame.itertuples()
    ]
    ax.set_yticks(y, labels)
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlim(0.25, 0.85)
    ax.set_xlabel("Macro AUROC on untouched test items")
    ax.set_title("Frozen full-vocabulary J-space indicators across open models")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "full_vocabulary_holdout.png", dpi=220)
    fig.savefig(args.output_dir / "full_vocabulary_holdout.pdf")
    plt.close(fig)

    plot_model_holdouts(
        per_model_holdout,
        args.output_dir / "per_model_full_vocabulary_holdout.png",
        args.output_dir / "per_model_full_vocabulary_holdout.pdf",
    )


if __name__ == "__main__":
    main()
