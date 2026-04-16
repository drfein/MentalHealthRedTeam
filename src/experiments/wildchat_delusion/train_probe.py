"""
train_probe.py
==============
Train a linear probe on mean-pooled layer activations from a Llama model
to discriminate delusional from non-delusional user messages.

Loads the WildDelusion HuggingFace dataset, extracts hidden states at the
specified layer, trains a logistic regression probe via 5-fold stratified CV,
and saves the fitted probe to a JSON file for use by scan_corpus.py.

Usage:
  python train_probe.py
  python train_probe.py --layer 13 --data danielfein/WildDelusion --out_dir ./output
  python train_probe.py --model meta-llama/Llama-3.1-8B --pos_threshold 7
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_LENGTH = 256
BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear probe on Llama layer activations for delusional content."
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B",
                        help="HuggingFace model ID (default: meta-llama/Llama-3.1-8B).")
    parser.add_argument("--layer", type=int, default=13,
                        help="Transformer layer index to extract activations from (default: 13).")
    parser.add_argument("--data", default="danielfein/WildDelusion",
                        help="HuggingFace dataset ID (default: danielfein/WildDelusion).")
    parser.add_argument("--config", default="combined",
                        help="Dataset config/split name (default: combined).")
    parser.add_argument("--pos_field", default="gpt_score",
                        help="Dataset field used to determine positive label (default: gpt_score).")
    parser.add_argument("--pos_threshold", type=int, default=5,
                        help="Rows with pos_field >= this value are labeled positive (default: 5).")
    parser.add_argument("--cache_dir", default="./cache",
                        help="Directory for activation cache files (default: ./cache).")
    parser.add_argument("--out_dir", default="./output",
                        help="Directory to write probe JSON and plots (default: ./output).")
    return parser.parse_args()


@torch.no_grad()
def encode_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    layer: int,
) -> np.ndarray:
    """Mean-pool the hidden state at `layer` over non-padding tokens."""
    enc = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH,
    ).to(model.device)
    out = model(**enc, output_hidden_states=True)
    # hidden_states: tuple of (B, S, D), index 0 = embedding, 1..N = transformer layers
    h = out.hidden_states[layer].float()
    mask = enc["attention_mask"].float().unsqueeze(-1)
    pooled = (h * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled.cpu().numpy()


def extract_activations(
    texts: list[str],
    model_name: str,
    layer: int,
    cache_dir: Path,
) -> np.ndarray:
    """Return activation matrix (N, D), loading from cache if available."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    act_cache = cache_dir / f"{safe_name}_layer{layer}_acts.npy"

    if act_cache.exists():
        print(f"Loading cached activations from {act_cache}")
        return np.load(str(act_cache))

    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    all_acts = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="encoding"):
        batch = texts[i: i + BATCH_SIZE]
        all_acts.append(encode_batch(model, tokenizer, batch, layer))

    acts = np.concatenate(all_acts, axis=0)
    np.save(str(act_cache), acts)
    print(f"Saved activations {acts.shape} to {act_cache}")
    return acts


def save_probe(
    clf: LogisticRegression,
    scaler: StandardScaler,
    out_dir: Path,
    model_name: str,
    layer: int,
) -> Path:
    """Serialize probe direction, intercept, and scaler stats to JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    probe_path = out_dir / f"{safe_name}_layer{layer}_probe.json"
    probe_data = {
        "model": model_name,
        "layer": layer,
        "direction": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    probe_path.write_text(json.dumps(probe_data, indent=2))
    print(f"Probe saved to {probe_path}")
    return probe_path


def save_pca_plot(
    X: np.ndarray,
    labels: np.ndarray,
    cv_results: dict,
    layer: int,
    out_dir: Path,
) -> None:
    """Save a 2-component PCA scatter plot of the activation space."""
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X2[labels == 0, 0], X2[labels == 0, 1], alpha=0.15, s=8,
               c="steelblue", label=f"Non-positive (n={int((labels == 0).sum())})")
    ax.scatter(X2[labels == 1, 0], X2[labels == 1, 1], alpha=0.8, s=30,
               c="crimson", label=f"Positive (n={int(labels.sum())})")
    roc = cv_results["test_roc_auc"].mean()
    bal = cv_results["test_balanced_accuracy"].mean()
    ax.set_title(f"Layer {layer} activations PCA\nROC-AUC={roc:.3f}  Bal-acc={bal:.3f}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / f"probe_layer{layer}_pca.png"
    fig.savefig(str(plot_path), dpi=150)
    print(f"PCA plot saved to {plot_path}")


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)

    # Load dataset
    print(f"Loading dataset {args.data} ({args.config}) ...")
    ds = load_dataset(args.data, args.config)["train"]

    texts = [r["flagged_text"] for r in ds if r.get("flagged_text")]
    labels = np.array([
        1 if (r.get(args.pos_field) or 0) >= args.pos_threshold else 0
        for r in ds if r.get("flagged_text")
    ])
    print(f"  {len(texts)} texts | pos={labels.sum()} neg={(labels == 0).sum()}")

    # Extract activations (cached)
    acts = extract_activations(texts, args.model, args.layer, cache_dir)
    print(f"Activation matrix: {acts.shape}  dtype={acts.dtype}")

    # Fit scaler and train probe via 5-fold CV
    print("\nTraining linear probe (5-fold stratified CV) ...")
    scaler = StandardScaler()
    X = scaler.fit_transform(acts)

    clf = LogisticRegression(
        class_weight="balanced", max_iter=1000, C=0.1, solver="lbfgs"
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        clf, X, labels, cv=cv,
        scoring=["balanced_accuracy", "roc_auc", "average_precision"],
        return_train_score=False,
    )

    print(f"\n{'─' * 50}")
    print(f"  Balanced accuracy : "
          f"{cv_results['test_balanced_accuracy'].mean():.3f} "
          f"+/- {cv_results['test_balanced_accuracy'].std():.3f}")
    print(f"  ROC-AUC           : "
          f"{cv_results['test_roc_auc'].mean():.3f} "
          f"+/- {cv_results['test_roc_auc'].std():.3f}")
    print(f"  PR-AUC            : "
          f"{cv_results['test_average_precision'].mean():.3f} "
          f"+/- {cv_results['test_average_precision'].std():.3f}")
    print(f"{'─' * 50}")

    # Fit final probe on all data for saving and analysis
    clf.fit(X, labels)

    probe_path = save_probe(clf, scaler, out_dir, args.model, args.layer)

    # Score distribution
    scores_all = X @ clf.coef_[0] + clf.intercept_[0]
    pos_scores = scores_all[labels == 1]
    neg_scores = scores_all[labels == 0]
    print(f"\nProbe score distribution:")
    print(f"  Positive (n={len(pos_scores)}): "
          f"mean={pos_scores.mean():.2f}  std={pos_scores.std():.2f}  "
          f"min={pos_scores.min():.2f}  max={pos_scores.max():.2f}")
    print(f"  Negative (n={len(neg_scores)}): "
          f"mean={neg_scores.mean():.2f}  std={neg_scores.std():.2f}  "
          f"min={neg_scores.min():.2f}  max={neg_scores.max():.2f}")

    # Top false positives
    neg_idx = np.where(labels == 0)[0]
    top_fp = neg_idx[np.argsort(-scores_all[neg_idx])]
    print("\nTop-5 false positives (non-positive with highest probe score):")
    for rank, i in enumerate(top_fp[:5]):
        print(f"  [{rank + 1}] score={scores_all[i]:.2f}  {texts[i][:120]}")

    save_pca_plot(X, labels, cv_results, args.layer, out_dir)
    print(f"\nProbe written to {probe_path}")


if __name__ == "__main__":
    main()
