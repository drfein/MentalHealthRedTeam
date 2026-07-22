# Reproducing the manuscript

The repository provides three reproducibility levels. They are intentionally
separate so that a reviewer never needs credentials merely to verify the paper.

## 1. Zero-network paper verification

From a fresh clone with `uv`, `pdflatex`, and `bibtex` installed:

```bash
uv sync --extra paper
scripts/rebuild_paper.sh
```

This validates the code manifest, checks every numerical manuscript claim
against committed text-free aggregates, regenerates the dataset overview, and
compiles the 17-page PDF. It does not access source conversations, call an API,
download a model, or require a GPU.

Expected output:

```text
output/pdf/wild_delusion_iclr2026.pdf
```

## 2. Recompute analyses from saved raw outputs

On the archival workspace containing ignored `data/` and `results/` inputs:

```bash
scripts/reproduce_paper_analyses.sh
scripts/rebuild_paper.sh
```

The first command recomputes every reported aggregate and figure from saved
generations, package-judge outputs, conversation reconstructions, and J-space
readouts. It makes no API calls and runs no language model. It then copies only
text-free aggregates into `paper/iclr2026/artifacts/`.

## 3. Regenerate API/GPU outputs

`EXPERIMENTS.md` gives the ordered commands, exact model IDs, package pin,
reasoning effort, thresholds, random seeds, context windows, and output paths
for data mining, response generation, judging, LDA, trajectories, context
ablations, and the Qwen Jacobian-lens audit.
The context recipes include both prefix ablation/truncation and the
three-arm assistant-history substitution experiment.

The machine-readable coverage map is
`configs/paper_experiments.json`. Validate it with:

```bash
uv run python scripts/verify_paper_code_manifest.py
```

## Evidence boundaries

- `CLAIM_AUDIT.md` maps each number to its committed aggregate.
- `DATASET_HUMAN_AUDIT_PROTOCOL.md` locks the pending final-release audit.
- `HUMAN_AUDIT_PROTOCOL.md` locks the pending assistant-response audit.
- Proprietary API outputs cannot be bitwise regenerated after a provider changes
  serving infrastructure, even when a dated model ID is available.
- Upstream wild-chat snapshots and LMSYS redistribution terms are external
  boundaries. The public release contains no LMSYS conversation text.
