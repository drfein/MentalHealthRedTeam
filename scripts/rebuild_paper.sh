#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

required_files=(
  paper/iclr2026/artifacts/mining_settings.json
  paper/iclr2026/artifacts/verification_summary.json
  paper/iclr2026/artifacts/human_validation_metrics.csv
  paper/iclr2026/artifacts/release_manifest.json
  paper/iclr2026/artifacts/jspace/conversation_disjoint_correction/behavior_by_frame.csv
  paper/iclr2026/artifacts/jspace/conversation_disjoint_correction/endpoint_performance.csv
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    printf 'Missing required artifact: %s\n' "$path" >&2
    exit 1
  fi
done

uv run --extra paper python scripts/plot_wilddelusion_dataset_overview.py \
  --settings paper/iclr2026/artifacts/mining_settings.json \
  --verification-summary paper/iclr2026/artifacts/verification_summary.json \
  --human-validation paper/iclr2026/artifacts/human_validation_metrics.csv \
  --release-manifest paper/iclr2026/artifacts/release_manifest.json \
  --output paper/iclr2026/figures/wilddelusion_dataset_overview.png
uv run --extra paper python scripts/plot_jspace_paper_main_figure_v2.py \
  --results-root paper/iclr2026/artifacts/jspace \
  --output paper/iclr2026/figures/heldout_main_v2.png

(
  cd paper/iclr2026
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
)

mkdir -p output/pdf
cp paper/iclr2026/main.pdf output/pdf/wild_delusion_jspace_iclr2026.pdf
printf 'Built %s\n' "$ROOT/output/pdf/wild_delusion_jspace_iclr2026.pdf"
