#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Keep the PDF and anonymous supplement byte-reproducible across rebuilds.
export SOURCE_DATE_EPOCH=1784592000
export FORCE_SOURCE_DATE=1

required_files=(
  paper/iclr2026/artifacts/mining_settings.json
  paper/iclr2026/artifacts/verification_summary.json
  paper/iclr2026/artifacts/release_characterization.json
  paper/iclr2026/artifacts/human_validation_metrics.csv
  paper/iclr2026/artifacts/release_manifest.json
  paper/iclr2026/artifacts/behavior/full_public/behavior_by_frame.csv
  paper/iclr2026/artifacts/behavior/full_public/paired_frame_contrasts.csv
  paper/iclr2026/artifacts/behavior/full_public/package_behavior_by_frame.csv
  paper/iclr2026/artifacts/behavior/full_public/package_paired_frame_contrasts.csv
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    printf 'Missing required artifact: %s\n' "$path" >&2
    exit 1
  fi
done

uv run python scripts/verify_paper_claims.py

uv run --extra paper python scripts/plot_wilddelusion_dataset_overview.py \
  --settings paper/iclr2026/artifacts/mining_settings.json \
  --verification-summary paper/iclr2026/artifacts/verification_summary.json \
  --release-characterization paper/iclr2026/artifacts/release_characterization.json \
  --release-manifest paper/iclr2026/artifacts/release_manifest.json \
  --output paper/iclr2026/figures/wilddelusion_dataset_overview.png
uv run --extra paper python scripts/plot_full_counterfactual_behavior.py \
  --results-dir paper/iclr2026/artifacts/behavior/full_public \
  --output paper/iclr2026/figures/full_behavior_main.png

(
  cd paper/iclr2026
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
)

mkdir -p output/pdf
cp paper/iclr2026/main.pdf output/pdf/wild_delusion_iclr2026.pdf
printf 'Built %s\n' "$ROOT/output/pdf/wild_delusion_iclr2026.pdf"
