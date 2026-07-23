#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# This is the reviewer-facing, zero-network rebuild. It consumes only committed,
# text-free aggregate artifacts and committed figures. Use
# scripts/reproduce_paper_analyses.sh to refresh aggregates from raw experiment
# outputs before running this command.
export SOURCE_DATE_EPOCH=1784592000
export FORCE_SOURCE_DATE=1

required_files=(
  paper/iclr2026/main.tex
  paper/iclr2026/references.bib
  paper/iclr2026/artifacts/mining_settings.json
  paper/iclr2026/artifacts/verification_summary.json
  paper/iclr2026/artifacts/combined_release_characterization.json
  paper/iclr2026/artifacts/combined_release_manifest.json
  paper/iclr2026/artifacts/combined_release_integrity.json
  paper/iclr2026/artifacts/human_validation_metrics.csv
  paper/iclr2026/artifacts/multimodel/summary.json
  paper/iclr2026/artifacts/response_longitudinal/summary.json
  paper/iclr2026/artifacts/all_turn_trajectories/analysis_summary.json
  paper/iclr2026/artifacts/response_judge_context_sensitivity/summary.json
  paper/iclr2026/artifacts/user_assistant_trajectories/summary.json
  paper/iclr2026/artifacts/context_ablation/summary.json
  paper/iclr2026/artifacts/context_truncation/summary.json
  paper/iclr2026/artifacts/assistant_history_substitution/summary.json
  paper/iclr2026/artifacts/assistant_history_substitution/substitution_effect_by_model.csv
  paper/iclr2026/artifacts/mini_model_hypothesis/summary.json
  paper/iclr2026/artifacts/discovery_route_benchmark/summary.json
  paper/iclr2026/artifacts/jspace/context_intervention/paired_effects.csv
  paper/iclr2026/artifacts/jspace/context_intervention/behavior_paired_effects.csv
  paper/iclr2026/artifacts/jspace/endorsement_indicator/summary.json
  paper/iclr2026/artifacts/jspace/hard_negative_control/summary.json
  paper/iclr2026/artifacts/jspace/cross_model_replication/performance.csv
  paper/iclr2026/artifacts/jspace/cross_model_replication/indicator_per_model.csv
  paper/iclr2026/artifacts/jspace/cross_model_replication/indicator_replication_macro.csv
  paper/iclr2026/artifacts/jspace/cross_model_replication/semantic_placebo_contrasts.csv
  paper/iclr2026/artifacts/jspace/cross_model_replication/hparams.json
  paper/iclr2026/figures/wilddelusion_dataset_overview.png
  paper/iclr2026/figures/multimodel_spirals_taxonomy.pdf
  paper/iclr2026/figures/multimodel_lda_topics.pdf
  paper/iclr2026/figures/response_longitudinal.pdf
  paper/iclr2026/figures/context_ablation.pdf
  paper/iclr2026/figures/context_truncation.pdf
  paper/iclr2026/figures/assistant_history_substitution.pdf
  paper/iclr2026/figures/jspace_cross_model_indicators.pdf
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    printf 'Missing committed paper input: %s\n' "$path" >&2
    exit 1
  fi
done

command -v uv >/dev/null || { printf 'uv is required\n' >&2; exit 1; }
command -v pdflatex >/dev/null || { printf 'pdflatex is required\n' >&2; exit 1; }
command -v bibtex >/dev/null || { printf 'bibtex is required\n' >&2; exit 1; }

uv run python scripts/verify_paper_code_manifest.py
uv run python scripts/verify_paper_claims.py

# Figure 1 is regenerated because every input is a committed aggregate. The
# remaining figures are committed outputs whose source scripts are enumerated in
# paper/iclr2026/EXPERIMENTS.md.
uv run --extra paper python scripts/plot_wilddelusion_dataset_overview.py \
  --settings paper/iclr2026/artifacts/mining_settings.json \
  --verification-summary paper/iclr2026/artifacts/verification_summary.json \
  --release-characterization paper/iclr2026/artifacts/combined_release_characterization.json \
  --release-manifest paper/iclr2026/artifacts/combined_release_manifest.json \
  --output paper/iclr2026/figures/wilddelusion_dataset_overview.png

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
