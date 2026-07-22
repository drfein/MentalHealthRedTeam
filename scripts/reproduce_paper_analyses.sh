#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Recompute every aggregate and figure reported in the paper from saved raw
# generations, judge outputs, and J-space readouts. This script makes no API
# calls and does not run a language model, but its inputs are intentionally not
# committed because they include licensed or sensitive conversation text.
required_inputs=(
  results/response_analysis/10model_low_reasoning/latest_success_responses.jsonl
  results/generated_response_annotations/gpt-5.4-mini/combined_8_flags/response_annotation_matrix.csv
  results/response_analysis/10model_low_reasoning/fine_topics_40/fine_topic_assignments.csv
  results/response_analysis/10model_low_reasoning/fine_topics_40/simple_lda_report/simple_lda_topic_table.csv
  results/observed_assistant_responses/package_judgments_target_context.jsonl
  results/observed_all_turn_trajectories/package_judgments.jsonl
  results/observed_all_turn_user_trajectories/package_judgments.jsonl
  results/context_ablation/target_only_package_judgments.jsonl
  results/context_ablation/truncation_sweep_package_judgments.jsonl
  results/assistant_history_substitution/package_judgments.jsonl
  results/assistant_history_substitution/coherence_judgments.jsonl
  results/discovery_route_benchmark/historical/package_judgments.jsonl
  data/jspace/context_interventions_probe_free.jsonl
  results/jspace_context_interventions/qwen2_5_7b_it_probe_free/jspace_layer_readouts.jsonl
  results/jspace_context_interventions/qwen2_5_7b_it_probe_free/openai_package_bot_endorses_judged.jsonl
  results/jspace_controls/qwen2_5_7b_probe_free_misunderstanding/jspace_layer_readouts.jsonl
  results/jspace_controls/qwen2_5_7b_probe_free_misunderstanding/prompt_variants.csv
  results/jspace_controls/qwen2_5_7b_hard_negative_full/jspace_layer_readouts.jsonl
  data/verification/whitened_top3k_verified_gpt54mini.jsonl
)

for path in "${required_inputs[@]}"; do
  if [[ ! -f "$path" ]]; then
    printf 'Missing raw experiment input: %s\n' "$path" >&2
    exit 1
  fi
done

run_python() {
  uv run --extra paper python "$@"
}

run_python scripts/build_paper_multimodel_results.py
run_python scripts/build_observed_response_dataset.py
run_python scripts/analyze_all_turn_trajectories.py
run_python scripts/analyze_response_judge_context_sensitivity.py
run_python scripts/analyze_user_assistant_trajectories.py
run_python scripts/analyze_mini_model_hypothesis.py
run_python scripts/analyze_response_longitudinal.py
run_python scripts/analyze_context_ablation.py
run_python scripts/analyze_context_truncation.py
run_python scripts/analyze_assistant_history_substitution.py
run_python scripts/analyze_discovery_route_sensitivity.py

context_root=results/jspace_controls/qwen2_5_7b_probe_free_misunderstanding
run_python scripts/analyze_jspace_context_interventions.py \
  --readouts results/jspace_context_interventions/qwen2_5_7b_it_probe_free/jspace_layer_readouts.jsonl \
  --input data/jspace/context_interventions_probe_free.jsonl \
  --prompt-variants "$context_root/prompt_variants.csv" \
  --max-prompt-tokens 1024 \
  --out-dir "$context_root/context_effects_valid_1024" \
  --bootstrap-draws 10000 \
  --seed 20260711
run_python scripts/analyze_jspace_behavior_link.py \
  --judged-responses results/jspace_context_interventions/qwen2_5_7b_it_probe_free/openai_package_bot_endorses_judged.jsonl \
  --jspace-readouts "$context_root/context_effects_valid_1024/context_intervention_readouts.parquet" \
  --prompt-variants "$context_root/prompt_variants.csv" \
  --score-column openai_score \
  --positive-threshold 7 \
  --layer 26 \
  --out-dir "$context_root/behavior_valid_1024" \
  --bootstrap-draws 10000 \
  --seed 20260711
run_python scripts/analyze_jspace_endorsement_gate.py \
  --readouts "$context_root/jspace_layer_readouts.jsonl" \
  --judged-responses results/jspace_context_interventions/qwen2_5_7b_it_probe_free/openai_package_bot_endorses_judged.jsonl \
  --metadata data/jspace/context_interventions_probe_free.jsonl \
  --prompt-variants "$context_root/prompt_variants.csv" \
  --max-prompt-tokens 1024 \
  --out-dir "$context_root/gating_controls_misinformation" \
  --layer 26 \
  --primary-token misinformation \
  --score-column openai_score \
  --positive-threshold 7 \
  --cv-folds 10 \
  --bootstrap-draws 5000 \
  --permutation-draws 2000 \
  --seed 20260715
run_python scripts/analyze_jspace_hard_negative_control.py \
  --readouts results/jspace_controls/qwen2_5_7b_hard_negative_full/jspace_layer_readouts.jsonl \
  --metadata data/verification/whitened_top3k_verified_gpt54mini.jsonl \
  --prompt-variants results/jspace_controls/qwen2_5_7b_hard_negative_full/prompt_variants.csv \
  --max-prompt-tokens 1024 \
  --out-dir results/jspace_controls/qwen2_5_7b_hard_negative_full/analysis_misinformation_filtered \
  --layer 26 \
  --primary-token misinformation \
  --cv-folds 10 \
  --bootstrap-draws 5000 \
  --seed 20260715

copy_group() {
  local destination="$1"
  shift
  mkdir -p "$destination"
  cp "$@" "$destination/"
}

copy_group paper/iclr2026/artifacts/response_longitudinal \
  results/response_longitudinal/summary.json \
  results/response_longitudinal/observed_rates_by_platform.csv \
  results/response_longitudinal/observed_progress_bins.csv \
  results/response_longitudinal/generated_longitudinal_by_model.csv
cp results/response_longitudinal/response_longitudinal.pdf \
  paper/iclr2026/figures/response_longitudinal.pdf

copy_group paper/iclr2026/artifacts/all_turn_trajectories \
  results/observed_all_turn_trajectories/analysis_summary.json \
  results/observed_all_turn_trajectories/all_turn_progress_bins.csv
copy_group paper/iclr2026/artifacts/response_judge_context_sensitivity \
  results/response_judge_context_sensitivity/summary.json
copy_group paper/iclr2026/artifacts/user_assistant_trajectories \
  results/user_assistant_trajectories/summary.json \
  results/user_assistant_trajectories/trajectory_bins.csv
copy_group paper/iclr2026/artifacts/mini_model_hypothesis \
  results/mini_model_hypothesis/summary.json \
  results/mini_model_hypothesis/paired_comparisons.csv
copy_group paper/iclr2026/artifacts/context_ablation \
  results/context_ablation/summary.json \
  results/context_ablation/context_effect_by_model.csv \
  results/context_ablation/context_effect_by_length.csv \
  results/context_ablation/context_effect_by_history_type.csv
cp results/context_ablation/context_ablation.pdf \
  paper/iclr2026/figures/context_ablation.pdf
copy_group paper/iclr2026/artifacts/context_truncation \
  results/context_ablation/truncation_sweep/summary.json \
  results/context_ablation/truncation_sweep/context_truncation_rates.csv
cp results/context_ablation/truncation_sweep/context_truncation.pdf \
  paper/iclr2026/figures/context_truncation.pdf
copy_group paper/iclr2026/artifacts/assistant_history_substitution \
  results/assistant_history_substitution/analysis/summary.json \
  results/assistant_history_substitution/analysis/substitution_effect_by_model.csv
cp results/assistant_history_substitution/analysis/assistant_history_substitution.pdf \
  paper/iclr2026/figures/assistant_history_substitution.pdf
cp results/assistant_history_substitution/analysis/assistant_history_substitution.png \
  paper/iclr2026/figures/assistant_history_substitution.png
copy_group paper/iclr2026/artifacts/discovery_route_benchmark \
  results/discovery_route_benchmark/summary.json \
  results/discovery_route_benchmark/endorsement_by_discovery_route.csv

copy_group paper/iclr2026/artifacts/jspace/context_intervention \
  "$context_root/context_effects_valid_1024/paired_effects.csv" \
  "$context_root/context_effects_valid_1024/hparams.json" \
  "$context_root/behavior_valid_1024/behavior_paired_effects.csv"
copy_group paper/iclr2026/artifacts/jspace/endorsement_indicator \
  "$context_root/gating_controls_misinformation/summary.json" \
  "$context_root/gating_controls_misinformation/hparams.json" \
  "$context_root/gating_controls_misinformation/token_placebo_ranking.csv" \
  "$context_root/gating_controls_misinformation/adjusted_token_placebo_ranking.csv"
copy_group paper/iclr2026/artifacts/jspace/hard_negative_control \
  results/jspace_controls/qwen2_5_7b_hard_negative_full/analysis_misinformation_filtered/summary.json \
  results/jspace_controls/qwen2_5_7b_hard_negative_full/analysis_misinformation_filtered/hparams.json

uv run python scripts/verify_paper_claims.py
printf 'Refreshed paper aggregates from raw experiment outputs.\n'
