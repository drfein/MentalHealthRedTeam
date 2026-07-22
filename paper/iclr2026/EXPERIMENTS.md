# Paper experiment recipes

This is the canonical execution map for every experiment reported in
`main.tex`. Commands run from the repository root. Raw `data/` and `results/`
files are ignored because they can contain sensitive or license-restricted
conversation text; settings, source code, text-free aggregates, and figures are
committed.

## Environment and fixed definitions

```bash
uv sync --extra dev --extra paper
export OPENAI_API_KEY=...
export HF_TOKEN=...
```

The SPIRALS package is pinned in `pyproject.toml` to
`jlcmoore/llm-delusions-annotations@3ea8d2117e55099a61feee1c762c6ee0c32a162b`.
Message and assistant endorsement use that package's exact prompts and score
cutoff of 7. Conversation verification uses GPT-5.4-mini with low reasoning.

## 1. Dataset construction and verification

The main mining route is the ordered CLI pipeline in the root `README.md`:

```bash
wild-delusion-miner generate-synthetic
wild-delusion-miner extract-users
wild-delusion-miner prepare-embedding-batches
wild-delusion-miner submit-embedding-batches
wild-delusion-miner poll-embedding-batches --wait
wild-delusion-miner download-embedding-outputs
wild-delusion-miner materialize-embeddings
wild-delusion-miner retrieve-initial
wild-delusion-miner make-calibration
wild-delusion-miner annotate-initial-above-threshold
wild-delusion-miner retrieve-from-true-positives
wild-delusion-miner annotate-true-positive-pass
wild-delusion-miner collect-verification-conversations
wild-delusion-miner verify-conversations
```

`configs/default.yaml` stores the embedding model, Batch API shard sizes,
retrieval cutoffs, whitening sample size, and output paths. The historical route
and combined release use:

```bash
uv run python scripts/build_legacy_expansion_candidates.py
wild-delusion-miner annotate-rerank-canonical-direct \
  --input-path data/legacy_expansion/candidates.jsonl \
  --output-path data/legacy_expansion/message_annotations_gpt55.jsonl \
  --model gpt-5.5
uv run python scripts/filter_positive_annotations.py \
  --input data/legacy_expansion/message_annotations_gpt55.jsonl \
  --output data/legacy_expansion/message_positive_gpt55.jsonl \
  --cutoff 7
wild-delusion-miner verify-conversations \
  --conversations-path data/legacy_expansion/message_positive_gpt55.jsonl \
  --out-path data/legacy_expansion/context_verified.jsonl \
  --model gpt-5.4-mini
uv run python scripts/build_wilddelusion_combined_release.py
uv run python scripts/verify_wilddelusion_release.py \
  --release data/releases/WildDelusionCombined/train.parquet \
  --exclude-source lmsys_chat_1m
```

## 2. Ten-model responses, SPIRALS flags, and LDA

The dated generation set is:

```text
gpt-3.5-turbo-0125
gpt-4-turbo-2024-04-09
gpt-4o-2024-05-13
gpt-4o-mini-2024-07-18
o1-2024-12-17
o3-mini-2025-01-31
gpt-4.1-mini-2025-04-14
gpt-5-mini-2025-08-07
gpt-5.2-2025-12-11
gpt-5.5-2026-04-23
```

Every model receives the complete source prefix through the target user turn,
the fixed `post_delusion_assistant_response_v1` system prompt, low reasoning,
and at most 2,000 output tokens. Generation is resumable:

```bash
wild-delusion-miner generate-post-delusion-responses \
  --input-path data/releases/WildDelusionVerified/train.parquet \
  --out-path data/generations/post_delusion_openai_responses_10model_low_reasoning.jsonl \
  --max-workers 8 \
  --max-output-tokens 2000 \
  --reasoning-effort low \
  --model gpt-3.5-turbo-0125 \
  --model gpt-4-turbo-2024-04-09 \
  --model gpt-4o-2024-05-13 \
  --model gpt-4o-mini-2024-07-18 \
  --model o1-2024-12-17 \
  --model o3-mini-2025-01-31 \
  --model gpt-4.1-mini-2025-04-14 \
  --model gpt-5-mini-2025-08-07 \
  --model gpt-5.2-2025-12-11 \
  --model gpt-5.5-2026-04-23
```

Run all eight pinned SPIRALS assistant annotations with GPT-5.4-mini and low
reasoning using `scripts/run_generated_response_annotations.py`; its
`--annotation-id` is run once for each of the eight IDs listed in
`scripts/build_paper_multimodel_results.py`. Fit the reported LDA model with:

```bash
uv run --extra paper python scripts/run_fine_response_topics.py \
  --input-path results/response_analysis/10model_low_reasoning/latest_success_responses.jsonl \
  --out-dir results/response_analysis/10model_low_reasoning/fine_topics_40 \
  --topic-count 40 \
  --random-state 17 \
  --label-model gpt-5.4-mini \
  --label-reasoning-effort low \
  --max-features 14000 \
  --min-df 3 \
  --max-df 0.8
```

`scripts/build_paper_multimodel_results.py` performs the public-cohort join,
removes the 30 generations associated with the three excluded LMSYS targets,
and produces Figure 1, Figure 2, and the model/taxonomy/topic aggregates.

## 3. Observed replies and trajectories

```bash
uv run --extra paper python scripts/build_observed_response_dataset.py
uv run --extra paper python scripts/build_all_turn_trajectory_inputs.py \
  --context-window 10
uv run python scripts/judge_generated_responses_with_package.py \
  --input results/observed_all_turn_trajectories/responses.jsonl \
  --prompts results/observed_all_turn_trajectories/prompts.jsonl \
  --output results/observed_all_turn_trajectories/package_judgments.jsonl \
  --model gpt-5.4-mini --reasoning-effort low --concurrency 40
uv run --extra paper python scripts/build_all_turn_user_inputs.py
uv run python scripts/judge_user_turns_with_package.py \
  --input results/observed_all_turn_user_trajectories/responses.jsonl \
  --prompts results/observed_all_turn_user_trajectories/prompts.jsonl \
  --output results/observed_all_turn_user_trajectories/package_judgments.jsonl \
  --model gpt-5.5 --reasoning-effort low --concurrency 40
```

The deterministic analyses are
`analyze_all_turn_trajectories.py`,
`analyze_response_judge_context_sensitivity.py`,
`analyze_user_assistant_trajectories.py`, and
`analyze_response_longitudinal.py`. Every bootstrap uses 10,000
source-conversation draws and seed `20260722`; turn-level regressions use
conversation fixed effects and small-sample-corrected clustered covariance.

## 4. Context ablation and truncation sweep

`build_context_ablation_inputs.py` removes every source message before the
target. The same ten model snapshots and generation settings are then applied.
`prepare_context_ablation_judgments.py` and
`judge_generated_responses_with_package.py` build and score the matched pairs.
`analyze_context_ablation.py` uses 10,000 conversation-cluster bootstrap draws
with seed `20260722`.

The post-hoc dose-response subset is built with:

```bash
uv run python scripts/build_context_truncation_inputs.py \
  --keep-prior 1 3 7 \
  --minimum-full-prior 13
```

Only GPT-4.1-mini and GPT-5.2 are generated for this sweep. The corresponding
preparation and analysis scripts are
`prepare_context_truncation_judgments.py` and
`analyze_context_truncation.py`.

## 5. Discovery-route sensitivity

```bash
uv run python scripts/build_discovery_split_benchmark.py \
  --discovery-split legacy_probe_gpt52
```

Generate the same ten-model continuations, then use
`prepare_discovery_route_judgments.py`, the exact package judge, and
`analyze_discovery_route_sensitivity.py`. The route comparison uses 10,000
independent route-specific conversation bootstrap draws and seed `20260722`.

## 6. Qwen J-space audit

Install the pinned CUDA environment:

```bash
uv sync --extra jspace --extra paper
```

Build three context arms per target, decode the public Qwen2.5-7B-Instruct
Jacobian lens at layer 26, and generate same-model replies:

```bash
uv run python scripts/build_jspace_context_interventions.py \
  --output data/jspace/context_interventions_probe_free.jsonl
uv run --extra jspace python scripts/run_jspace_context_ablation.py \
  --input data/jspace/context_interventions_probe_free.jsonl \
  --dataset-id '' \
  --out-dir results/jspace_controls/qwen2_5_7b_probe_free_misunderstanding \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --lens-repo neuronpedia/jacobian-lens \
  --lens-file qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt \
  --max-rows 1308 \
  --context-message-counts -1 \
  --layers 26 \
  --top-k 100 \
  --max-seq-len 1024 \
  --no-4bit
uv run --extra jspace python scripts/generate_context_intervention_responses.py \
  --input data/jspace/context_interventions_probe_free.jsonl \
  --output results/jspace_context_interventions/qwen2_5_7b_it_probe_free/generated_responses.jsonl \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --max-input-tokens 1024 \
  --max-new-tokens 192
```

Judge the generated replies with the same exact package prompt and
GPT-5.4-mini/low setup. The four deterministic J-space analyses are:

```text
scripts/analyze_jspace_context_interventions.py
scripts/analyze_jspace_behavior_link.py
scripts/analyze_jspace_endorsement_gate.py
scripts/analyze_jspace_hard_negative_control.py
```

Their exact paths and parameters are executable in
`scripts/reproduce_paper_analyses.sh`. The primary indicator is the layer-26
`misinformation` maximum logit; grouped CV has 10 folds; bootstrap and
permutation seeds are `20260715`. The hard negatives passed the message judge
but failed contextual verification.

The local gate metadata corresponds exactly to Hugging Face revision
`6cc1293e352eb1ed44a0d38ba4a2bff1a5774a97` of
`danielfein/WildDelusionVerified`. The local archival input is preferred so a
mutable dataset head cannot change covariate alignment.

## Final deterministic refresh

After raw outputs exist:

```bash
scripts/reproduce_paper_analyses.sh
scripts/rebuild_paper.sh
```
