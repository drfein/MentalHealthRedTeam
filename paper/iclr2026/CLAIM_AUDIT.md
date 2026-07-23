# Manuscript claim audit

This file maps every quantitative result in the paper to the text-free aggregate
that is committed with the manuscript. `scripts/verify_paper_claims.py` checks
the exact values, cross-artifact invariants, and corresponding manuscript text.
`configs/paper_experiments.json` separately maps every experiment to its source
code, and `scripts/verify_paper_code_manifest.py` verifies that map.

## Dataset and verification

| Manuscript result | Committed source |
|---|---|
| Approximately 6.13M materialized user-message embeddings; 107/1,000 bootstrap positives; 200,000-row whitening fit | `artifacts/mining_settings.json` |
| 715 contextual-verifier inputs from 418 conversations; 436 positive, 271 negative, 8 uncertain | `artifacts/verification_summary.json` |
| 522 released targets from 321 conversations; 433 primary-route and 89 historical-route targets | `artifacts/combined_release_characterization.json`, `artifacts/combined_release_manifest.json` |
| Zero redistributed LMSYS rows and zero target-integrity errors | `artifacts/combined_release_integrity.json` |
| Earlier transfer audit: precision 35/38, specificity 26/29, sensitivity 35/79, AUROC 0.824 | `artifacts/human_validation_metrics.csv` |

## Ten-model response benchmark

| Manuscript result | Committed source |
|---|---|
| 4,316 successful public-cohort responses from 10 model snapshots, covering 433 targets and 232 conversations | `artifacts/multimodel/summary.json` |
| All eight SPIRALS totals and model-specific rates, including 289 strict endorsement flags | `artifacts/multimodel/summary.json`, `artifacts/multimodel/spirals_taxonomy_by_model.csv` |
| Fixed 40-topic LDA prevalence and model-specific topic assignments | `artifacts/multimodel/summary.json`, `artifacts/multimodel/lda_topics_by_model.csv` |
| Historical-route benchmark: 885 judged generations and 41 endorsements | `artifacts/discovery_route_benchmark/summary.json`, `artifacts/discovery_route_benchmark/endorsement_by_discovery_route.csv` |
| Four paired mini/flagship comparisons and pooled descriptive contrast | `artifacts/mini_model_hypothesis/summary.json`, `artifacts/mini_model_hypothesis/paired_comparisons.csv` |

## Observed replies and trajectories

| Manuscript result | Committed source |
|---|---|
| 295 recovered target-adjacent replies from 156 conversations; 171 strict flags; serving-platform counts | `artifacts/response_longitudinal/summary.json`, `artifacts/response_longitudinal/observed_rates_by_platform.csv` |
| 1,367 assistant turns in 27 repeated-target conversations; 667 strict flags; +28.1-point endpoint change | `artifacts/all_turn_trajectories/analysis_summary.json`, `artifacts/all_turn_trajectories/all_turn_progress_bins.csv` |
| Same-166-reply judge-context comparison: 122 target-only versus 132 context-supplied positives | `artifacts/response_judge_context_sensitivity/summary.json` |
| 1,365 aligned user/assistant pairs; user trajectory and assistant model adjusted for the contemporaneous user label | `artifacts/user_assistant_trajectories/summary.json`, `artifacts/user_assistant_trajectories/trajectory_bins.csv` |
| Controlled repeated-target response trajectory across 2,619 responses | `artifacts/response_longitudinal/summary.json`, `artifacts/response_longitudinal/generated_longitudinal_by_model.csv` |

## Behavioral context experiments

| Manuscript result | Committed source |
|---|---|
| 4,315 matched target-only/full-prefix pairs; 2.9% versus 6.7%; +3.78 points | `artifacts/context_ablation/summary.json` |
| Model, history-length, and history-role heterogeneity | `artifacts/context_ablation/context_effect_by_model.csv`, `context_effect_by_length.csv`, `context_effect_by_history_type.csv` |
| 222-target truncation sweep for GPT-4.1-mini and GPT-5.2 | `artifacts/context_truncation/summary.json`, `artifacts/context_truncation/context_truncation_rates.csv` |
| 94-conversation assistant-history intervention; 924 complete three-arm sets; 18.1% original versus 8.8% GPT-5.2-substituted | `artifacts/assistant_history_substitution/summary.json`, `artifacts/assistant_history_substitution/substitution_effect_by_model.csv` |
| Replacement coherence: 70/94 pass at 7/10; coherent-only substitution effect $-8.55$ points | `artifacts/assistant_history_substitution/summary.json` |

## Exploratory J-space audit

| Manuscript result | Committed source |
|---|---|
| 417 paired valid targets; validating-minus-reality-testing shifts for misinformation, reality-testing, and falsity-concern coordinates | `artifacts/jspace/context_intervention/paired_effects.csv` |
| Behavioral effect of the same intervention: -0.24 points [-2.40, 1.92] | `artifacts/jspace/context_intervention/behavior_paired_effects.csv` |
| Grouped 10-fold AUROC 0.596 to 0.732; +0.135 [0.028, 0.249]; 32 behaviorally varying items; predefined epistemic/placebo comparison | `artifacts/jspace/endorsement_indicator/summary.json`, `token_placebo_ranking.csv` |
| Contextual hard-negative incremental AUROC +0.019 [-0.005, 0.044] | `artifacts/jspace/hard_negative_control/summary.json` |
| Four-model frozen semantic-panel macro AUROC 0.603 [0.546, 0.657]; inverse-misinformation-rank macro AUROC 0.601 [0.544, 0.656] | `artifacts/jspace/cross_model_replication/indicator_replication_macro.csv` |
| Semantic-minus-optimized-placebo macro contrast +0.052 [-0.019, 0.121] | `artifacts/jspace/cross_model_replication/semantic_placebo_contrasts.csv` |
| Per-model same-model outcome rates and indicator AUROCs | `artifacts/jspace/cross_model_replication/performance.csv`, `indicator_per_model.csv` |

## Evidence boundaries

- Raw generations, judge rationales, and source conversation text are not
  committed. They may contain sensitive or license-restricted text. The public
  repository commits code, settings, text-free aggregates, and figures.
- The zero-network rebuild verifies consistency between the committed evidence
  bundle and manuscript; it does not recreate proprietary API outputs.
- Rows are target turns, not independent users or conversations. Reported
  uncertainty follows the conversation-clustering rule specified by each
  experiment.
- Retrieval sampling estimates precision in rank ranges, not corpus prevalence
  or recall.
- The final 522-row release still requires the independently double-coded audit
  marked as pending in the manuscript.
