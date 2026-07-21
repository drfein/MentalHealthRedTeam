# Manuscript claim audit

This file maps the paper's quantitative claims to authoritative saved artifacts.
It is intended to prevent prose revisions from drifting away from the analyzed
data.

| Claim | Value in manuscript | Authoritative artifact |
|---|---:|---|
| Materialized embedding corpus | about 6.13M user messages | `artifacts/mining_settings.json` |
| Top-1k bootstrap positives | 107/1,000 | `artifacts/mining_settings.json` |
| Top-500 whitened retrieval hit rate | 12/30 (40.0%) | `artifacts/retrieval_ablation_results.csv` |
| Contextually verified set | 436 positive target turns, 271 negative, 8 uncertain | `artifacts/verification_summary.json` |
| Public release unit count | 433 target turns from 232 source conversations; 62 multi-target conversations; maximum 39 targets/conversation | `artifacts/release_characterization.json` |
| License exclusion | 3 verified LMSYS-Chat-1M targets excluded from redistribution and reported experiments | `artifacts/release_manifest.json` |
| Release characteristics | median 21 stored messages; target index 13; target length 103 words | `artifacts/release_characterization.json` |
| Hosted release integrity | 433/433 exact rows; canonical SHA-256 `dd34ec93...4dc` | `artifacts/release_integrity.json` |
| Earlier human transfer audit | strict-rule precision 35/38 = 92.1% [79.2, 97.3] | `artifacts/human_validation_metrics.csv` |
| Public discovery/evaluation split | 289/144 target turns; 36 source conversations overlap | `artifacts/jspace/conversation_disjoint_correction/split_integrity.json` |
| Conversation-disjoint sensitivity set | 67 target turns from 62 source conversations absent from discovery | same correction `split_integrity.json` |
| Counterfactual artifact | 144 x 8 = 1,152 public responses per frozen layer | correction `split_integrity.json` plus private counterfactual artifacts |
| Direct-assertion endorsement | 23/144 (16.0%), source-conversation cluster CI [8.4, 24.6] | correction `behavior_by_frame.csv` |
| Conversation-macro direct endorsement | 13.7%; every other frame at most 1.0% | correction `behavior_by_frame.csv` |
| Smallest paired direct-versus-alternative effect | Direct vs reported belief: +13.9 points [6.8, 21.7], Holm-adjusted exact McNemar p=1.1e-5 | correction `behavior_paired_contrasts.csv` |
| Corrected frozen indicator | AUROC 0.687 [0.475, 0.872], 10/67 positives | correction `endpoint_performance.csv` |
| Corrected exact-package sensitivity | AUROC 0.833 [0.686, 0.967], 4/67 positives | correction `endpoint_performance.csv` |
| Public benchmark vs true/neutral framing delta | 1.219 vs 1.206 logits; Welch p=.955 | `artifacts/jspace/register_control.json` |

Interpretive boundaries are equally important:

- The retrieval ablation estimates precision in sampled rank buckets, not recall
  or corpus prevalence.
- The 108-case manual transfer audit is non-random and predates the final
  retrieval pool; it supports a precision-oriented release but does not convert
  all 433 rows into human-adjudicated gold labels. The decisions were not
  independently double-coded, so no inter-rater estimate is available.
- Rows are target turns, not independent conversations. All benchmark
  uncertainty must cluster by `(source, conversation_id)`.
- The original J-space split leaked 36 source conversations. The 67-turn
  metadata-only correction is post hoc, the primary interval includes chance,
  and the package endpoint has four positives; this is sensitivity evidence,
  not a confirmatory result.
- The ten-model series is descriptive; release date is confounded with model
  family, size, training, and API behavior.
- The layer-26 indicator was frozen before the evaluation counterfactual readouts;
  the layer-12 versus layer-26 sign reversal is exploratory.
- The true/neutral control rules out a delusion-specific interpretation of the
  global direct-versus-question shift.
- Predictive AUROC supports at most an observational indicator, not a
  semantically faithful or causal gate.
