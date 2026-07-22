# Manuscript claim audit

This file maps the paper's quantitative claims to authoritative saved artifacts.
It is intended to prevent prose revisions from drifting away from the analyzed
data. `scripts/verify_paper_claims.py` checks the listed aggregate inputs,
cross-artifact invariants, generation and judge settings, confidence intervals,
paired-test outputs, and corresponding manuscript strings during every full
paper rebuild. Its latest machine-readable result is
`artifacts/claim_verification.json`.

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
| Combined release | 522 target turns from 321 source conversations; 433 whitened-embedding and 89 legacy-probe targets | `artifacts/combined_release_characterization.json`, `artifacts/combined_release_manifest.json` |
| Hosted combined integrity | 522 rows, zero LMSYS rows, zero target errors; parquet SHA-256 `4aa19e12...ab3` | `artifacts/combined_release_integrity.json` |
| Earlier human transfer audit | strict-rule precision 35/38 = 92.1% [79.2, 97.3] | `artifacts/human_validation_metrics.csv` |
| Counterfactual artifact | 433 x 8 = 3,464 public responses | `artifacts/behavior/full_public/hparams.json` plus private response cache |
| Direct-assertion endorsement | 75/433 (17.3%), source-conversation cluster CI [13.1, 22.4] | `artifacts/behavior/full_public/behavior_by_frame.csv` |
| Conversation-macro direct endorsement | 17.2%; every other frame at most 1.8% | same `behavior_by_frame.csv` |
| Smallest paired direct-versus-alternative effect | Direct vs reported belief: +14.3 points [10.5, 18.8], Holm-adjusted exact source-conversation sign-flip p=5.5e-13 | `artifacts/behavior/full_public/paired_frame_contrasts.csv` |
| Exact package frame ordering | Direct 20/433 (4.6%) [2.5, 7.1], translation 4/433, every other frame at most 1/433 | `artifacts/behavior/full_public/package_behavior_by_frame.csv` |
| Smallest exact-package paired effect | Direct vs translation: +3.7 points [1.5, 6.2], Holm-adjusted exact source-conversation sign-flip p=.0022 | `artifacts/behavior/full_public/package_paired_frame_contrasts.csv` |
| Direct-response rubric overlap | 19 shared positives, 56 framing-only, 1 package-only | `artifacts/behavior/full_public/endpoint_overlap_by_frame.csv` |
| Source-stratified directional check | Direct assertion is highest in ShareChat--ChatGPT (63/356), ShareChat--Grok (7/33), and WildChat (5/44) | `artifacts/behavior/full_public/behavior_by_source_and_frame.csv` |

Interpretive boundaries are equally important:

- The retrieval ablation estimates precision in sampled rank buckets, not recall
  or corpus prevalence.
- The 108-case manual transfer audit is non-random and predates the final
  retrieval pool; it provides limited supporting evidence about verifier
  transfer but does not convert all 433 rows into human-adjudicated gold labels.
  The decisions were not independently double-coded, so no inter-rater estimate
  is available.
- Rows are target turns, not independent conversations. All benchmark
  uncertainty must cluster by `(source, conversation_id)`.
- The primary behavioral analysis uses all 433 targets from the whitened-
  embedding discovery cohort, not the 89-row legacy expansion. It has no
  outcome-selected train/evaluation split. Its uncertainty clusters the 232
  source conversations.
- The framing-aware and exact package judges preserve the direct-assertion
  ordering but produce materially different absolute rates; 17.3% is not an
  endpoint-free failure rate. The exact package is not framing-aware, so its
  translation positives are not interpreted as real-world endorsement.
