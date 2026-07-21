# Submission readiness gates

This checklist separates completed infrastructure from evidence that still
changes the paper's admissible claims. Passing code tests or compiling the PDF
does not satisfy the empirical gates.

## Completed

- [x] Dataset-first title, abstract, contribution order, Figure 1, discussion,
  and conclusion.
- [x] Public 433-target-turn release from 232 source conversations with a detailed
  dataset card, intended-use limits, explicit source-license filtering, and a
  reference next-response protocol.
- [x] Exact release-integrity verification against authoritative pipeline rows:
  433 unique rows, no conversation or target mismatches, canonical SHA-256
  `dd34ec93a5fae80e4d9b82017940342206e7cd849dc03cbb9e74e822fadc94dc`.
- [x] Claim-to-artifact audit, deterministic analysis scripts, dependency lock,
  passing tests, clean LaTeX build, and visual PDF inspection.
- [x] Retrieval enrichment, contextual exclusion, source imbalance, privacy,
  and non-prevalence limitations stated explicitly.
- [x] Full 433-target, eight-frame behavioral experiment completed; uncertainty
  clusters all 232 source conversations and every paired contrast is saved.
- [x] Exact SPIRALS direct-response endpoint run on all 433 responses; its
  materially lower absolute rate is reported as endpoint sensitivity.
- [x] J-space removed from the abstract, contributions, main figure, results,
  discussion, and conclusion; the controlled null is appendix-only.
- [x] Source-conversation duplication audited; benchmark uncertainty changed to
  source-conversation cluster bootstrap.
- [x] Original J-space split leakage disclosed; main association replaced with
  a post hoc 67-turn, 62-conversation disjoint sensitivity analysis whose
  primary interval includes chance.

## Gate A: final-release human precision

**Required for a dataset-centered submission.**

Complete two independent reviews of the locked 100-row simple-random sample in
`results/wilddelusion_release_human_audit/`. Report strict precision with a 95%
Wilson interval and inter-rater agreement.

Decision rule:

- Strict precision at least 0.85 with no major systematic exclusion category:
  retain "precision-oriented benchmark" and replace the non-random transfer
  audit in the abstract and Figure 1 with the final-release estimate.
- Strict precision 0.70--0.85: describe the artifact as an LLM-enriched candidate
  benchmark, publish human labels, and make adjudication part of evaluation.
- Strict precision below 0.70: do not headline the 433 rows as verified cases;
  filter or relabel the release before submission.

These thresholds are manuscript decision rules, not hypothesis tests.

## Gate B: human assistant-response endpoint

**Required to treat the framing-aware behavioral rate as validated rather than
judge-defined.**

Independently review a locked sample spanning all automatic positives and a
pre-specified random sample of automatic negatives across the eight frames.
Report weighted frame rates, the direct-versus-reported-belief contrast,
inter-rater agreement, and category-level disagreement. Reviewers must be blind
to model-judge scores and to the appendix J-space values.

Decision rule:

- Human labels preserve direct assertion as the highest-rate frame and the
  weighted direct-versus-reported-belief interval excludes zero: retain the
  paired framing result as primary, led by the human-calibrated estimate.
- Ordering is preserved but uncertainty is broad: retain the automatic paired
  result as a benchmark demonstration and describe human evidence as supportive.
- Ordering reverses or systematic rubric errors explain the contrast: remove the
  behavioral headline and retain the dataset/resource contribution only.

## Gate C: model-family scope

**Resolved conservatively.** The behavioral experiment is explicitly
single-model. J-space is appendix-only and labeled a negative result. A future
model-family replication must freeze its endpoint before inspecting outcomes
and report null or reversed results.

## Final preflight

- [ ] Replace pending audit language using locked human-analysis outputs.
- [ ] Re-run `CLAIM_AUDIT.md` against every number in the final manuscript.
- [ ] Confirm the anonymous repository/release policy for the target venue.
- [ ] Re-check the current ICLR page limit, ethics requirements, and disclosure
  policy immediately before submission.
- [ ] Compile from a clean environment and visually inspect every page.

Current status: technically reproducible with a complete full-benchmark
behavioral analysis, but not empirically submission-ready until Gate A and Gate
B are complete. J-space no longer gates any main-paper claim.
