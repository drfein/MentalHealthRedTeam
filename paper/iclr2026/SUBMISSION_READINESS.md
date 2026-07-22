# Submission readiness gates

## Double-blind and policy checks

- [x] Main manuscript contains no author-identifying repository or dataset URL.
- [x] Author block and PDF metadata are anonymous.
- [x] Main text ends on page 7, within the ICLR 2026 nine-page submission limit.
- [x] Appendix includes the required disclosure of material LLM use.
- [x] Deterministic anonymized supplement built at
  `output/submission/wilddelusion_anonymous_supplement.zip`; an independent
  extraction verified 433 release rows, 3,464 rows in each behavioral artifact,
  no LMSYS text, no Git history, and no blocked identifiers or credentials.
- [ ] Audit the final OpenReview PDF and supplementary archive immediately before
  submission; identity disclosure in either artifact is desk-rejectable.

This checklist separates completed infrastructure from evidence that still
changes the paper's admissible claims. Passing code tests or compiling the PDF
does not satisfy the empirical gates.

## Completed

- [x] Dataset-first title, abstract, contribution order, Figure 1, discussion,
  and conclusion.
- [x] Public 522-target-turn release from 321 source conversations with two
  explicit discovery splits, a detailed dataset card, intended-use limits,
  explicit source-license filtering, and a reference next-response protocol.
- [x] Hosted combined parquet verified byte-for-byte: 522 rows, no LMSYS text,
  no target integrity errors, SHA-256
  `4aa19e12933e16ef3119be00d3031fc16a8fd22db56a18d84c6b53ae51c8cab3`.
- [x] Exact release-integrity verification against authoritative pipeline rows:
  433 unique rows, no conversation or target mismatches, canonical SHA-256
  `dd34ec93a5fae80e4d9b82017940342206e7cd849dc03cbb9e74e822fadc94dc`.
- [x] Claim-to-artifact audit, deterministic analysis scripts, dependency lock,
  passing tests, clean LaTeX build, and visual PDF inspection.
- [x] Retrieval enrichment, contextual exclusion, source imbalance, privacy,
  and non-prevalence limitations stated explicitly.
- [x] Full 433-target, eight-frame behavioral experiment completed; uncertainty
  clusters all 232 source conversations and every paired contrast is saved.
- [x] Exact SPIRALS endpoint run on all 3,464 responses; it preserves the
  direct-assertion ordering while producing a materially lower absolute rate.
- [x] Weak J-space results removed from the manuscript and anonymous supplement;
  no interpretability claim remains in the submission.
- [x] Source-conversation duplication audited; benchmark uncertainty changed to
  source-conversation cluster bootstrap.
- [x] Locked blinded browser interfaces prepared for two independent raters on
  both remaining human audits; each uses a distinct local-storage key and
  exports only analyzer-compatible complete CSVs. The response audit uses the
  103 automatic-positive census plus 50 controls in each primary arm and 12 in
  each secondary arm (275 rows total),
  and a tested third-review adjudication path produces one primary endpoint.

## Gate A: final-release human precision

**Required for a dataset-centered submission.**

Complete two independent reviews of the locked 100-row sample in
`results/wilddelusion_combined_human_audit/`, containing 62 primary-route and 38
legacy-route rows. Report strict precision with 95% Wilson intervals by route,
post-stratified combined precision with a stratified bootstrap interval, and
inter-rater agreement.

Decision rule:

- Strict precision at least 0.85 with no major systematic exclusion category:
  retain "precision-oriented benchmark" and replace the non-random transfer
  audit in the abstract and Figure 1 with the final-release estimate.
- Strict precision 0.70--0.85: describe the artifact as an LLM-enriched candidate
  benchmark, publish human labels, and make adjudication part of evaluation.
- Strict precision below 0.70: do not headline the 522 rows as verified cases;
  filter or relabel the release before submission.

These thresholds are manuscript decision rules, not hypothesis tests.

## Gate B: human assistant-response endpoint

**Required to treat the framing-aware behavioral rate as validated rather than
judge-defined.**

Independently review the locked 275-response sample spanning all automatic
positives, 50 controls in each primary arm, and 12 controls in each secondary
arm. Report post-stratified frame rates, the
direct-versus-reported-belief contrast with finite-population posterior-
predictive intervals, inter-rater agreement, adjudicated results, and
category-level disagreement. Reviewers must be blind to model-judge scores and
unrelated exploratory analyses.

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
single-model. A future model-family replication must freeze its endpoint before
inspecting outcomes and report null or reversed results.

## Final preflight

- [ ] Replace pending audit language using locked human-analysis outputs.
- [x] Re-run the executable claim audit against committed artifacts and current
  manuscript strings; nine quantitative claim groups pass.
- [x] Confirm the anonymous repository/release policy for the target venue and
  remove the identifying dataset URL from the manuscript.
- [x] Check the ICLR 2026 submission-page, ethics, and LLM-disclosure policies;
  main text ends on page 7 and the appendix now includes an LLM-use disclosure.
- [x] Extract the anonymous ZIP into a fresh directory, install from the locked
  dependency file, run all tests and the claim verifier, and rebuild the PDF;
  the clean-room PDF is byte-identical to the development build.
- [x] Visually inspect every page of the rebuilt PDF; no clipping, overlap,
  link-border artifacts, or malformed figures remain.

Current status: technically reproducible with a complete full-benchmark
behavioral analysis, but not empirically submission-ready until Gate A and Gate
B are complete.
