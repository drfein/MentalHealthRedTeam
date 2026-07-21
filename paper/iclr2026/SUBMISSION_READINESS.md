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
- [x] Exact SPIRALS endpoint run on all 3,464 responses; it preserves the
  direct-assertion ordering while producing a materially lower absolute rate.
- [x] Weak J-space results removed from the manuscript and anonymous supplement;
  no interpretability claim remains in the submission.
- [x] Source-conversation duplication audited; benchmark uncertainty changed to
  source-conversation cluster bootstrap.
- [x] Locked blinded browser interfaces prepared for two independent raters on
  both remaining human audits; each uses a distinct local-storage key and
  exports only analyzer-compatible complete CSVs.

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
to model-judge scores and unrelated exploratory analyses.

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
