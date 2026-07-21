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
- [x] J-space semantic placebo and true/neutral framing controls reported
  without semantic or causal overclaiming; weak steering results removed from
  the manuscript.
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

**Required to keep the J-space predictor in the main claims.**

Complete the blinded 144-response public audit in
`results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals/human_audit_direct_assertion/`.
Use score 4--5 as the locked primary endpoint and recompute the unchanged inverse
layer-26 `misinformation` AUROC on the 67-turn conversation-disjoint subset,
with source-conversation cluster intervals. The complete 144-turn result is
secondary.

Decision rule:

- Human AUROC remains directionally above 0.5 and the corrected interval/result
  is reasonably informative: retain the observational J-space association, led
  by the human endpoint; move LLM-rubric estimates to robustness analysis.
- Human labels are too sparse or AUROC is unstable: describe the J-space result
  as exploratory and remove it from the abstract.
- Human AUROC is near or below 0.5: move J-space to the appendix or remove it;
  retain only the behavioral framing benchmark if human labels validate that
  result.

## Gate C: model-family scope

**Resolved conservatively.** The manuscript keeps all interpretability claims
explicitly single-model, removes J-space from the title, and labels the result a
post hoc sensitivity analysis. A future model-family replication must freeze its
endpoint before inspecting outcomes and report null or reversed results.

## Final preflight

- [ ] Replace pending audit language and figures using locked analysis outputs.
- [ ] Re-run `CLAIM_AUDIT.md` against every number in the final manuscript.
- [ ] Confirm the anonymous repository/release policy for the target venue.
- [ ] Re-check the current ICLR page limit, ethics requirements, and disclosure
  policy immediately before submission.
- [ ] Compile from a clean environment and visually inspect every page.

Current status: technically reproducible but not empirically submission-ready
until Gate A is complete. Gate B determines whether J-space belongs in the main
paper; Gate C determines the defensible model-family scope.
