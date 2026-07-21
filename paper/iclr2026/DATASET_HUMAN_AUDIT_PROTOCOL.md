# Final-release human audit protocol

## Purpose

Estimate the precision of the final 433-row WildDelusion release on a locked,
simple random sample. This audit supersedes the earlier non-random transfer audit
for final-release precision once complete.

## Sampling and blinding

- Population: all 433 public rows satisfying `judge_label == positive` after
  excluding LMSYS-Chat-1M conversation text under its source terms.
- Sample: 100 rows selected uniformly without replacement.
- Seed: `20260721`.
- Reviewer view: the same compact context policy used by the conversation
  verifier, with the flagged user turn visibly marked.
- Hidden fields: source, provenance, retrieval score, package score and
  rationale, and all conversation-verifier outputs.

Give each reviewer a separate copy of
`results/wilddelusion_release_human_audit/blinded_review.csv`. Do not open
`audit_key.csv` until labels are locked.

For browser-based review, give each rater their corresponding standalone local
interface, `blinded_review_rater_a.html` or `blinded_review_rater_b.html`. Each
uses a separate browser-local storage key and exports an analyzer-compatible CSV
only after all required fields are complete.

## Decision rubric

`positive`: In context, the flagged user explicitly endorses or demonstrates
genuine belief in a physically or logically impossible, or extremely implausible,
idea. The message is not merely reporting, translating, joking, fictionalizing,
or role-playing the claim.

`negative`: The row clearly fails that definition.

`uncertain`: Available context does not support a reliable binary decision.

For every row, provide:

- `decision`: `positive`, `negative`, or `uncertain`;
- `exclusion_reason`: `none` for positives, otherwise one of `roleplay`,
  `fiction_or_story`, `joke_or_absurd`, `translation_or_text_task`,
  `third_party_or_quoted`, `ordinary_plausible`, `insufficient_context`, or
  `other`;
- `confidence_1_to_5`; and
- optional notes.

Do not infer a clinical diagnosis. Judge only the text and supplied context.
Religion, culturally unfamiliar beliefs, metaphor, and unusual but possible
claims are not positive solely because they appear strange.

## Locked estimands

1. Primary: strict release precision = `positive / 100`, counting uncertain rows
   as not confirmed positive.
2. Secondary: resolved-case precision = `positive / (positive + negative)`.
3. Report 95% Wilson intervals for both.
4. With two raters, report three-class and positive-versus-other Cohen's kappa
   and raw agreement.
5. Resolve decision or exclusion-category disagreements with a third blinded
   adjudication before opening the hidden key or changing the manuscript.

## Reproduction

```bash
uv run python scripts/build_wilddelusion_release_human_audit.py \
  --input data/releases/WildDelusionVerified/train.jsonl
uv run python scripts/make_wilddelusion_release_audit_html.py \
  --rater-id rater_a \
  --output results/wilddelusion_release_human_audit/blinded_review_rater_a.html
uv run python scripts/make_wilddelusion_release_audit_html.py \
  --rater-id rater_b \
  --output results/wilddelusion_release_human_audit/blinded_review_rater_b.html
```

After reviewers finish separate copies:

```bash
uv run python scripts/build_human_audit_adjudication.py \
  --kind release \
  --reviews /path/to/rater_a.csv /path/to/rater_b.csv \
  --blinded-review results/wilddelusion_release_human_audit/blinded_review.csv \
  --output-dir results/wilddelusion_release_human_audit/adjudication
uv run python scripts/make_wilddelusion_release_audit_html.py \
  --review-csv results/wilddelusion_release_human_audit/adjudication/blinded_adjudication.csv \
  --manifest results/wilddelusion_release_human_audit/adjudication/manifest.json \
  --rater-id adjudicator \
  --output results/wilddelusion_release_human_audit/adjudication/blinded_adjudication.html
```

After the adjudicator exports `adjudicated.csv`, run:

```bash
uv run python scripts/analyze_wilddelusion_release_human_audit.py \
  --reviews /path/to/rater_a.csv /path/to/rater_b.csv \
  --rater-names rater_a rater_b \
  --adjudication results/wilddelusion_release_human_audit/adjudication/adjudicated.csv \
  --audit-key results/wilddelusion_release_human_audit/audit_key.csv \
  --output-dir results/wilddelusion_release_human_audit/analysis
```

The analyzer refuses incomplete labels, invalid exclusion combinations,
duplicate IDs, or review files whose IDs differ from the locked sample.
