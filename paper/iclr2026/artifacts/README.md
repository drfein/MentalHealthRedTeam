# Paper aggregate artifacts

This directory is the public, text-free evidence bundle for the manuscript. It
contains aggregate counts, rates, intervals, fitted-model summaries, frozen
hyperparameters, and figure inputs. It contains no conversation text, model
responses, judge rationales, user identifiers, or API credentials.

Top-level JSON and CSV files characterize dataset construction, verification,
release integrity, and the earlier human transfer audit. Subdirectories contain
the ten-model taxonomy/LDA summaries, observed-reply trajectories, judge-context
sensitivity, user/assistant trajectories, matched context ablations, truncation
sweeps, assistant-history substitution, discovery-route checks, descriptive
mini-model comparisons, and the exploratory J-space controls reported in the
paper. The J-space cross-model directory contains only aggregate performance,
frozen settings, and text-free indicator comparisons; row-level messages,
generations, and readouts are intentionally excluded.

`../CLAIM_AUDIT.md` maps every manuscript result to these files.
`../../../configs/paper_experiments.json` maps each result family to its source
code. `../../../scripts/verify_paper_claims.py` checks exact values and
cross-artifact invariants during every paper rebuild.

The full analysis scripts operate on ignored archival inputs under `data/` and
`results/`, which can include sensitive or license-restricted text.
`scripts/reproduce_paper_analyses.sh` refreshes this bundle from those raw
outputs. `scripts/rebuild_paper.sh` is the reviewer-facing zero-network path and
uses only this bundle plus committed figures.
