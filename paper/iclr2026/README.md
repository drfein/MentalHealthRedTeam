# ICLR paper draft

The anonymous ICLR 2026-format manuscript introduces `WildDelusion`, a
context-verified wild-conversation benchmark, and reports a ten-model response
benchmark, observed production-reply trajectories, behavioral context
ablations, descriptive SPIRALS/LDA analyses, and an exploratory Qwen
Jacobian-lens audit.

## Reproducibility entry points

- `REPRODUCE.md`: three-level verification and reproduction workflow.
- `EXPERIMENTS.md`: exact data, API, GPU, judging, and analysis commands.
- `CLAIM_AUDIT.md`: every numerical claim mapped to a committed aggregate.
- `configs/paper_experiments.json`: machine-readable paper-to-code coverage.
- `scripts/rebuild_paper.sh`: zero-network verification and PDF build.
- `scripts/reproduce_paper_analyses.sh`: deterministic refresh from saved raw
  outputs in the archival workspace.

## Remaining validation

`DATASET_HUMAN_AUDIT_PROTOCOL.md` defines the pending blinded final-release
audit. `HUMAN_AUDIT_PROTOCOL.md` defines the separate assistant-response audit.
The manuscript leaves those cells visibly pending rather than treating
automatic-judge agreement as human ground truth.

## Direct compilation

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Use `scripts/rebuild_paper.sh` from the repository root for the full claim and
code-manifest checks before compilation.

The official ICLR 2026 style and bibliography files are vendored here. Author
names remain anonymous; camera-ready identity and `\iclrfinalcopy` changes must
be made only after review.
