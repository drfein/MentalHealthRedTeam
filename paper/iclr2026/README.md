# ICLR paper draft

The manuscript is an anonymous ICLR 2026-format draft centered on
`WildDelusion`, a precision-oriented wild-conversation benchmark. The controlled
same-model behavioral study is a benchmark application, and the J-space result
is a secondary post hoc conversation-disjoint sensitivity analysis. Exploratory
cross-model and topic analyses remain outside the main claim.

`HUMAN_AUDIT_PROTOCOL.md` defines the locked, blinded validation needed to replace
the current judge-defined primary endpoint with independent human adjudication.
`DATASET_HUMAN_AUDIT_PROTOCOL.md` defines the separate simple-random audit of 100
rows from the final 433-row release. Completing that audit is the highest-value
remaining validation for the dataset-centered submission.
`SUBMISSION_READINESS.md` locks the evidence gates and the conditions under which
weak J-space claims must be demoted or removed.

Compile from this directory:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

For a complete cached-artifact rebuild from the repository root, run
`scripts/rebuild_paper.sh`; see `REPRODUCE.md` for evidence boundaries.

The official, unmodified ICLR 2026 style and bibliography files are vendored in
this directory. `template_reference.tex` is retained only as the official
formatting reference.

Author names remain anonymous. Before a camera-ready release, uncomment
`\iclrfinalcopy`, replace the author block, and re-check the then-current ICLR
format and disclosure requirements.
