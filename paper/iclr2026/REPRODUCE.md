# Reproducing the manuscript

## Cached-artifact rebuild

From the repository root:

```bash
scripts/rebuild_paper.sh
```

This command uses the committed, text-free aggregate artifacts under
`paper/iclr2026/artifacts/` to regenerate Figure 1, Figure 2, and the complete
PDF. It does not call an API, download source corpora, access private
conversation text, or run a GPU model. It requires the locked Python environment
and a local `pdflatex`/`bibtex` installation.

Expected output:

`output/pdf/wild_delusion_jspace_iclr2026.pdf`

## Evidence boundaries

- `CLAIM_AUDIT.md` maps each manuscript number to its saved authoritative
  artifact.
- `DATASET_HUMAN_AUDIT_PROTOCOL.md` locks the pending simple-random final-release
  audit.
- `HUMAN_AUDIT_PROTOCOL.md` locks the pending 144-response public human endpoint.
- `SUBMISSION_READINESS.md` specifies when unsupported J-space claims must be
  demoted or removed.

The cached rebuild establishes claim-to-aggregate and rendering consistency, not
independent replication of proprietary API calls. Full mining starts from the
commands in the root `README.md`; API model behavior, licensed source snapshots,
and changing upstream corpora remain external reproducibility boundaries.
