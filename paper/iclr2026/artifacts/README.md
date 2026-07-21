# Paper aggregate artifacts

These files contain the text-free aggregate values needed to audit manuscript
claims and regenerate both paper figures. They contain no conversation text,
user identifiers, API credentials, or model responses.

`behavior/full_public/` contains the primary 433-target, eight-frame behavioral
analysis under both rubric formulations: frame rates, paired risk differences,
endpoint overlap, and frozen analysis settings. Exploratory interpretability
artifacts are not part of the submission evidence bundle.

The full analysis scripts operate on local source and model-output caches that
cannot all be redistributed. `scripts/rebuild_paper.sh` intentionally consumes
this compact bundle instead, making the public figure and PDF build independent
of those private caches. `CLAIM_AUDIT.md` maps every reported number to a file in
this directory.

The public dataset itself is independently verified by `release_integrity.json`.
Its canonical content digest is
`dd34ec93a5fae80e4d9b82017940342206e7cd849dc03cbb9e74e822fadc94dc`.
