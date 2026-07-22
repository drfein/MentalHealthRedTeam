# Paper aggregate artifacts

These files contain the text-free aggregate values needed to audit manuscript
claims and regenerate both paper figures. They contain no conversation text,
user identifiers, API credentials, or model responses.

`combined_release_*.json` characterizes and verifies the 522-row combined
release while preserving the 433-row whitened-embedding and 89-row legacy-probe
discovery splits. `behavior/full_public/` contains the eight-frame behavioral
analysis on the 433-target primary discovery cohort under both rubric
formulations: frame rates, paired risk differences, source-stratified
descriptive rates, endpoint overlap, and frozen analysis settings. Exploratory
interpretability artifacts are not part of the submission evidence bundle.

The full analysis scripts operate on local source and model-output caches that
cannot all be redistributed. `scripts/rebuild_paper.sh` intentionally consumes
this compact bundle instead, making the public figure and PDF build independent
of those private caches. `CLAIM_AUDIT.md` maps every reported number to a file in
this directory.

The primary release is independently verified by `release_integrity.json`; the
hosted combined parquet is independently verified by
`combined_release_integrity.json` at SHA-256
`4aa19e12933e16ef3119be00d3031fc16a8fd22db56a18d84c6b53ae51c8cab3`.
