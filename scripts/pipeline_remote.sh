#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs artifacts

run_stage() {
  echo "[$(date -Is)] START $*"
  "$@"
  echo "[$(date -Is)] DONE  $*"
}

run_stage wild-delusion-miner generate-synthetic
run_stage wild-delusion-miner extract-users

run_stage wild-delusion-miner prepare-embedding-batches
run_stage wild-delusion-miner submit-embedding-batches
run_stage wild-delusion-miner poll-embedding-batches --wait
run_stage wild-delusion-miner download-embedding-outputs
run_stage wild-delusion-miner materialize-embeddings

run_stage wild-delusion-miner retrieve-initial
run_stage wild-delusion-miner make-calibration
run_stage wild-delusion-miner annotate-initial-above-threshold

run_stage wild-delusion-miner retrieve-from-true-positives
run_stage wild-delusion-miner annotate-true-positive-pass

run_stage wild-delusion-miner verify-final
run_stage wild-delusion-miner inspect-samples --out-path artifacts/inspection.html --per-source 50
