#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs artifacts

wild-delusion-miner generate-synthetic
wild-delusion-miner extract-users

wild-delusion-miner prepare-embedding-batches
wild-delusion-miner submit-embedding-batches
wild-delusion-miner poll-embedding-batches --wait
wild-delusion-miner download-embedding-outputs
wild-delusion-miner materialize-embeddings

wild-delusion-miner retrieve-initial
wild-delusion-miner make-calibration
wild-delusion-miner annotate-initial-above-threshold

wild-delusion-miner retrieve-from-true-positives
wild-delusion-miner annotate-true-positive-pass

wild-delusion-miner verify-final
wild-delusion-miner inspect-samples --out-path artifacts/inspection.html --per-source 50
