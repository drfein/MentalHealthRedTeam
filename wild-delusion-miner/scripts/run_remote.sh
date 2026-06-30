#!/usr/bin/env bash
set -euo pipefail

: "${RUNPOD_SSH_USER:?set RUNPOD_SSH_USER}"
: "${RUNPOD_SSH_HOST:=ssh.runpod.io}"
: "${RUNPOD_SSH_KEY:=~/.ssh/id_ed25519}"
: "${RUNPOD_WORKDIR:=~/wild-delusion-miner}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
: "${HF_TOKEN:?set HF_TOKEN}"

REMOTE="${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}"

rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'data/' \
  -e "ssh -i ${RUNPOD_SSH_KEY}" \
  ./ "${REMOTE}:${RUNPOD_WORKDIR}/"

ssh -i "${RUNPOD_SSH_KEY}" "${REMOTE}" bash -lc "'
set -euo pipefail
cd ${RUNPOD_WORKDIR}
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
export OPENAI_API_KEY=${OPENAI_API_KEY@Q}
export HF_TOKEN=${HF_TOKEN@Q}
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
wild-delusion-miner inspect-samples
'"
