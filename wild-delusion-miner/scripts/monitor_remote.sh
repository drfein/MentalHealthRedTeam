#!/usr/bin/env bash
set -euo pipefail

: "${RUNPOD_SSH_USER:?set RUNPOD_SSH_USER}"
: "${RUNPOD_SSH_HOST:=ssh.runpod.io}"
: "${RUNPOD_SSH_KEY:=~/.ssh/id_ed25519}"
: "${RUNPOD_WORKDIR:=~/wild-delusion-miner}"

REMOTE="${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}"

ssh -i "${RUNPOD_SSH_KEY}" "${REMOTE}" bash -lc "'
set -euo pipefail
cd ${RUNPOD_WORKDIR}
source .venv/bin/activate
wild-delusion-miner status
wild-delusion-miner inspect-samples --out-path artifacts/inspection.html --per-source 20
'"

mkdir -p artifacts/remote
rsync -az \
  -e "ssh -i ${RUNPOD_SSH_KEY}" \
  "${REMOTE}:${RUNPOD_WORKDIR}/artifacts/inspection.html" \
  artifacts/remote/inspection.html
