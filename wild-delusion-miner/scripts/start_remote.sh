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
  --exclude 'artifacts/' \
  --exclude 'logs/' \
  -e "ssh -i ${RUNPOD_SSH_KEY}" \
  ./ "${REMOTE}:${RUNPOD_WORKDIR}/"

ssh -i "${RUNPOD_SSH_KEY}" "${REMOTE}" bash -lc "'
set -euo pipefail
cd ${RUNPOD_WORKDIR}
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
mkdir -p logs artifacts
chmod +x scripts/pipeline_remote.sh
export OPENAI_API_KEY=${OPENAI_API_KEY@Q}
export HF_TOKEN=${HF_TOKEN@Q}
nohup scripts/pipeline_remote.sh > logs/pipeline.log 2>&1 < /dev/null &
echo \$! > logs/pipeline.pid
echo started \$(cat logs/pipeline.pid)
'"
