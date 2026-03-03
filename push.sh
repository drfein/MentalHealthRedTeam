#!/bin/bash

# Code-only sync to remote; never transfer models or data.
set -euo pipefail

REMOTE="drfein@scdt.stanford.edu"
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=2"
RSYNC_SSH="ssh ${SSH_OPTS}"

# Fail fast if SSH is down (prevents rsync hanging silently)
${RSYNC_SSH} "${REMOTE}" "echo ssh_ok" >/dev/null

# IMPORTANT:
# - This intentionally excludes all data/model/checkpoint artifacts.
# - It only syncs code/config/log scripts to ~/hr on remote.
rsync -avzu -e "${RSYNC_SSH}" \
  --exclude '.git' \
  --exclude 'node_modules' \
  --exclude '.DS_Store' \
  --exclude 'data/' \
  --exclude '**/checkpoint-*/' \
  --exclude '**/*.safetensors' \
  --exclude '**/*.bin' \
  --exclude '**/*.pt' \
  --exclude '**/*.pth' \
  --exclude '**/*.ckpt' \
  --exclude '**/*.npz' \
  --exclude '**/*.npy' \
  ./ "${REMOTE}:~/hr/"

echo "Code-only sync complete. No model/data artifacts were transferred."
