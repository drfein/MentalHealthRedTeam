#!/bin/bash

# Pull code and configs from the remote machine without copying models or data.
set -euo pipefail

rsync -avzu \
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
  drfein@scdt.stanford.edu:~/hr/ ./

echo "Code-only pull complete. No model/data artifacts were transferred."
