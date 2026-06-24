#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

: "${OPENAI_API_KEY:?OPENAI_API_KEY required}"

follow_pid="${1:-}"
if [[ -z "$follow_pid" && -f logs/extract_embed.pid ]]; then
  follow_pid="$(cat logs/extract_embed.pid)"
fi

if [[ -z "$follow_pid" ]]; then
  wild-delusion-miner watch-corpus-embedding-batches --submit
else
  wild-delusion-miner watch-corpus-embedding-batches --follow-pid "$follow_pid" --submit
fi
