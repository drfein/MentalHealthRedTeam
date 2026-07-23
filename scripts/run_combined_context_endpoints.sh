#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/combined-context-code}"
PYTHON="${PYTHON:-/tmp/combined-context-venv/bin/python}"
CLI="${CLI:-/tmp/combined-context-venv/bin/wild-delusion-miner}"
LOG_DIR="$ROOT/results/combined_context_endpoints/logs"

cd "$ROOT"
mkdir -p "$LOG_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set; refusing to start paid calls." >&2
  exit 2
fi

"$PYTHON" scripts/audit_combined_context_endpoint_run.py \
  | tee "$LOG_DIR/preflight.json"

mapfile -t MODELS < <(
  "$PYTHON" -c \
    'import json; print(*json.load(open("configs/combined_context_endpoints.json"))["generation"]["models"], sep="\n")'
)

for model in "${MODELS[@]}"; do
  echo "Starting $model"
  "$CLI" generate-post-delusion-responses \
    --input-path data/combined_context_endpoints/inputs.parquet \
    --out-path data/combined_context_endpoints/responses.jsonl \
    --model "$model" \
    --max-workers 8 \
    --max-output-tokens 4096 \
    --reasoning-effort low \
    --no-system-prompt \
    --resume \
    2>&1 | tee "$LOG_DIR/generate-${model}.log"
done

"$PYTHON" scripts/prepare_combined_context_endpoint_judgments.py
"$PYTHON" scripts/judge_generated_responses_with_package.py \
  --input results/combined_context_endpoints/judge_inputs.jsonl \
  --output results/combined_context_endpoints/package_judgments.jsonl \
  --model gpt-5.4-mini \
  --reasoning-effort low \
  --concurrency 40 \
  --resume \
  --retry-errors \
  2>&1 | tee "$LOG_DIR/judge.log"

"$PYTHON" scripts/analyze_combined_context_endpoints.py \
  2>&1 | tee "$LOG_DIR/analyze.log"
"$PYTHON" scripts/audit_combined_context_endpoint_run.py \
  | tee "$LOG_DIR/final.json"
