# Context dose-response experiment

This experiment measures assistant endorsement after retaining the final
`0, 1, 2, 4, 8, or all` source messages before the same verified target user
turn. User and assistant messages both count. The common cohort contains only
targets with at least eight prior source messages, so every model and arm is
evaluated on the same items.

The generation request has no system message. The judge sees only the target
user message and generated assistant reply, regardless of generation arm.
Fixed settings are recorded in `configs/context_dose_response.json`.

## Build inputs

```bash
uv run python scripts/build_context_dose_response_inputs.py
```

The expected manifest contains 269 targets, 124 source conversations, and
1,614 prompts.

## Generate responses

```bash
uv run wild-delusion-miner generate-post-delusion-responses \
  --input-path data/context_dose_response/inputs.parquet \
  --out-path data/context_dose_response/responses.jsonl \
  --manifest-path data/context_dose_response/responses.manifest.json \
  --no-system-prompt \
  --reasoning-effort low \
  --max-output-tokens 4096 \
  --max-workers 40 \
  --model gpt-3.5-turbo-0125 \
  --model gpt-4-turbo-2024-04-09 \
  --model gpt-4o-2024-05-13 \
  --model gpt-4o-mini-2024-07-18 \
  --model o1-2024-12-17 \
  --model o3-mini-2025-01-31 \
  --model gpt-4.1-mini-2025-04-14 \
  --model gpt-5-mini-2025-08-07 \
  --model gpt-5.2-2025-12-11 \
  --model gpt-5.5-2026-04-23
```

This requests 16,140 generations. Resume incomplete runs with the same command
and add `--retry-errors`.

## Judge responses

```bash
uv run python scripts/prepare_context_dose_response_judgments.py

uv run python scripts/judge_generated_responses_with_package.py \
  --input results/context_dose_response/judge_inputs.jsonl \
  --output results/context_dose_response/package_judgments.jsonl \
  --model gpt-5.4-mini \
  --reasoning-effort low \
  --concurrency 40 \
  --resume \
  --retry-errors
```

Do not pass `--prompts` to the judge. Omitting it is what keeps the outcome
measurement identical across context arms.

## Analyze

```bash
uv run --extra paper python scripts/analyze_context_dose_response.py
```

The analysis keeps only complete six-arm model-target sets and reports
per-model endorsement rates with 95% source-conversation-cluster bootstrap
intervals from 10,000 draws.
