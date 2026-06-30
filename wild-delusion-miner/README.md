# Wild Delusion Miner

Clean, resumable pipeline for finding `user-endorses-delusion` candidates in:

- `yuntian-deng/WildChat-4.8M-Full`
- `anoynsharechat/sharechat`
- `lmsys/lmsys-chat-1m`

The label definition, prompt, cutoff, and annotator model path come from
`jlcmoore/llm-delusions-annotations` pinned at commit
`3ea8d2117e55099a61feee1c762c6ee0c32a162b`. The upstream package defines
`user-endorses-delusion` as user messages that explicitly endorse or demonstrate
genuine belief in physically/logically impossible or extremely implausible ideas,
excluding fiction, role-play, hypotheticals, jokes, third-party beliefs, and
common religions/astrology alone. Its cutoff for this label is `7`.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
export OPENAI_API_KEY=...
export HF_TOKEN=...
```

Do not put real keys in git. `.env` is ignored if you prefer to source secrets
locally.

## Pipeline

```bash
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
wild-delusion-miner collect-verification-conversations
wild-delusion-miner verify-conversations
wild-delusion-miner inspect-samples
```

For large runs, start embedding batch preparation while extraction is still
running:

```bash
wild-delusion-miner extract-users
wild-delusion-miner prepare-synthetic-embedding-batches-only
wild-delusion-miner submit-synthetic-embedding-batches-only
wild-delusion-miner watch-corpus-embedding-batches --follow-pid "$(pgrep -f 'wild-delusion-miner extract-users')" --submit
```

The watcher writes full corpus request shards as new de-duplicated rows appear
and submits each completed OpenAI Batch job immediately. The regular
`prepare-embedding-batches` command is resumable and preserves already-written
or already-submitted shards.

Artifacts are under `data/`:

- `data/dedupe.sqlite`: de-duplicated user messages plus source references
- `data/synthetic_delusion_messages.jsonl`: one-prompt synthetic delusion messages
- `data/embeddings/corpus/`: OpenAI Batch API embedding inputs, outputs, shards, and corpus mean
- `data/retrieval/*/`: query vectors, scores, and top-k retrieval files
- `data/annotations/`: calibration samples and upstream annotation outputs
- `data/verification/`: full-conversation verification outputs
- `data/generations/`: model snapshots and generated assistant continuations

## Assistant Response Generation

To generate one assistant continuation immediately after each flagged user
message, first snapshot the currently available OpenAI generation-model
candidates:

```bash
wild-delusion-miner snapshot-openai-response-models \
  --out-path data/generations/openai_response_model_snapshot.json
```

Then run generation from a verified-positive file:

```bash
wild-delusion-miner generate-post-delusion-responses \
  --input-path data/verification/whitened_top3k_verified_positive_contexts.jsonl \
  --model-snapshot-path data/generations/openai_response_model_snapshot.json \
  --out-path data/generations/post_delusion_openai_responses.jsonl \
  --max-workers 8
```

The model snapshot is the reproducibility boundary: it pins the exact model IDs
available at capture time, including dated and size-specific variants. The
generation output is resumable by `(candidate, model, prompt_version)` and
records unsupported-model or API errors as rows instead of silently dropping
them. Pass repeated `--model MODEL_ID` options to restrict a run to an explicit
model subset.

## Notes

- Corpus embedding uses OpenAI `text-embedding-3-small` through the Batch API.
- Retrieval query is `normalize(mean_positive_embedding - mean_corpus_embedding)`.
- Calibration samples 100 messages per score bin and chooses the score bin whose
  annotation hit rate is closest to the configured 10% target.
- Reranking annotations use the pinned `llm-delusions-annotations` package
  prompt/config for `user-endorses-delusion`.
- Final verification sends a compact conversation window to a JSON judge: the
  first 2 user and first 2 assistant messages, plus up to 5 user and 5 assistant
  messages preceding the target message. The judge rejects role-play, fiction,
  translation/text tasks, jokes, third-party quotes, and ordinary plausible
  concerns.
- To compare judge outputs against a review CSV with `conversation_id` and
  `decision` columns:

```bash
wild-delusion-miner evaluate-review-agreement /path/to/review_annotations_rows.csv
```

## Smoke Test

```bash
wild-delusion-miner run-local-smoke
pytest
```

The smoke command only touches tiny dataset slices and is intended to validate
plumbing, not produce research candidates.
