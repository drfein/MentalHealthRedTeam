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
wild-delusion-miner verify-final
wild-delusion-miner inspect-samples
```

Artifacts are under `data/`:

- `data/dedupe.sqlite`: de-duplicated user messages plus source references
- `data/synthetic_delusion_messages.jsonl`: one-prompt synthetic delusion messages
- `data/embeddings/corpus/`: OpenAI Batch API embedding inputs, outputs, shards, and corpus mean
- `data/retrieval/*/`: query vectors, scores, and top-k retrieval files
- `data/annotations/`: calibration samples and upstream annotation outputs
- `data/verification/`: full-conversation verification outputs

## Notes

- Corpus embedding uses OpenAI `text-embedding-3-small` through the Batch API.
- Retrieval query is `normalize(mean_positive_embedding - mean_corpus_embedding)`.
- Calibration samples 100 messages per score bin and chooses the score bin whose
  annotation hit rate is closest to the configured 10% target.
- Final verification sends the whole recovered conversation, with the target
  message marked, to a separate JSON verifier prompt that rejects role-play and
  explicitly fictional contexts.

## Smoke Test

```bash
wild-delusion-miner run-local-smoke
pytest
```

The smoke command only touches tiny dataset slices and is intended to validate
plumbing, not produce research candidates.
