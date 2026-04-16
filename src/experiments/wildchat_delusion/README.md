# WildChat Delusion Pipeline

Scripts for building the WildDelusion dataset from WildChat-4.8M. Dataset on HuggingFace: `danielfein/WildDelusion`.

## Pipeline Steps

### 1. `annotate_delusion.py` — annotate SAE-activated messages

Annotates an input JSONL of messages with a GPT classifier that scores user-endorses-delusion on a 0–10 scale. Run first on messages flagged by a Sparse Autoencoder feature over WildChat.

```
python annotate_delusion.py --input sae_hits.jsonl --output annotated_sae.jsonl
```

### 2. `train_probe.py` — train Llama layer-13 linear probe

Uses confirmed-positive and negative examples from the WildDelusion HF dataset to train a logistic regression probe on mean-pooled layer-13 residual stream activations. Reports 5-fold CV metrics (balanced accuracy, ROC-AUC, PR-AUC) and saves the probe to JSON.

```
python train_probe.py --out_dir ./output
```

### 3. `scan_corpus.py` — scan all WildChat user messages with the probe

Scores every user message in the WildChat parquet corpus through the layer-13 probe using an early-exit hook for memory efficiency. Writes hits above a threshold score.

```
python scan_corpus.py --corpus_dir /path/to/wildchat --probe output/probe.json --out probe_hits.jsonl
```

### 4. `annotate_delusion.py` (again on probe hits) — verify probe hits

Re-run the same annotation script on the probe hits JSONL. The `--threshold` flag filters to high-scoring hits before calling the API.

```
python annotate_delusion.py --input probe_hits.jsonl --output annotated_probe_hits.jsonl --threshold 2.0
```

### 5. `classify_roleplay.py` — remove roleplay/fiction conversations

Classifies the first user message in each conversation to detect fictional/roleplay framing. Conversations flagged as roleplay should be excluded before generation.

```
python classify_roleplay.py --output roleplay_classifications.jsonl
```

### 6. `generate_responses.py` — generate 64 model responses per conversation

For each remaining conversation, builds a prompt from all turns up to and including the flagged delusional user message and samples 64 independent completions from Llama-3.1-8B-Instruct.

```
python generate_responses.py --out generations.jsonl
```

## Dependencies

```
pip install openai transformers datasets torch scikit-learn pandas tqdm matplotlib
```

All scripts require `OPENAI_API_KEY` set in the environment (steps 1, 4, 5).  
Steps 2, 3, and 6 require a GPU and HuggingFace access to `meta-llama/Llama-3.1-8B`.
