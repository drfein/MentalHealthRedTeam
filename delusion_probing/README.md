# Synthetic User Turn Generator

This repo contains a small script for generating synthetic user turns from the taxonomy rows in `annotations.csv`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` file with:

```bash
OPENAI_API_KEY=...
```

## Usage

Generate five examples for the transition-relevant user labels:

```bash
python3 generate_user_turns.py \
  --csv-path /Users/gabrielaaranguiz-dias/Downloads/annotations.csv \
  --output-path synthetic_user_turns.csv
```

Default target IDs:

```text
user-violent-intent
user-suicidal-intent
user-mental-health-diagnosis
user-expresses-isolation
user-endorses-delusion
user-metaphysical-themes
```

Generate only a subset of IDs:

```bash
python3 generate_user_turns.py \
  --csv-path /Users/gabrielaaranguiz-dias/Downloads/annotations.csv \
  --ids user-endorses-delusion user-metaphysical-themes \
  --examples-per-id 5 \
  --output-path subset_user_turns.csv
```

The CLI also accepts the paper-style aliases `user-violent-thoughts` and `user-suicidal-thoughts`, which are mapped to the CSV IDs `user-violent-intent` and `user-suicidal-intent`.

The output CSV includes the taxonomy metadata, the generated user turn, and neutral variation metadata for setting, style, framing, and length. The generator avoids injecting extra mental-health context beyond what the taxonomy ID itself calls for.

Defaults:
- model: `gpt-5.2`
- length: exactly 3 sentences per generated user turn

## Bot Turn Generation

Generate paired bot replies from the synthetic user-turn CSV:

```bash
python3 generate_bot_turns.py \
  --csv-path /Users/gabrielaaranguiz-dias/Desktop/delusion_probing/annotations.csv \
  --user-turns-path /Users/gabrielaaranguiz-dias/Desktop/delusion_probing/synthetic_user_turns_100.csv \
  --output-path synthetic_bot_turns.csv
```

The bot generator expands only the transition set discussed for the Spirals negative cases. It creates one reply that expresses the target bot code and one reply that does not, except for the directly harmful facilitation classes (`bot-facilitates-self-harm` and `bot-facilitates-violence`), where positive examples are marked `manual_required` instead of being synthesized.
