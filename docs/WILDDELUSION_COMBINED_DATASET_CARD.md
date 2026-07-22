---
license: other
pretty_name: WildDelusionCombined
task_categories:
- text-generation
- text-classification
size_categories:
- n<1K
---

# WildDelusionCombined

WildDelusionCombined is a retrieval-enriched collection of 522 LLM-context-
verified target turns from 321 source conversations that potentially endorse
highly implausible beliefs. It contains only redistributable WildChat and
ShareChat material. No LMSYS-Chat-1M conversation text is included.

Labels describe messages in conversational context. They are not diagnoses of
users, human-adjudicated clinical ground truth, or prevalence estimates.

## Discovery Splits

The release deliberately preserves two retrieval routes:

| Discovery split | Retrieval method | Target turns | Source conversations |
|---|---|---:|---:|
| `openai_embedding_whitened` | Mean-centered, whitened `text-embedding-3-small` query bootstrapped from package-judged positives | 433 | 232 |
| `legacy_probe_gpt52` | Historical linear-probe routes with GPT-5.2 candidate confirmation; the WildChat implementation uses mean-pooled layer-13 activations from `meta-llama/Llama-3.1-8B` | 89 | 89 |
| **Combined** |  | **522** | **321** |

The legacy split was reconstructed from the full conversations in immutable
historical revisions, not from their truncated verifier windows. It was then
deduplicated against the primary split and rerun through the current filters.
It is a retrieval-route replication, not an independent annotation set: both
splits use the same label definition and current judges. The exact historical
model is retained per row where recoverable; ShareChat rows are conservatively
described as the historical probe route because the archived artifact does not
pin its base representation model.

## Current Admission Rule

Every newly admitted legacy row passes:

1. the exact `user-endorses-delusion` prompt from
   [`jlcmoore/llm-delusions-annotations`](https://github.com/jlcmoore/llm-delusions-annotations)
   pinned at commit `3ea8d2117e55099a61feee1c762c6ee0c32a162b`, using GPT-5.5 and the
   package cutoff of 7; and
2. a GPT-5.4-mini contextual verifier with low reasoning effort, which examines
   the first two user and assistant turns plus up to five user and assistant
   turns before the target and excludes role-play, fiction, jokes, text tasks,
   third-party claims, ordinary plausible concerns, and insufficient context.

The primary 433 rows are the previously published strict-verifier-positive
release. For the historical expansion, 200 old verifier positives were audited:
9 LMSYS rows, 8 exact conversation overlaps, 28 additional target-text overlaps,
and 20 within-route duplicate targets were removed. Of 135 reconstructed full-
conversation candidates, 104 passed the current package score cutoff and 89
passed the current contextual verifier.

## Source Composition

| Source | Target turns | Source conversations |
|---|---:|---:|
| ShareChat--ChatGPT | 405 | 216 |
| WildChat | 68 | 68 |
| ShareChat--Grok | 42 | 30 |
| ShareChat--Gemini | 5 | 5 |
| ShareChat--Claude | 2 | 2 |
| **Total** | **522** | **321** |

ShareChat dominates the release. Report source-specific counts and preserve the
`discovery_split` field in analyses.

## Schema

- `source`, `split`, `conversation_id`, and `message_hash`: provenance and
  stable matching fields.
- `messages`: reconstructed conversation.
- `target_message_index` and `target_text`: exact location and text of the
  flagged user turn.
- `discovery_split`, `discovery_model`, and `discovery_score`: retrieval-route
  provenance.
- `annotation_model`, `annotation_score`, and `annotation_rationale`: exact
  package-backed message judgment.
- `judge_*`: contextual-verifier label, confidence, exclusion, rationale, and
  supporting excerpts.
- `legacy_*`: immutable historical revision and retrieval metadata where
  applicable.

The median conversation contains 23 messages (IQR 9--48), and the target occurs
at median zero-based index 13 (IQR 4--27). Sixty-two source conversations
contribute multiple primary-split targets; uncertainty must therefore cluster
by `(source, conversation_id)`.

## Validation Boundary

An earlier non-random 108-case transfer audit of the strict contextual rule
estimated precision at 35/38 = 92.1% (95% Wilson interval [79.2%, 97.3%]). It
predates the final retrieval pools, was not independently double-coded, and is
supporting evidence only. The 522 rows are LLM-filtered candidates, not
exhaustively human-adjudicated positives.

## Intended Use

- next-response and model-behavior audits on natural, context-dependent claims;
- controlled counterfactual studies that preserve claim content;
- evaluation of assistant endorsement and reality-oriented response policies;
- interpretability research on a fixed set of ecologically grounded cases.

Do not use this dataset for clinical diagnosis, person-level inference,
surveillance, moderation, adverse decisions, or prevalence estimation.

## Privacy and Licensing

The rows originate in public conversational datasets and may contain sensitive
or identifying text. Public availability does not eliminate privacy or
contextual-integrity risks. Do not deanonymize, contact, profile, or make
decisions about source users.

The release uses `license: other` because WildChat is distributed under ODC-By
and ShareChat specifies CC BY-NC 4.0. Users must satisfy both sources' terms;
the combined artifact should be treated as non-commercial absent separate
permission. No LMSYS-Chat-1M conversation text is redistributed.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset("danielfein/WildDelusionCombined", split="train")
```

Reproducible mining, legacy reconstruction, verification, and release code is
maintained at
[`drfein/MentalHealthRedTeam`](https://github.com/drfein/MentalHealthRedTeam).
