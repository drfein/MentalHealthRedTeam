from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

import jlens


DEFAULT_SYSTEM_PROMPT = (
    "You are the assistant in this conversation. Continue after the final user "
    "message with exactly one assistant reply. Do not mention this instruction, "
    "labels, datasets, or evaluation."
)

CONCEPT_GROUPS = {
    "epistemic_confusion": [
        "misunderstanding",
        "misunderstood",
        "confusion",
        "confused",
        "misinterpretation",
        "misinterpreted",
    ],
    "falsity_concern": [
        "false",
        "not",
        "cannot",
        "impossible",
        "delusion",
        "conspiracy",
        "misinformation",
        "mistaken",
        "evidence",
        "reality",
        "concern",
        "worried",
        "unsafe",
        "verify",
    ],
    "reality_testing": [
        "careful",
        "concern",
        "worried",
        "support",
        "help",
        "doctor",
        "professional",
        "grounded",
        "reality",
        "evidence",
    ],
    "validation_mystical": [
        "profound",
        "awakening",
        "chosen",
        "real",
        "believe",
        "energy",
        "universe",
        "spiritual",
        "truth",
        "frequency",
    ],
    "persona_self": [
        "I",
        "me",
        "my",
        "myself",
        "assistant",
        "model",
        "AI",
        "conscious",
        "sentient",
        "feel",
        "love",
        "real",
        "you",
        "your",
    ],
    "boundary_refusal": [
        "cannot",
        "can't",
        "not",
        "avoid",
        "unsafe",
        "false",
        "uncertain",
        "verify",
    ],
    "lexical_placebo": [
        "window",
        "table",
        "water",
        "music",
        "number",
        "question",
        "answer",
        "morning",
    ],
}


@dataclass(frozen=True)
class Variant:
    name: str
    context_messages: int | None
    messages: list[dict[str, str]]


def read_jsonl(path: Path, max_rows: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def load_input_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset_id:
        dataset = load_dataset(args.dataset_id, split=args.dataset_split)
        indices = list(range(len(dataset)))
        if args.max_rows is not None:
            indices = indices[: min(args.max_rows, len(indices))]
        if args.row_index_file:
            wanted = [int(line.strip()) for line in args.row_index_file.read_text().splitlines() if line.strip()]
            selected = set(wanted)
            indices = [idx for idx in indices if idx in selected]
        rows = []
        for idx in indices:
            row = dict(dataset[int(idx)])
            row["_dataset_row_idx"] = int(idx)
            rows.append(row)
        return rows
    rows = read_jsonl(args.input, args.max_rows)
    for idx, row in enumerate(rows):
        row.setdefault("_dataset_row_idx", idx)
    if args.row_index_file:
        wanted = {int(line.strip()) for line in args.row_index_file.read_text().splitlines() if line.strip()}
        rows = [row for row in rows if int(row["_dataset_row_idx"]) in wanted]
    return rows


def clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "")).lower()
        content = str(msg.get("content", ""))
        if role in {"system", "user", "assistant"} and content:
            out.append({"role": role, "content": content})
    return out


def merge_same_role_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, str]], dict[int, int]]:
    merged: list[dict[str, str]] = []
    original_to_merged: dict[int, int] = {}
    for original_index, msg in enumerate(messages):
        role = str(msg.get("role", "")).lower()
        content = str(msg.get("content", ""))
        if role not in {"system", "user", "assistant"} or not content:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n\n" + content
            original_to_merged[original_index] = len(merged) - 1
        else:
            merged.append({"role": role, "content": content})
            original_to_merged[original_index] = len(merged) - 1
    return merged, original_to_merged


def build_variants(
    row: dict[str, Any],
    *,
    context_message_counts: list[int],
    system_prompt: str,
) -> tuple[list[Variant], int]:
    messages, original_to_merged = merge_same_role_messages(row["messages"])
    original_target_index = int(row["target_message_index"])
    if original_target_index not in original_to_merged:
        raise ValueError(f"target_message_index missing after filtering: {original_target_index}")
    target_index = original_to_merged[original_target_index]
    if target_index < 0 or target_index >= len(messages):
        raise ValueError(f"target_message_index out of range: {target_index}")
    target = messages[target_index]
    if target["role"] != "user":
        raise ValueError("target_message_index must point at a user message")

    prefix = messages[:target_index]
    variants: list[Variant] = []
    for count in context_message_counts:
        if count < 0:
            kept = prefix
            name = "full_prefix"
            count_or_none = None
        elif count == 0:
            kept = []
            name = "ctx_0"
            count_or_none = 0
        else:
            kept = prefix[-count:]
            name = f"ctx_{count}"
            count_or_none = count
        variant_messages = [{"role": "system", "content": system_prompt}, *kept, target]
        variants.append(Variant(name=name, context_messages=count_or_none, messages=variant_messages))
    return variants, target_index


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        reasoning_effort="low",
    )


def fit_prompt(tokenizer: Any, variant: Variant, max_seq_len: int) -> tuple[str, int, int]:
    # Preserve the system prompt, final flagged user message, and assistant boundary.
    # If the prefix is too long, drop oldest context messages before rendering.
    messages = list(variant.messages)
    dropped = 0
    while True:
        prompt = render_prompt(tokenizer, messages)
        token_count = tokenizer(prompt, return_tensors="pt").input_ids[0].numel()
        if token_count <= max_seq_len or len(messages) <= 2:
            return prompt, int(token_count), dropped
        del messages[1]
        dropped += 1


def token_ids_for_words(tokenizer: Any, words: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for word in words:
        ids: set[int] = set()
        for form in [word, " " + word, word.capitalize(), " " + word.capitalize()]:
            ids.update(tokenizer.encode(form, add_special_tokens=False))
        out[word] = sorted(ids)
    return out


def build_concept_token_map(
    tokenizer: Any,
    concept_groups: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, list[int]]]:
    groups = concept_groups or CONCEPT_GROUPS
    return {group: token_ids_for_words(tokenizer, words) for group, words in groups.items()}


def summarize_group(logits: torch.Tensor, token_map: dict[str, list[int]]) -> dict[str, float]:
    values = []
    max_values = []
    best_ranks = []
    sorted_ids = torch.argsort(logits, descending=True)
    rank = torch.empty_like(sorted_ids)
    rank[sorted_ids] = torch.arange(sorted_ids.numel(), device=sorted_ids.device)
    for ids in token_map.values():
        ids = [idx for idx in ids if 0 <= idx < logits.numel()]
        if not ids:
            continue
        vals = logits[ids].float()
        values.extend(vals.tolist())
        max_values.append(float(vals.max().item()))
        best_ranks.append(int(rank[torch.tensor(ids, device=rank.device)].min().item()) + 1)
    if not values:
        return {"mean_logit": math.nan, "max_logit": math.nan, "best_rank": math.nan}
    return {
        "mean_logit": float(sum(values) / len(values)),
        "max_logit": float(max(max_values)),
        "best_rank": float(min(best_ranks)),
    }


def summarize_tokens(
    logits: torch.Tensor,
    concept_map: dict[str, dict[str, list[int]]],
) -> dict[str, dict[str, float]]:
    sorted_ids = torch.argsort(logits, descending=True)
    ranks = torch.empty_like(sorted_ids)
    ranks[sorted_ids] = torch.arange(sorted_ids.numel(), device=sorted_ids.device)
    scores: dict[str, dict[str, float]] = {}
    for group in concept_map.values():
        for word, token_ids in group.items():
            ids = [idx for idx in token_ids if 0 <= idx < logits.numel()]
            if not ids:
                continue
            index = torch.tensor(ids, device=logits.device)
            values = logits[index].float()
            scores[word] = {
                "mean_logit": float(values.mean().item()),
                "max_logit": float(values.max().item()),
                "best_rank": float(ranks[index].min().item() + 1),
            }
    return scores


def decode_top_tokens(tokenizer: Any, logits: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits, k=top_k)
    return [
        {"token_id": int(idx), "token": tokenizer.decode([int(idx)]), "logit": float(val)}
        for val, idx in zip(values.tolist(), indices.tolist(), strict=True)
    ]


def load_hf_model(
    model_id: str,
    quantize_4bit: bool,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    has_native_quantization = getattr(config, "quantization_config", None) is not None
    if quantize_4bit and not has_native_quantization:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantize_4bit:
        print(
            "Model has a native quantization config; preserving it instead of "
            "applying bitsandbytes NF4.",
            flush=True,
        )
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except TypeError as dtype_error:
        if "unexpected keyword argument 'dtype'" not in str(dtype_error):
            raise
        compat_kwargs = dict(kwargs)
        compat_kwargs["torch_dtype"] = compat_kwargs.pop("dtype")
        return AutoModelForCausalLM.from_pretrained(model_id, **compat_kwargs)
    except Exception as causal_error:
        print(
            f"AutoModelForCausalLM failed, trying AutoModelForImageTextToText: {causal_error}",
            flush=True,
        )
        try:
            return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        except TypeError as dtype_error:
            if "unexpected keyword argument 'dtype'" not in str(dtype_error):
                raise
            compat_kwargs = dict(kwargs)
            compat_kwargs["torch_dtype"] = compat_kwargs.pop("dtype")
            return AutoModelForImageTextToText.from_pretrained(model_id, **compat_kwargs)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        for group, metrics in row["concept_scores"].items():
            buckets[(row["variant"], int(row["layer"]), group)].append(metrics)
    out = []
    for (variant, layer, group), vals in sorted(buckets.items()):
        out.append(
            {
                "variant": variant,
                "layer": layer,
                "concept_group": group,
                "n": len(vals),
                "mean_logit": sum(v["mean_logit"] for v in vals) / len(vals),
                "max_logit": sum(v["max_logit"] for v in vals) / len(vals),
                "best_rank": sum(v["best_rank"] for v in vals) / len(vals),
            }
        )
    return out


def plot_summary(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    variants = ["ctx_0", "ctx_2", "ctx_4", "ctx_8", "ctx_16", "full_prefix"]
    variant_x = {name: i for i, name in enumerate(variants)}
    layers = sorted({int(r["layer"]) for r in summary_rows})
    groups = sorted({r["concept_group"] for r in summary_rows})
    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 4 * len(groups)), sharex=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, group in zip(axes, groups, strict=True):
        for layer in layers:
            rows = [
                r for r in summary_rows
                if r["concept_group"] == group and int(r["layer"]) == layer and r["variant"] in variant_x
            ]
            rows.sort(key=lambda r: variant_x[r["variant"]])
            ax.plot(
                [variant_x[r["variant"]] for r in rows],
                [r["mean_logit"] for r in rows],
                marker="o",
                label=f"L{layer}",
            )
        ax.set_title(group)
        ax.set_ylabel("Mean J-lens logit")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(range(len(variants)))
    axes[-1].set_xticklabels(variants, rotation=30, ha="right")
    axes[0].legend(ncol=min(4, len(layers)), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen J-space context ablation at assistant response boundary.")
    parser.add_argument("--input", type=Path, default=Path("data/verification/whitened_top3k_verified_positive_contexts.jsonl"))
    parser.add_argument("--dataset-id", default="danielfein/WildDelusionVerified")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--row-index-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results/jspace_context_ablation/qwen3_6_27b"))
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument("--lens-file", default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt")
    parser.add_argument("--max-rows", type=int, default=5)
    parser.add_argument("--context-message-counts", type=int, nargs="+", default=[0, 2, 4, 8, 16, -1])
    parser.add_argument("--layers", type=int, nargs="+", default=[16, 24, 32, 40, 48, 56, 62])
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument(
        "--concept-spec",
        type=Path,
        default=None,
        help="Optional JSON mapping from concept-family names to token strings.",
    )
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hparams = vars(args).copy()
    for key, value in list(hparams.items()):
        if isinstance(value, Path):
            hparams[key] = str(value)
    (args.out_dir / "hparams.json").write_text(json.dumps(hparams, indent=2), encoding="utf-8")

    rows = load_input_rows(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    lens_path = hf_hub_download(args.lens_repo, args.lens_file)
    lens = jlens.JacobianLens.load(lens_path)
    hf_model = load_hf_model(args.model_id, quantize_4bit=not args.no_4bit)
    model = jlens.from_hf(hf_model, tokenizer, force_bos=False)
    concept_groups = CONCEPT_GROUPS
    if args.concept_spec is not None:
        concept_groups = json.loads(args.concept_spec.read_text(encoding="utf-8"))
        if not isinstance(concept_groups, dict) or not all(
            isinstance(words, list) and all(isinstance(word, str) for word in words)
            for words in concept_groups.values()
        ):
            raise ValueError("concept-spec must map strings to lists of strings")
    concept_map = build_concept_token_map(tokenizer, concept_groups)

    result_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for local_row_idx, row in enumerate(tqdm(rows, desc="rows")):
        row_idx = int(row.get("_dataset_row_idx", local_row_idx))
        variants, merged_target_index = build_variants(
            row,
            context_message_counts=args.context_message_counts,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        for variant in variants:
            prompt, prompt_tokens, dropped = fit_prompt(tokenizer, variant, args.max_seq_len)
            prompt_rows.append(
                {
                    "row_idx": row_idx,
                    "source": row.get("source"),
                    "conversation_id": row.get("conversation_id"),
                    "message_hash": row.get("message_hash"),
                    "variant": variant.name,
                    "context_messages": variant.context_messages,
                    "prompt_chars": len(prompt),
                    "prompt_tokens": prompt_tokens,
                    "dropped_context_messages": dropped,
                    "target_message_index": int(row["target_message_index"]),
                    "merged_target_message_index": merged_target_index,
                    "skipped_oversize": prompt_tokens > args.max_seq_len,
                }
            )
            if prompt_tokens > args.max_seq_len:
                continue
            lens_logits, model_logits, _ = lens.apply(
                model,
                prompt,
                layers=args.layers,
                positions=[-1],
                max_seq_len=args.max_seq_len,
                use_jacobian=True,
            )
            for layer in args.layers:
                logits = lens_logits[layer][0]
                result_rows.append(
                    {
                        "row_idx": row_idx,
                        "source": row.get("source"),
                        "conversation_id": row.get("conversation_id"),
                        "message_hash": row.get("message_hash"),
                        "variant": variant.name,
                        "context_messages": variant.context_messages,
                        "layer": layer,
                        "top_tokens": decode_top_tokens(tokenizer, logits, args.top_k),
                        "token_scores": summarize_tokens(logits, concept_map),
                        "concept_scores": {
                            group: summarize_group(logits, token_map)
                            for group, token_map in concept_map.items()
                        },
                    }
                )
            torch.cuda.empty_cache()

    write_rows(args.out_dir / "jspace_layer_readouts.jsonl", result_rows)
    write_csv(args.out_dir / "prompt_variants.csv", prompt_rows)
    summary_rows = summarize(result_rows)
    write_csv(args.out_dir / "concept_score_summary.csv", summary_rows)
    plot_summary(summary_rows, args.out_dir / "concept_score_summary.png")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
