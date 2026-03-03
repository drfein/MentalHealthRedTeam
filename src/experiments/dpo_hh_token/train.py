"""
DPO on Anthropic-HH with a single new token as the only trainable parameter.

The token <hh_reminder> is inserted into the prompt immediately before the model
generates the assistant response.  The HH dataset conversations are parsed into
role/content message lists and formatted with the model's chat template
(tokenizer.apply_chat_template).  <hh_reminder> is appended right after the
assistant generation-prompt header that apply_chat_template emits, so it is the
very first thing the model attends to when producing its response.

Both chosen and rejected completions see the token in their shared prompt; through
DPO its embedding learns to steer generation toward HH-preferred responses.

Trainable parameters: exactly one token embedding row — embed_tokens[new_token_id].
Everything else (all other embedding rows, all transformer layers) is frozen.
A gradient hook zeros out gradient updates for all rows except new_token_id.
"""

import argparse
import os
import re
import yaml
import torch
from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

from src.core.data_paths import load_data_paths, resolve_path_spec


# ---------------------------------------------------------------------------
# Token / model setup
# ---------------------------------------------------------------------------

def add_new_token(model: torch.nn.Module, tokenizer, token_str: str) -> int:
    """Add token_str to tokenizer and resize model embeddings.

    The new token's embedding is randomly initialized (default Gaussian from
    torch's embedding init).  Returns the new token's id.
    """
    tokenizer.add_special_tokens({"additional_special_tokens": [token_str]})
    new_token_id = tokenizer.convert_tokens_to_ids(token_str)
    model.resize_token_embeddings(len(tokenizer))
    return new_token_id


def freeze_all_except_new_token(model: torch.nn.Module, new_token_id: int) -> tuple[int, int]:
    """Freeze all parameters; keep only embed_tokens[new_token_id] trainable.

    A gradient hook zeros out all rows of embed_tokens.weight except new_token_id,
    so the optimizer only ever sees a non-zero gradient for the one new row.

    Returns (trainable_scalar_params, total_scalar_params).
    """
    total = 0
    for param in model.parameters():
        total += param.numel()
        param.requires_grad = False

    embed = model.get_input_embeddings()
    embed.weight.requires_grad = True

    def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(grad)
        mask[new_token_id] = 1.0
        return grad * mask

    embed.weight.register_hook(_mask_grad)

    # Trainable = one embedding row
    trainable = embed.weight.shape[1]  # hidden_size
    return trainable, total


# ---------------------------------------------------------------------------
# Dataset preprocessing
# ---------------------------------------------------------------------------

_HH_TURN_RE = re.compile(r"\n\nHuman: |\n\nAssistant: ")


def _parse_hh_text(text: str) -> list[dict]:
    """Parse an HH-style conversation string into a list of {role, content} dicts.

    HH conversations are formatted as:
        \n\nHuman: <msg>\n\nAssistant: <msg>\n\nHuman: <msg>...
    """
    markers = _HH_TURN_RE.findall(text)
    parts = _HH_TURN_RE.split(text)
    # parts[0] is the empty string before the first marker
    messages = []
    for marker, content in zip(markers, parts[1:]):
        role = "user" if "Human" in marker else "assistant"
        messages.append({"role": role, "content": content.strip()})
    return messages


def preprocess_hh(
    dataset: DatasetDict,
    tokenizer,
    new_token: str,
    split: str = "train",
):
    """Convert Anthropic-HH to DPO format using the model's chat template.

    Each conversation is parsed into role/content messages.  The prompt is
    formatted via tokenizer.apply_chat_template with add_generation_prompt=True,
    which emits the model's assistant-turn header at the end.  <hh_reminder> is
    then appended right after that header, placing it as the very first token the
    model attends to during generation.

    Input rows:  chosen (str), rejected (str)
    Output rows: prompt (str), chosen (str), rejected (str)
    """

    def _process(row):
        chosen_msgs = _parse_hh_text(row["chosen"])
        rejected_msgs = _parse_hh_text(row["rejected"])

        if (
            not chosen_msgs
            or chosen_msgs[-1]["role"] != "assistant"
            or not rejected_msgs
            or rejected_msgs[-1]["role"] != "assistant"
        ):
            return {"prompt": None, "chosen": None, "rejected": None}

        chosen_completion = chosen_msgs[-1]["content"]
        rejected_completion = rejected_msgs[-1]["content"]

        # Prompt = all turns except the final assistant response.
        # Both chosen and rejected share the same conversation history.
        prompt_msgs = chosen_msgs[:-1]

        # apply_chat_template with add_generation_prompt=True appends the
        # assistant-turn header (e.g. <|start_header_id|>assistant<|end_header_id|>\n\n)
        # so the model knows it is the assistant's turn to speak.
        prompt_str = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        # <hh_reminder> goes right after the generation-prompt header.
        # The model will attend to it as the first token of its response context.
        prompt_str = prompt_str + new_token

        return {
            "prompt": prompt_str,
            "chosen": chosen_completion,
            "rejected": rejected_completion,
        }

    ds = dataset[split]
    processed = ds.map(_process, remove_columns=ds.column_names)
    processed = processed.filter(
        lambda x: x["prompt"] is not None and x["chosen"] is not None
    )
    return processed


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train HH DPO token model.")
    parser.add_argument(
        "--config",
        default=os.environ.get("HR_DPO_CONFIG", "configs/dpo_hh_token.yaml"),
    )
    parser.add_argument(
        "--data-paths-config",
        default="configs/data_paths.yaml",
        help="Path to data path registry yaml.",
    )
    args = parser.parse_args(argv)

    config_path = args.config
    cfg = load_config(config_path)
    data_paths = load_data_paths(args.data_paths_config)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    output_cfg = cfg["output"]

    model_id = model_cfg["model_id"]
    new_token = data_cfg["new_token"]
    max_length = model_cfg["max_length"]

    output_dir = resolve_path_spec(output_cfg["base_dir"], data_paths) / output_cfg["run_name"]
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # standard for causal LM DPO

    # ------------------------------------------------------------------
    # Policy model
    # ------------------------------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = {"": local_rank} if local_rank >= 0 else "auto"

    print(f"Loading policy model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    new_token_id = add_new_token(model, tokenizer, new_token)
    print(f"Added token {new_token!r} with id {new_token_id}")

    trainable, total = freeze_all_except_new_token(model, new_token_id)
    print(f"Trainable params: {trainable:,} scalars (1 embedding row) / {total:,} total")

    # ------------------------------------------------------------------
    # Reference model — same base, same new token, fully frozen.
    # Both policy and ref start from identical random init for new_token_id,
    # so the DPO loss starts unbiased w.r.t. the new token.
    # ------------------------------------------------------------------
    print("Loading reference model (frozen)")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    ref_model.config.use_cache = False
    ref_model.resize_token_embeddings(len(tokenizer))

    # Copy the same random init for new_token_id so ref and policy start identical.
    with torch.no_grad():
        ref_model.get_input_embeddings().weight[new_token_id].copy_(
            model.get_input_embeddings().weight[new_token_id]
        )

    for param in ref_model.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset: {data_cfg['dataset_name']}")
    raw = load_dataset(data_cfg["dataset_name"])

    print("Preprocessing train split...")
    train_ds = preprocess_hh(raw, tokenizer, new_token, split="train")
    print(f"  train: {len(train_ds):,} examples")

    eval_ds = None
    if "test" in raw:
        print("Preprocessing test split...")
        eval_ds = preprocess_hh(raw, tokenizer, new_token, split="test")
        print(f"  test: {len(eval_ds):,} examples")

    # ------------------------------------------------------------------
    # DPO training
    # ------------------------------------------------------------------
    dpo_cfg = DPOConfig(
        output_dir=str(output_dir),
        beta=train_cfg["beta"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=train_cfg.get("eval_steps", train_cfg["save_steps"]),
        seed=train_cfg["seed"],
        report_to="none",
        push_to_hub=False,
        gradient_checkpointing=True,
        max_length=max_length,
        max_prompt_length=train_cfg.get("max_prompt_length", max_length // 2),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print(f"Starting DPO training — only {new_token!r} (id={new_token_id}) is trainable...")
    trainer.train()

    # ------------------------------------------------------------------
    # Save: only the new token's embedding row matters.
    # ------------------------------------------------------------------
    rank = int(os.environ.get("LOCAL_RANK", -1))
    if rank <= 0:
        save_path = output_dir / "hh_reminder_embedding.pt"
        token_vec = model.get_input_embeddings().weight[new_token_id].detach().cpu()
        torch.save({"token": new_token, "token_id": new_token_id, "embedding": token_vec}, str(save_path))
        print(f"Saved trained embedding -> {save_path}  (shape {token_vec.shape})")
        tokenizer.save_pretrained(output_dir)
        print(f"Saved tokenizer (with {new_token!r}) -> {output_dir}")


if __name__ == "__main__":
    main()
