"""Generate multi-turn conversations for seeded preference files."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.reader import load_yaml

NEUTRAL_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Respond naturally and conversationally. "
    "Do not add unsolicited warnings or advice."
)


@torch.inference_mode()
def generate(model, tokenizer, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_messages(turns: list[dict], next_user: str) -> list[dict]:
    messages = [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT}]
    for turn in turns:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": next_user})
    return messages


def generate_conversation(prompts: list[str], model, tokenizer, config: dict) -> list[dict]:
    conversation: list[dict] = []
    for prompt in prompts:
        messages = build_messages(conversation, prompt)
        response = generate(
            model,
            tokenizer,
            messages,
            max_new_tokens=config["max_new_tokens_neutral"],
            temperature=config["temperature"],
        )
        conversation.append({"user": prompt, "assistant": response})
    return conversation


def run_from_config(config: dict) -> None:
    case_id = config.get("case_id")
    overwrite = bool(config.get("overwrite", False))

    data_path = Path(config["data_path"])
    output_dir = Path(config["output_dir"])

    with open(data_path, "r", encoding="utf-8") as f:
        raw_prompts = {c["id"]: c["prompts"] for c in json.load(f)["cases"]}

    pref_files = sorted(output_dir.glob("*.json"))
    if case_id:
        pref_files = [f for f in pref_files if f.stem == case_id]

    if not pref_files:
        raise FileNotFoundError(
            f"No preference files found in {output_dir}. Add your preference JSON files there first."
        )

    print(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    for pref_file in pref_files:
        with open(pref_file, "r", encoding="utf-8") as f:
            record = json.load(f)

        if record.get("conversation") and not overwrite:
            print(f"[skip] {record['case_id']} - conversation already exists")
            continue

        prompts = raw_prompts[record["case_id"]]
        print(f"[generate] {record['name']} ({len(prompts)} turns)")
        record["conversation"] = generate_conversation(prompts, model, tokenizer, config)

        with open(pref_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"  -> saved {pref_file}")


def run_from_config_path(config_path: str) -> None:
    run_from_config(load_yaml(config_path))


if __name__ == "__main__":
    run_from_config_path("configs/harm_kl/generate_preferences.yaml")
