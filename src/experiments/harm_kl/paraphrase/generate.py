"""Generate alternate versions of X for the paraphrase-validation flow.

This file reads each saved preference record, takes its original user message X,
fills the prompt template in configs/harm_kl/paraphrase/prompt_template.txt, and
asks the model to return paraphrased versions of X as JSON. This prompt is only
for creating paraphrases. It does not replace the main NEUTRAL_SYSTEM_PROMPT in
generate_preferences.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.reader import load_yaml

# System instruction for paraphrase generation only.
DEFAULT_SYSTEM_PROMPT = (
    "You rewrite user prompts exactly as instructed and return valid JSON only. "
    "Do not add commentary before or after the JSON."
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


def load_prompt_template(path: str | Path) -> str:
    # This is the main editable prompt file for changing how new X variants are written.
    with open(Path(path), "r", encoding="utf-8") as f:
        return f.read().strip()


def extract_json_payload(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    starts = [idx for idx in (stripped.find("["), stripped.find("{")) if idx != -1]
    if not starts:
        raise ValueError("Model output did not contain JSON.")
    start = min(starts)

    bracket_pairs = {"[": "]", "{": "}"}
    opener = stripped[start]
    closer = bracket_pairs[opener]
    depth = 0
    for idx in range(start, len(stripped)):
        char = stripped[idx]
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return stripped[start : idx + 1]
    raise ValueError("Model output contained an unterminated JSON payload.")


def normalize_generated_variants(case_id: str, raw_payload: str, include_original: bool, original_text: str) -> list[dict]:
    payload = json.loads(extract_json_payload(raw_payload))
    if isinstance(payload, dict):
        variants = payload.get("variants")
        if variants is None:
            raise ValueError("JSON object output must include a 'variants' key.")
    elif isinstance(payload, list):
        variants = payload
    else:
        raise ValueError("Paraphrase output must be a JSON list or object.")

    normalized: list[dict[str, str]] = []
    if include_original:
        normalized.append(
            {
                "variant_id": "original",
                "label": "original",
                "text": original_text,
            }
        )

    for idx, item in enumerate(variants):
        if isinstance(item, str):
            text = item.strip()
            label = f"variant_{idx + 1}"
            variant_id = f"{case_id}_v{idx + 1}"
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            label = str(item.get("label") or f"variant_{idx + 1}")
            variant_id = str(item.get("variant_id") or item.get("id") or f"{case_id}_v{idx + 1}")
        else:
            raise ValueError(f"Unsupported paraphrase item type: {type(item).__name__}")

        if not text:
            raise ValueError(f"Generated paraphrase {idx + 1} for {case_id} is missing text.")
        normalized.append(
            {
                "variant_id": variant_id,
                "label": label,
                "text": text,
            }
        )

    return normalized


def build_prompt(template: str, record: dict, variant_count: int) -> str:
    return template.format(
        case_id=record["case_id"],
        name=record.get("name", ""),
        theme=record.get("theme", ""),
        harm_type=record.get("harm_type", ""),
        user_message=record["user_message"],
        y_pos=record.get("y_pos", ""),
        y_neg=record.get("y_neg", ""),
        y_refusal=record.get("y_refusal", ""),
        variant_count=variant_count,
    )


def run_from_config(config: dict) -> None:
    case_id = config.get("case_id")
    overwrite = bool(config.get("overwrite", False))
    include_original = bool(config.get("include_original", True))
    variant_count = int(config.get("variant_count", 3))

    preferences_dir = Path(config["preferences_dir"])
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_template = load_prompt_template(config["prompt_template_path"])
    system_prompt = config.get("generation_system_prompt", DEFAULT_SYSTEM_PROMPT).strip()

    pref_files = sorted(preferences_dir.glob("*.json"))
    if case_id:
        pref_files = [f for f in pref_files if f.stem == case_id]
    if not pref_files:
        raise FileNotFoundError(f"No preference files found in {preferences_dir}.")

    existing_cases: dict[str, dict] = {}
    if output_path.exists() and not overwrite:
        with open(output_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        for case in existing_data.get("cases", []):
            existing_cases[case["id"]] = case

    print(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    cases = []
    for pref_file in pref_files:
        with open(pref_file, "r", encoding="utf-8") as f:
            record = json.load(f)

        if record["case_id"] in existing_cases:
            print(f"[skip] {record['case_id']} - already exists")
            cases.append(existing_cases[record["case_id"]])
            continue

        prompt = build_prompt(prompt_template, record, variant_count)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        print(f"[generate] {record['name']} ({record['case_id']})")
        raw_output = generate(
            model,
            tokenizer,
            messages,
            max_new_tokens=int(config["max_new_tokens"]),
            temperature=float(config["temperature"]),
        )
        try:
            variants = normalize_generated_variants(
                case_id=record["case_id"],
                raw_payload=raw_output,
                include_original=include_original,
                original_text=record["user_message"],
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to parse paraphrases for {record['case_id']}. Raw output:\n{raw_output}"
            ) from exc

        cases.append(
            {
                "id": record["case_id"],
                "name": record.get("name", ""),
                "variants": variants,
            }
        )

    payload = {"cases": cases}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved paraphrase variants -> {output_path}")


def run_from_config_path(config_path: str) -> None:
    run_from_config(load_yaml(config_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paraphrased variants of X.")
    parser.add_argument(
        "--config",
        default="configs/harm_kl/paraphrase/generate.yaml",
        help="Path to the YAML config for this step.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_from_config_path(args.config)
