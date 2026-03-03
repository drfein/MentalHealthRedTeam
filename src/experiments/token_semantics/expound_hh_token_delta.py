from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.data_paths import load_data_paths, resolve_path_spec


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "token_semantics" / "hh_token_delta.yaml"
SPECIAL_TOKEN = "<hh_reminder>"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_model_and_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model, tok


def generate(model, tokenizer, user_content: str, max_new_tokens: int, do_sample: bool, temperature: float) -> str:
    """Tokenize a user message through the chat template and generate a response.

    <hh_reminder> appearing in user_content is tokenized to its registered id,
    which carries whatever embedding is currently in the embedding matrix.
    """
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        add_special_tokens=False,
    ).to(model.device)
    kwargs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = max(temperature, 1e-6)
    with torch.no_grad():
        out = model.generate(**kwargs)
    new_ids = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe hh token semantics.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument(
        "--data-paths-config",
        default="configs/data_paths.yaml",
        help="Path to data path registry yaml.",
    )
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    data_paths = load_data_paths(args.data_paths_config)
    task = cfg["task"]
    model, tok = load_model_and_tokenizer(cfg["model"]["judge_model_id"])
    if tok.chat_template is None:
        raise ValueError("Tokenizer has no chat template.")

    # Register the special token so it tokenizes to a single id.
    tok.add_special_tokens({"additional_special_tokens": [SPECIAL_TOKEN]})
    token_id = tok.convert_tokens_to_ids(SPECIAL_TOKEN)
    model.resize_token_embeddings(len(tok))

    # Load trained embedding and patch it in.
    token_embedding_path = resolve_path_spec(cfg["paths"]["token_embedding_pt"], data_paths)
    ckpt = torch.load(token_embedding_path, map_location="cpu")
    trained_vec = ckpt["embedding"].detach()
    emb = model.get_input_embeddings()
    with torch.no_grad():
        emb.weight[token_id] = trained_vec.to(dtype=emb.weight.dtype, device=emb.weight.device)

    # Also stash a random-init copy for the comparison baseline.
    random_vec = torch.nn.init.normal_(torch.empty_like(trained_vec))

    token_name = ckpt.get("token", SPECIAL_TOKEN)
    saved_token_id = ckpt.get("token_id", token_id)

    max_new_tokens = int(task.get("max_new_tokens", 800))
    do_sample = bool(task.get("do_sample", False))
    temperature = float(task.get("temperature", 0.0))
    num_samples = int(task.get("delta_num_samples", 3))

    # Substitute the literal special token string into the prompt templates so
    # <hh_reminder> appears as a real token in the user message, carrying
    # whichever embedding is currently active in the matrix.
    compare_prompt = str(task["compare_prompt_template"]).replace("{phrase}", SPECIAL_TOKEN)
    delta_prompt = str(task["delta_prompt_template"]).replace("{phrase}", SPECIAL_TOKEN)

    header = (
        f"Token checkpoint : {token_embedding_path}\n"
        f"Token label      : {token_name}\n"
        f"Token id (ckpt)  : {saved_token_id}\n"
        f"Token id (live)  : {token_id}\n"
        f"Injection mode   : {SPECIAL_TOKEN!r} embedded inline in user message\n"
        f"Compare baseline : random-init embedding at same token id\n"
    )

    # --- Compare: trained vs random embedding, same prompt ---
    if bool(task.get("run_compare", True)):
        # Trained embedding is already active.
        response_trained = generate(model, tok, compare_prompt, max_new_tokens, do_sample, temperature)

        # Swap in random embedding for baseline.
        with torch.no_grad():
            emb.weight[token_id] = random_vec.to(dtype=emb.weight.dtype, device=emb.weight.device)
        response_random = generate(model, tok, compare_prompt, max_new_tokens, do_sample, temperature)

        # Restore trained embedding for subsequent runs.
        with torch.no_grad():
            emb.weight[token_id] = trained_vec.to(dtype=emb.weight.dtype, device=emb.weight.device)

        out_compare = resolve_path_spec(cfg["paths"]["output_compare_txt"], data_paths)
        out_compare.parent.mkdir(parents=True, exist_ok=True)
        out_compare.write_text(
            (
                f"{header}\n"
                f"=== Compare probe ===\n"
                f"Prompt (literal):\n{compare_prompt}\n\n"
                f"--- TRAINED embedding ---\n{response_trained}\n\n"
                f"--- RANDOM-INIT embedding (baseline) ---\n{response_random}\n"
            ),
            encoding="utf-8",
        )
        print(f"Saved compare: {out_compare}")

    # --- Delta: multiple samples with trained embedding ---
    if bool(task.get("run_delta", True)):
        trained_samples = []
        random_samples = []

        for _ in range(max(1, num_samples)):
            trained_samples.append(generate(model, tok, delta_prompt, max_new_tokens, do_sample, temperature))

            with torch.no_grad():
                emb.weight[token_id] = random_vec.to(dtype=emb.weight.dtype, device=emb.weight.device)
            random_samples.append(generate(model, tok, delta_prompt, max_new_tokens, do_sample, temperature))
            with torch.no_grad():
                emb.weight[token_id] = trained_vec.to(dtype=emb.weight.dtype, device=emb.weight.device)

        trained_block = "\n\n".join(
            f"--- Sample {i + 1} (TRAINED) ---\n{txt}" for i, txt in enumerate(trained_samples)
        )
        random_block = "\n\n".join(
            f"--- Sample {i + 1} (RANDOM baseline) ---\n{txt}" for i, txt in enumerate(random_samples)
        )

        out_delta = resolve_path_spec(cfg["paths"]["output_delta_txt"], data_paths)
        out_delta.parent.mkdir(parents=True, exist_ok=True)
        out_delta.write_text(
            (
                f"{header}\n"
                f"=== Delta probe ===\n"
                f"Prompt (literal):\n{delta_prompt}\n\n"
                f"{trained_block}\n\n"
                f"{random_block}\n"
            ),
            encoding="utf-8",
        )
        print(f"Saved delta: {out_delta}")


if __name__ == "__main__":
    main()
