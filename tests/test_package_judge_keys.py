from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/judge_generated_responses_with_package.py"
)
SPEC = importlib.util.spec_from_file_location("package_judge", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_source_key_is_independent_of_generation_model() -> None:
    prompt = {"original_row_idx": 7, "intervention_arm": "direct_assertion"}
    generation = {
        **prompt,
        "model_id": "openai/gpt-oss-20b",
        "response": "Example",
    }

    assert MODULE.source_key(prompt) == MODULE.source_key(generation)
    assert MODULE.row_key(prompt) != MODULE.row_key(generation)
