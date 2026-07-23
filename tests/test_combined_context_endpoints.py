from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "build_combined_context_endpoint_inputs.py"
)
SPEC = importlib.util.spec_from_file_location("combined_context_endpoints", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_builder_excludes_targets_without_prior_context() -> None:
    rows = []
    for index in (0, 2):
        rows.append(
            {
                "source": "source",
                "conversation_id": f"conversation-{index}",
                "message_hash": f"hash-{index}",
                "discovery_split": "route",
                "target_message_index": index,
                "messages": [
                    {
                        "role": "user" if message_index % 2 == 0 else "assistant",
                        "content": str(message_index),
                    }
                    for message_index in range(index + 1)
                ],
            }
        )

    output = MODULE.build_endpoint_rows(pd.DataFrame(rows))

    assert len(output) == 2
    assert set(output["context_arm"]) == {"prior_0", "prior_all"}
    assert set(output["message_hash"]) == {"hash-2:prior_0", "hash-2:prior_all"}
    assert output["discovery_split"].eq("route").all()
