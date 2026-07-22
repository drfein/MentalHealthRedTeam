from __future__ import annotations

from wild_delusion_miner.context_interventions import (
    build_history_arms,
    build_replacement_input,
    direct_preceding_assistant_index,
)


def example_row() -> dict:
    return {
        "source": "test",
        "conversation_id": "conversation",
        "message_hash": "target-hash",
        "target_message_index": 4,
        "target_text": "You confirmed the signal is meant for me.",
        "messages": [
            {"role": "user", "content": "I think the broadcast contains a signal."},
            {"role": "assistant", "content": "What evidence supports that?"},
            {"role": "user", "content": "It repeated my birthday."},
            {"role": "assistant", "content": "That proves it is a personal signal."},
            {"role": "user", "content": "You confirmed the signal is meant for me."},
        ],
    }


def test_replacement_input_hides_original_assistant_and_later_target() -> None:
    row = example_row()

    replacement = build_replacement_input(row, 12, max_prior_messages=32)

    assert replacement is not None
    assert replacement["original_row_idx"] == 12
    assert replacement["target_text"] == "It repeated my birthday."
    assert replacement["target_message_index"] == 2
    assert len(replacement["messages"]) == 3
    assert "proves" not in str(replacement["messages"])
    assert "confirmed" not in str(replacement["messages"])


def test_history_arms_change_only_the_directly_preceding_assistant() -> None:
    row = example_row()
    arms = {
        item["context_arm"]: item
        for item in build_history_arms(row, 12, "There may be other explanations.", "gen")
    }

    assert set(arms) == {"original", "assistant_removed", "gpt52_substituted"}
    assert arms["original"]["messages"][-2]["content"].startswith("That proves")
    assert len(arms["assistant_removed"]["messages"]) == 4
    assert arms["gpt52_substituted"]["messages"][-2]["content"] == (
        "There may be other explanations."
    )
    assert all(item["messages"][-1]["content"] == row["target_text"] for item in arms.values())


def test_target_without_direct_assistant_is_ineligible() -> None:
    row = example_row()
    row["messages"][3]["role"] = "user"

    assert direct_preceding_assistant_index(row) is None
    assert build_replacement_input(row, 0) is None
