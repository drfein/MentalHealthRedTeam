import json

import pandas as pd

from wild_delusion_miner.verify import (
    build_context_window,
    evaluate_review_agreement,
    import_review_conversations,
    judge_input_payload,
)


def test_context_window_includes_opening_preceding_and_target():
    messages = []
    for i in range(16):
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"})

    window = build_context_window(messages, target_message_index=14)

    assert [item.index for item in window] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    assert [item.index for item in window if item.section == "opening"] == [0, 1, 2, 3]
    assert [item.index for item in window if item.section == "target"] == [14]
    assert window[-1].is_target is True


def test_judge_payload_marks_target_message():
    row = {
        "conversation_id": "c1",
        "message_hash": "h1",
        "target_message_index": 2,
        "messages": [
            {"role": "user", "content": "write a story"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "the moon is a projector"},
        ],
    }

    payload = judge_input_payload(row)

    assert payload["target_message_index"] == 2
    assert [message["is_target"] for message in payload["messages"]] == [False, False, True]


def test_evaluate_review_agreement(tmp_path):
    review_path = tmp_path / "review.csv"
    judge_path = tmp_path / "judge.jsonl"
    out_path = tmp_path / "summary.json"

    pd.DataFrame(
        [
            {"conversation_id": "a", "decision": "saved"},
            {"conversation_id": "b", "decision": "rejected"},
            {"conversation_id": "c", "decision": "saved"},
        ]
    ).to_csv(review_path, index=False)
    rows = [
        {"conversation_id": "a", "judge_label": "positive", "judge_exclusion": "none"},
        {"conversation_id": "b", "judge_label": "positive", "judge_exclusion": "none"},
        {"conversation_id": "c", "judge_label": "negative", "judge_exclusion": "roleplay"},
    ]
    judge_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = evaluate_review_agreement(
        review_csv_path=review_path,
        judge_output_path=judge_path,
        out_path=out_path,
    )

    assert summary["matched_rows"] == 3
    assert summary["accuracy"] == 1 / 3
    assert summary["confusion"] == {
        "true_positive": 1,
        "true_negative": 0,
        "false_positive": 1,
        "false_negative": 1,
    }
    assert json.loads(out_path.read_text())["matched_rows"] == 3


def test_import_review_conversations(tmp_path):
    review_path = tmp_path / "review.csv"
    out_path = tmp_path / "conversations.jsonl"
    messages = [
        {"role": "user", "content": "write a story"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "the moon is a projector"},
    ]
    pd.DataFrame(
        [
            {
                "annotation_id": "ann1",
                "conversation_id": "conv1",
                "decision": "rejected",
                "notes": "",
                "flagged_msg_idx": 2,
                "flagged_text": "the moon is a projector",
                "source": "probe",
                "row_index": 1,
                "language": "English",
                "delusion_theme_primary": "bizarre",
                "escalation_classification": "neutral",
                "gpt_score": 8,
                "probe_score": 7,
                "messages_json": json.dumps(messages),
            }
        ]
    ).to_csv(review_path, index=False)

    count = import_review_conversations(review_csv_path=review_path, out_path=out_path)
    row = json.loads(out_path.read_text().strip())

    assert count == 1
    assert row["conversation_id"] == "conv1"
    assert row["target_message_index"] == 2
    assert row["messages"] == messages
