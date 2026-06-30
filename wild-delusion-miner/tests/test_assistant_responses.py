import json
from types import SimpleNamespace

from wild_delusion_miner import assistant_responses as assistant_responses_module
from wild_delusion_miner.assistant_responses import (
    GenerationParams,
    ResponseCallResult,
    build_post_delusion_input,
    generation_id,
    is_probable_generation_model,
    load_done_generation_ids,
    resolve_generation_models,
    target_text,
)


def test_generation_model_filter_includes_text_models_and_excludes_non_text_models():
    assert is_probable_generation_model("gpt-5.4-mini")
    assert is_probable_generation_model("gpt-4.1-2025-04-14")
    assert is_probable_generation_model("o3-mini-2025-01-31")
    assert not is_probable_generation_model("text-embedding-3-small")
    assert not is_probable_generation_model("gpt-image-1")
    assert not is_probable_generation_model("omni-moderation-latest")
    assert not is_probable_generation_model("gpt-4o-realtime-preview")


def test_build_post_delusion_input_stops_at_target_user_message():
    row = {
        "target_message_index": 2,
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "the satellites are reading my aura"},
            {"role": "assistant", "content": "future response should not be included"},
        ],
    }

    messages = build_post_delusion_input(row, system_prompt="system")

    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "the satellites are reading my aura"},
    ]


def test_target_text_and_generation_id_are_stable():
    row = {
        "source": "x",
        "split": "train",
        "row_offset": 4,
        "message_hash": "abc",
        "target_message_index": 0,
        "messages": [{"role": "user", "content": "I can bend time"}],
    }

    assert target_text(row) == "I can bend time"
    assert generation_id(row, "gpt-5.4-mini", "v1") == generation_id(row, "gpt-5.4-mini", "v1")
    assert generation_id(row, "gpt-5.4-mini", "v1") != generation_id(row, "gpt-5.5", "v1")


def test_resolve_models_from_snapshot(tmp_path):
    snapshot_path = tmp_path / "models.json"
    snapshot_path.write_text(
        json.dumps({"models": [{"id": "gpt-a"}, {"id": "gpt-b"}]}),
        encoding="utf-8",
    )

    assert resolve_generation_models(explicit_models=None, model_snapshot_path=snapshot_path) == [
        "gpt-a",
        "gpt-b",
    ]
    assert resolve_generation_models(explicit_models=["gpt-b", "gpt-b"], model_snapshot_path=None) == [
        "gpt-b"
    ]


def test_load_done_generation_ids_can_retry_errors(tmp_path):
    path = tmp_path / "out.jsonl"
    rows = [
        {"generation_id": "success", "error_type": None},
        {"generation_id": "error", "error_type": "BadRequestError"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert load_done_generation_ids(path, include_errors=True) == {"success", "error"}
    assert load_done_generation_ids(path, include_errors=False) == {"success"}


def test_generate_one_marks_incomplete_empty_response_as_error(monkeypatch):
    row = {
        "source": "x",
        "split": "train",
        "row_offset": 4,
        "message_hash": "abc",
        "target_message_index": 0,
        "messages": [{"role": "user", "content": "I can bend time"}],
    }

    def fake_call_response_model(**kwargs):
        return ResponseCallResult(
            response=SimpleNamespace(
                id="response-id",
                status="incomplete",
                output_text="",
                output=[],
                usage={"input_tokens": 10, "output_tokens": 1},
                incomplete_details={"reason": "max_output_tokens"},
            )
        )

    monkeypatch.setattr(
        assistant_responses_module,
        "_call_response_model",
        fake_call_response_model,
    )

    result = assistant_responses_module._generate_one(
        row,
        "gpt-test",
        GenerationParams(max_output_tokens=1),
        timeout_seconds=30,
    )

    assert result["response_id"] == "response-id"
    assert result["error_type"] == "IncompleteResponseError"
    assert json.loads(result["error_message"]) == {"reason": "max_output_tokens"}


def test_generate_one_records_reasoning_fallback(monkeypatch):
    row = {
        "source": "x",
        "split": "train",
        "row_offset": 4,
        "message_hash": "abc",
        "target_message_index": 0,
        "messages": [{"role": "user", "content": "I can bend time"}],
    }

    def fake_call_response_model(**kwargs):
        return ResponseCallResult(
            response=SimpleNamespace(
                id="response-id",
                status="completed",
                output_text="A response.",
                output=[],
                usage={"input_tokens": 10, "output_tokens": 3},
            ),
            reasoning_effort_fallback=True,
        )

    monkeypatch.setattr(
        assistant_responses_module,
        "_call_response_model",
        fake_call_response_model,
    )

    result = assistant_responses_module._generate_one(
        row,
        "gpt-test",
        GenerationParams(reasoning_effort="low"),
        timeout_seconds=30,
    )

    assert result["error_type"] is None
    assert result["reasoning_effort"] == "low"
    assert result["reasoning_effort_fallback"] is True
