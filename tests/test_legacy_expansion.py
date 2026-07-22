from wild_delusion_miner.legacy_expansion import (
    conversation_fingerprint,
    release_source,
    select_legacy_candidates,
)


def row(conversation_id: str, text: str, *, route: str = "probe", label: str = "positive"):
    return {
        "conversation_hash": conversation_id,
        "source": route,
        "messages": [{"role": "user", "content": text}],
        "target_message_index": 0,
        "target_text": text,
        "gpt_score": 8,
        "judge_label": label,
        "judge_model": "gpt-5.4-mini",
        "judge_confidence": 0.9,
    }


def test_release_source_excludes_lmsys_and_maps_sharechat():
    assert release_source("lmsys_probe_gpt52", {"source_corpus": "lmsys"}) is None
    assert release_source(
        "sharechat_probe_gpt52", {"source_corpus": "sharechat", "config": "grok"}
    ) == "sharechat_grok"
    assert release_source("probe", {"source_corpus": "existing_wilddelusion_combined"}) == (
        "wildchat_full"
    )


def test_fingerprint_ignores_whitespace_and_role_case():
    left = [{"role": "USER", "content": "hello   world"}]
    right = [{"role": "user", "content": "hello world"}]
    assert conversation_fingerprint(left) == conversation_fingerprint(right)


def test_selection_excludes_overlap_and_deduplicates_targets():
    current = [
        {
            "messages": [{"role": "user", "content": "already present"}],
            "target_message_index": 0,
            "target_text": "already present",
        }
    ]
    verified = [
        row("overlap", "already present"),
        row("best", "same target"),
        {**row("duplicate", "same  target"), "judge_confidence": 0.8},
        row("lmsys", "excluded", route="lmsys_probe_gpt52"),
        row("negative", "not selected", label="negative"),
    ]
    combined = [
        {
            "conversation_hash": item["conversation_hash"],
            "full_conversation": item["messages"],
            "flagged_msg_idx": item["target_message_index"],
            "flagged_text": item["target_text"],
            "provenance": (
                '{"source_corpus":"lmsys"}'
                if item["source"] == "lmsys_probe_gpt52"
                else '{"source_corpus":"existing_wilddelusion_combined"}'
            ),
        }
        for item in verified
    ]
    selected, manifest = select_legacy_candidates(verified, combined, current)
    assert [item["conversation_id"] for item in selected] == ["best"]
    assert manifest["historical_positive"] == 4
    assert manifest["lmsys_excluded"] == 1
    assert manifest["conversation_overlap_excluded"] == 1
    assert manifest["target_overlap_excluded"] == 0
    assert manifest["within_legacy_target_duplicates"] == 1
