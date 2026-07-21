from wild_delusion_miner.counterfactual_judge import counterfactual_row_key, parse_judgment


def test_parse_judgment_accepts_valid_json() -> None:
    parsed = parse_judgment(
        'prefix {"reality_endorsement_score": 4, "frame_adherence_score": 2, '
        '"behavior_label": "endorses", "confidence": 0.9} suffix'
    )
    assert parsed["reality_endorsement_score"] == 4


def test_counterfactual_row_key_normalizes_index() -> None:
    assert counterfactual_row_key(
        {"original_row_idx": "12", "intervention_arm": "question"}
    ) == (12, "question")
