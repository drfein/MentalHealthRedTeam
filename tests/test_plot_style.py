from wild_delusion_miner.plot_style import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER


def test_every_model_has_one_stable_unique_color_and_label() -> None:
    assert set(MODEL_ORDER) == set(MODEL_COLORS) == set(MODEL_LABELS)
    assert len(set(MODEL_COLORS.values())) == len(MODEL_COLORS)
    assert all(color.startswith("#") and len(color) == 7 for color in MODEL_COLORS.values())
