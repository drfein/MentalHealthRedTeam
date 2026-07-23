from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analyze_jspace_full_vocabulary.py"
)
SPEC = importlib.util.spec_from_file_location("full_vocab", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_auc_columns_matches_sklearn_with_ties() -> None:
    scores = np.array(
        [
            [0.0, 3.0],
            [1.0, 2.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ]
    )
    outcome = np.array([0, 0, 1, 1])

    actual = MODULE.auc_columns(scores, outcome)

    expected = np.array(
        [roc_auc_score(outcome, scores[:, index]) for index in range(scores.shape[1])]
    )
    np.testing.assert_allclose(actual, expected)


def test_canonical_word_filters_fragments_and_special_tokens() -> None:
    assert MODULE.canonical_word(" misinformation") == "misinformation"
    assert MODULE.canonical_word("don't") == "don't"
    assert MODULE.canonical_word("##ing") is None
    assert MODULE.canonical_word("<eos>", is_special=True) is None


def test_split_items_is_disjoint_and_complete() -> None:
    item_ids = np.arange(12)
    outcomes = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        ]
    )

    discovery, test = MODULE.split_items(item_ids, outcomes, 0.5, 7)

    assert set(discovery).isdisjoint(test)
    assert set(discovery) | set(test) == set(item_ids)
