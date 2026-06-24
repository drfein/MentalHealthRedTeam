import numpy as np

from wild_delusion_miner.retrieval import _normalize


def test_normalize_returns_unit_vector():
    vector = _normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert np.allclose(np.linalg.norm(vector), 1.0)
