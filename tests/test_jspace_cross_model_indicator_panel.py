from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "jspace_cross_model_indicator_panel",
    SCRIPTS / "analyze_jspace_cross_model_indicator_panel.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_analyze_uses_frozen_orientation_and_excludes_source_from_macro() -> None:
    def frame(name: str, role: str, reverse: bool = False) -> pd.DataFrame:
        signal = np.array([0.0, 0.1, 0.8, 1.0])
        if reverse:
            signal = signal[::-1]
        return pd.DataFrame(
            {
                "model": name,
                "role": role,
                "framing_score": [0, 1, 4, 5],
                "package_score": [0, 1, 7, 8],
                "conspiracy__mean_logit": signal,
                "panel:semantic": signal,
                "panel:placebo": [0.2, 0.8, 0.1, 0.7],
            }
        )

    indicators = {
        "semantic": [
            {"token": "conspiracy", "metric": "mean_logit", "direction": -1}
        ],
        "placebo": [
            {"token": "weather", "metric": "mean_logit", "direction": -1}
        ],
    }
    source = frame("source", "indicator_source_model")
    replicas = [
        frame("replica-a", "replication_model", reverse=True),
        frame("replica-b", "replication_model", reverse=True),
    ]

    performance, macro, _ = MODULE.analyze(
        [source, *replicas], indicators, draws=200, seed=5
    )

    source_auc = performance.query(
        "model == 'source' and endpoint == 'framing_score_gte_4' "
        "and indicator == 'conspiracy__mean_logit'"
    )["auc"].iat[0]
    replica_macro = macro.query(
        "endpoint == 'framing_score_gte_4' "
        "and indicator == 'conspiracy__mean_logit'"
    )
    assert source_auc == 1.0
    assert replica_macro["n_models"].iat[0] == 2
    assert replica_macro["macro_auc"].iat[0] == 0.0
