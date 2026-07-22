from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_jspace_behavior_link",
    ROOT / "scripts/analyze_jspace_behavior_link.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
normalize_readout_columns = MODULE.normalize_readout_columns


def test_paper_experiment_manifest_paths_exist() -> None:
    manifest_path = ROOT / "configs/paper_experiments.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    experiment_ids = [experiment["id"] for experiment in manifest["experiments"]]
    assert len(experiment_ids) == len(set(experiment_ids))
    assert len(experiment_ids) == 7

    for experiment in manifest["experiments"]:
        assert experiment["paper_sections"]
        assert experiment["code"]
        assert experiment["artifacts"]
        for relative_path in [*experiment["code"], *experiment["artifacts"]]:
            assert (ROOT / relative_path).is_file(), relative_path


def test_public_paper_artifacts_exclude_row_level_text_formats() -> None:
    artifact_root = ROOT / "paper/iclr2026/artifacts"
    forbidden_suffixes = {".jsonl", ".parquet", ".sqlite", ".npy", ".npz"}
    assert not [
        path
        for path in artifact_root.rglob("*")
        if path.is_file() and path.suffix in forbidden_suffixes
    ]

    forbidden_columns = {
        "conversation_id",
        "message_hash",
        "target_text",
        "response",
        "annotation_rationale",
        "annotation_quotes",
    }
    for path in artifact_root.rglob("*.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            columns = next(csv.reader(handle), [])
        sensitive = {
            column
            for column in columns
            for forbidden in forbidden_columns
            if column == forbidden or column.startswith(f"{forbidden}_")
        }
        assert not sensitive, (path, sensitive)


def test_jspace_behavior_link_accepts_explicit_readout_schema() -> None:
    frame = pd.DataFrame(
        {
            "original_row_idx": [0],
            "intervention_arm": ["neutral"],
            "layer": [26],
            "misinformation_max_logit": [0.1],
            "falsity_concern_mean_logit": [0.2],
            "reality_testing_mean_logit": [0.3],
        }
    )

    normalized = normalize_readout_columns(frame)

    assert normalized.loc[0, "arm"] == "neutral"
    assert normalized.loc[0, "misinformation_logit"] == 0.1
    assert normalized.loc[0, "falsity_concern"] == 0.2
    assert normalized.loc[0, "reality_testing"] == 0.3
