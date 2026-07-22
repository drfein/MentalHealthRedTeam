from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def source_row(index: int, *, score: float = 0.25) -> dict:
    return {
        "source": "wildchat_full",
        "split": "train",
        "row_offset": index,
        "conversation_id": f"conversation-{index}",
        "raw_keys": ["conversation"],
        "stable_key": f"stable-{index}",
        "message_hash": f"hash-{index}",
        "messages": [
            {"role": "user", "content": "Opening"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": f"Target {index}"},
        ],
        "target_message_index": 2,
        "retrieval_score": score,
        "annotation_score": 8,
        "annotation_rationale": "Rationale",
        "judge_model": "judge",
        "judge_prompt": "prompt",
        "judge_label": "positive",
        "judge_potential_user_endorses_delusion": True,
        "judge_exclusion": "none",
        "judge_confidence": 0.9,
        "judge_rationale": "Rationale",
        "judge_supporting_quotes": [],
    }


def with_source(row: dict, source: str) -> dict:
    copied = dict(row)
    copied["source"] = source
    return copied


def release_row(row: dict) -> dict:
    projected = {key: value for key, value in row.items() if key not in {"raw_keys", "stable_key"}}
    projected["target_text"] = projected["messages"][projected["target_message_index"]]["content"]
    return projected


def test_release_verifier_accepts_documented_export_projection(tmp_path: Path) -> None:
    authoritative = tmp_path / "authoritative.jsonl"
    release = tmp_path / "release.jsonl"
    result = tmp_path / "verification.json"
    expected = source_row(1)
    observed = release_row(expected)
    observed["retrieval_score"] += 1e-15
    write_jsonl(authoritative, [expected])
    write_jsonl(release, [observed])

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_wilddelusion_release.py"),
            "--authoritative",
            str(authoritative),
            "--release",
            str(release),
            "--output",
            str(result),
        ],
        check=True,
    )
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["rows"] == 1
    assert payload["message_mismatches"] == 0


def test_release_verifier_rejects_target_text_mismatch(tmp_path: Path) -> None:
    authoritative = tmp_path / "authoritative.jsonl"
    release = tmp_path / "release.jsonl"
    expected = source_row(1)
    observed = release_row(expected)
    observed["target_text"] = "Wrong target"
    write_jsonl(authoritative, [expected])
    write_jsonl(release, [observed])

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_wilddelusion_release.py"),
            "--authoritative",
            str(authoritative),
            "--release",
            str(release),
            "--output",
            str(tmp_path / "verification.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "target text does not match" in completed.stderr


def test_public_release_excludes_nonredistributable_source(tmp_path: Path) -> None:
    authoritative = tmp_path / "authoritative.jsonl"
    output = tmp_path / "release"
    write_jsonl(
        authoritative,
        [source_row(1), with_source(source_row(2), "lmsys_chat_1m")],
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_wilddelusion_public_release.py"),
            "--input",
            str(authoritative),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    release = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert len(release) == 1
    assert release[0]["source"] == "wildchat_full"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["excluded_positive_rows"] == 1

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_wilddelusion_release.py"),
            "--authoritative",
            str(authoritative),
            "--release",
            str(output / "train.jsonl"),
            "--exclude-source",
            "lmsys_chat_1m",
            "--output",
            str(tmp_path / "verification.json"),
        ],
        check=True,
    )


def test_release_audit_sample_and_rater_interfaces_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    write_jsonl(source, [source_row(index) for index in range(6)])
    first = tmp_path / "first"
    second = tmp_path / "second"
    command = [
        sys.executable,
        str(ROOT / "scripts/build_wilddelusion_release_human_audit.py"),
        "--input",
        str(source),
        "--sample-size",
        "4",
        "--seed",
        "17",
    ]
    subprocess.run(command + ["--output-dir", str(first)], check=True)
    subprocess.run(command + ["--output-dir", str(second)], check=True)
    assert (first / "blinded_review.csv").read_bytes() == (
        second / "blinded_review.csv"
    ).read_bytes()
    review = pd.read_csv(first / "blinded_review.csv")
    assert len(review) == 4
    assert review["decision"].isna().all()
    assert "judge_label" not in review.columns

    html_a = tmp_path / "rater_a.html"
    html_b = tmp_path / "rater_b.html"
    html_command = [
        sys.executable,
        str(ROOT / "scripts/make_wilddelusion_release_audit_html.py"),
        "--review-csv",
        str(first / "blinded_review.csv"),
        "--manifest",
        str(first / "manifest.json"),
    ]
    subprocess.run(html_command + ["--rater-id", "rater_a", "--output", str(html_a)], check=True)
    subprocess.run(html_command + ["--rater-id", "rater_b", "--output", str(html_b)], check=True)
    assert "seed-17-rater_a" in html_a.read_text(encoding="utf-8")
    assert "seed-17-rater_b" in html_b.read_text(encoding="utf-8")


def test_release_audit_can_require_discovery_split_coverage(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = []
    for index in range(20):
        item = source_row(index)
        item["discovery_split"] = "primary" if index < 16 else "legacy"
        rows.append(item)
    write_jsonl(source, rows)
    output = tmp_path / "audit"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_wilddelusion_release_human_audit.py"),
            "--input",
            str(source),
            "--output-dir",
            str(output),
            "--sample-size",
            "10",
            "--seed",
            "19",
            "--stratify-field",
            "discovery_split",
            "--min-per-stratum",
            "3",
        ],
        check=True,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sample_stratum_counts"]["primary"] >= 3
    assert manifest["sample_stratum_counts"]["legacy"] >= 3


def test_poststratified_precision_uses_population_weights(tmp_path: Path) -> None:
    key = tmp_path / "audit_key.csv"
    review = tmp_path / "review.csv"
    output = tmp_path / "analysis"
    pd.DataFrame(
        {
            "review_id": ["a", "b", "c", "d"],
            "discovery_split": ["primary", "primary", "legacy", "legacy"],
        }
    ).to_csv(key, index=False)
    pd.DataFrame(
        {
            "review_id": ["a", "b", "c", "d"],
            "decision": ["positive", "positive", "negative", "negative"],
            "exclusion_reason": ["none", "none", "other", "other"],
            "confidence_1_to_5": [5, 5, 5, 5],
        }
    ).to_csv(review, index=False)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_wilddelusion_release_human_audit.py"),
            "--reviews",
            str(review),
            "--rater-names",
            "rater",
            "--audit-key",
            str(key),
            "--output-dir",
            str(output),
            "--stratum-field",
            "discovery_split",
            "--stratum-population",
            "primary=80",
            "--stratum-population",
            "legacy=20",
            "--bootstrap-draws",
            "100",
        ],
        check=True,
    )
    result = pd.read_csv(output / "poststratified_precision.csv")
    strict = result[result["metric"] == "strict_release_precision"].iloc[0]
    assert strict["estimate"] == 0.8
