#!/usr/bin/env python3
"""Fail if quantitative manuscript claims drift from committed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=Path("paper/iclr2026"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/iclr2026/artifacts/claim_verification.json"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def close(actual: float | str, expected: float, tolerance: float = 5e-10) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"Expected {expected}, got {actual}")


def require_text(manuscript: str, fragment: str) -> None:
    if fragment not in manuscript:
        raise AssertionError(f"Manuscript is missing audited fragment: {fragment}")


def main() -> None:
    args = parse_args()
    artifacts = args.paper_dir / "artifacts"
    behavior_dir = artifacts / "behavior/full_public"

    mining = read_json(artifacts / "mining_settings.json")
    verification = read_json(artifacts / "verification_summary.json")
    release = read_json(artifacts / "release_characterization.json")
    release_manifest = read_json(artifacts / "release_manifest.json")
    integrity = read_json(artifacts / "release_integrity.json")
    behavior_hparams = read_json(behavior_dir / "hparams.json")
    human = read_csv(artifacts / "human_validation_metrics.csv", "metric")
    retrieval_rows = list(
        csv.DictReader(
            (artifacts / "retrieval_ablation_results.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    behavior = read_csv(behavior_dir / "behavior_by_frame.csv", "arm")
    package_behavior = read_csv(
        behavior_dir / "package_behavior_by_frame.csv", "arm"
    )
    paired = read_csv(behavior_dir / "paired_frame_contrasts.csv", "comparison_arm")
    package_paired = read_csv(
        behavior_dir / "package_paired_frame_contrasts.csv", "comparison_arm"
    )
    overlap = read_csv(behavior_dir / "endpoint_overlap_by_frame.csv", "arm")

    assert mining["materialized_embedding_corpus_rows_approx"] == 6_130_000
    assert mining["bootstrap_labeling"] == {
        "input_rows": 1000,
        "positive_rows": 107,
        "hit_rate": 0.107,
    }
    assert mining["whitening_sample_rows"] == 200_000
    assert mining["whitening_eps"] == 1e-5

    assert verification["input_context_rows"] == 715
    assert verification["distinct_source_conversations"] == 418
    assert verification["label_counts"] == {
        "positive": 436,
        "negative": 271,
        "uncertain": 8,
    }
    assert sum(verification["label_counts"].values()) == 715
    assert verification["positive_source_counts"]["lmsys_chat_1m"] == 3

    assert release["rows"] == release_manifest["rows"] == 433
    assert release["distinct_source_conversations"] == 232
    assert release["multi_target_source_conversations"] == 62
    assert release["target_turns_per_source_conversation"]["max"] == 39
    assert release["source_counts"] == release_manifest["source_target_turn_counts"]
    assert release["source_counts"] == {
        "sharechat_chatgpt": 356,
        "sharechat_grok": 33,
        "wildchat_full": 44,
    }
    assert release["source_conversation_counts"] == {
        "sharechat_chatgpt": 167,
        "sharechat_grok": 21,
        "wildchat_full": 44,
    }
    assert release["annotation_score_counts"] == {
        "7": 87,
        "8": 135,
        "9": 131,
        "10": 80,
    }
    assert release["stored_context_messages"] == {
        "median": 21.0,
        "q1": 9.0,
        "q3": 40.0,
        "min": 1.0,
        "max": 186.0,
    }
    assert release["target_message_index"]["median"] == 13
    assert release["target_message_index"]["q1"] == 4
    assert release["target_message_index"]["q3"] == 26
    assert release["target_words"]["median"] == 103
    assert release["target_words"]["q1"] == 52
    assert release["target_words"]["q3"] == 198
    assert release["verifier_confidence"]["median"] == 0.94
    assert release["verifier_confidence"]["q1"] == 0.89
    assert release["verifier_confidence"]["q3"] == 0.97
    assert release_manifest["excluded_positive_rows"] == 3
    assert integrity["passed"] is True
    assert integrity["canonical_sha256"] == (
        "dd34ec93a5fae80e4d9b82017940342206e7cd849dc03cbb9e74e822fadc94dc"
    )

    precision = human["audited_precision_ppv"]
    assert int(float(precision["numerator"])) == 35
    assert int(float(precision["denominator"])) == 38
    close(precision["estimate"], 35 / 38)
    specificity = human["specificity"]
    sensitivity = human["sensitivity"]
    judge_auc = human["judge_keep_score_auroc"]
    assert (int(float(specificity["numerator"])), int(specificity["denominator"])) == (
        26,
        29,
    )
    assert (int(float(sensitivity["numerator"])), int(sensitivity["denominator"])) == (
        35,
        79,
    )
    close(judge_auc["estimate"], 0.8236577913574858)

    whitened_top = next(
        row
        for row in retrieval_rows
        if row["retrieval_name"] == "positive_bootstrap_v1_positive_whitened"
        and row["bucket"] == "rank_000001_000500"
    )
    assert int(whitened_top["n"]) == 30
    assert int(whitened_top["positives"]) == 12
    expected_retrieval_positives = {
        "hypothetical_bucket_calibration": [4, 4, 1, 0, 0],
        "positive_bootstrap_v1_positive_raw": [10, 3, 2, 1, 1],
        "positive_bootstrap_v1_positive_whitened": [12, 2, 1, 1, 0],
    }
    for retrieval_name, expected in expected_retrieval_positives.items():
        observed = [
            int(row["positives"])
            for row in retrieval_rows
            if row["retrieval_name"] == retrieval_name
        ]
        assert observed == expected

    assert behavior_hparams["n_target_turns"] == 433
    assert behavior_hparams["n_source_conversations"] == 232
    assert behavior_hparams["rows"] == 3_464
    assert behavior_hparams["bootstrap_draws"] == 10_000
    assert behavior_hparams["generation_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert behavior_hparams["decoding"] == "greedy"
    assert behavior_hparams["max_new_tokens"] == 192
    assert behavior_hparams["framing_judge_model"] == "gpt-5.4-mini"
    assert behavior_hparams["framing_judge_reasoning_effort"] == "low"
    expected_framing_counts = {
        "direct_assertion": 75,
        "reported_belief": 13,
        "reconsideration": 4,
        "question": 2,
        "explicit_fiction": 2,
        "quotation_analysis": 1,
        "skeptical_roleplay": 1,
        "translation": 0,
    }
    assert {arm: int(row["positive_n"]) for arm, row in behavior.items()} == (
        expected_framing_counts
    )
    direct = behavior["direct_assertion"]
    assert int(direct["positive_n"]) == 75
    close(direct["positive_rate"], 75 / 433)
    assert int(behavior["reported_belief"]["positive_n"]) == 13
    reported = paired["reported_belief"]
    close(reported["risk_difference"], 62 / 433)
    assert int(reported["direct_positive_comparison_negative"]) == 67
    assert int(reported["direct_negative_comparison_positive"]) == 5
    close(reported["ci_low"], 0.1049234105708199)
    close(reported["ci_high"], 0.18829600907121305)
    close(reported["cluster_sign_flip_holm_p"], 5.523359547510154e-13)
    assert all(float(row["ci_low"]) > 0 for row in paired.values())

    package_direct = package_behavior["direct_assertion"]
    expected_package_counts = {
        "direct_assertion": 20,
        "reported_belief": 1,
        "reconsideration": 1,
        "question": 1,
        "explicit_fiction": 0,
        "quotation_analysis": 0,
        "skeptical_roleplay": 0,
        "translation": 4,
    }
    assert {
        arm: int(row["positive_n"]) for arm, row in package_behavior.items()
    } == expected_package_counts
    assert int(package_direct["positive_n"]) == 20
    assert int(package_behavior["translation"]["positive_n"]) == 4
    package_translation = package_paired["translation"]
    close(package_translation["risk_difference"], 16 / 433)
    close(package_translation["ci_low"], 0.014896426793517341)
    close(package_translation["ci_high"], 0.062344289157202915)
    close(package_translation["cluster_sign_flip_holm_p"], 0.0022144317626953125)
    assert all(float(row["ci_low"]) > 0 for row in package_paired.values())
    assert overlap["direct_assertion"]["both_positive"] == "19"
    assert overlap["direct_assertion"]["framing_only_positive"] == "56"
    assert overlap["direct_assertion"]["package_only_positive"] == "1"

    manuscript = (args.paper_dir / "main.tex").read_text(encoding="utf-8")
    for fragment in (
        "433 LLM-context-verified target turns from 232 source conversations",
        "6.13M embedded user messages",
        "reconstructs context for 715 candidate turns, verifies 436",
        "35 of 38 strict verifier positives",
        "75/433 direct-assertion responses",
        "14.3 percentage points [10.5, 18.8]",
        "20/433 direct responses, 4.6\\%",
        "3.7 points [1.5, 6.2]",
        "62 conversations contribute multiple retained targets",
        "Of 715 reconstructed candidate target contexts from 418 source conversations",
        "436 target turns are verified, 271 are rejected, and 8 are uncertain",
        "ShareChat-ChatGPT (356), WildChat (44), and ShareChat-Grok (33)",
        "21 [9, 40]",
        "13 [4, 26]",
        "103 [52, 198]",
        "87 / 135 / 131 / 80",
        "capped at 192 tokens",
        "67 direct-only and 5 reported-belief-only positives",
        "0.824 [0.723, 0.906]",
    ):
        require_text(manuscript, fragment)

    result = {
        "passed": True,
        "check_groups": [
            "mining",
            "conversation verification",
            "release characterization and integrity",
            "transfer audit",
            "retrieval ablation",
            "generation and judge settings",
            "full framing-aware behavior",
            "exact-package behavior",
            "manuscript rendering strings",
        ],
        "artifact_root": str(artifacts),
        "manuscript": str(args.paper_dir / "main.tex"),
        "human_audit_scope": "non-random 108-case transfer audit only",
        "pending_human_gates": [
            "final-release precision audit",
            "assistant-response endpoint audit",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
