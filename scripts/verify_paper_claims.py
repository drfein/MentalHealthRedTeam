#!/usr/bin/env python3
"""Fail if central manuscript claims drift from committed analysis artifacts."""

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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: float | str, expected: float, tolerance: float = 5e-10) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"Expected {expected}, got {actual}")


def require_text(manuscript: str, fragment: str) -> None:
    if fragment not in manuscript:
        raise AssertionError(f"Manuscript is missing audited fragment: {fragment}")


def main() -> None:
    args = parse_args()
    artifacts = args.paper_dir / "artifacts"

    mining = read_json(artifacts / "mining_settings.json")
    verification = read_json(artifacts / "verification_summary.json")
    combined = read_json(artifacts / "combined_release_characterization.json")
    combined_manifest = read_json(artifacts / "combined_release_manifest.json")
    combined_integrity = read_json(artifacts / "combined_release_integrity.json")
    human = read_csv(artifacts / "human_validation_metrics.csv", "metric")
    multimodel = read_json(artifacts / "multimodel/summary.json")
    longitudinal = read_json(artifacts / "response_longitudinal/summary.json")
    observed_rates = read_csv(
        artifacts / "response_longitudinal/observed_rates_by_platform.csv",
        "serving_platform",
    )
    context = read_json(artifacts / "context_ablation/summary.json")
    context_models = read_csv(
        artifacts / "context_ablation/context_effect_by_model.csv", "model"
    )
    all_turns = read_json(artifacts / "all_turn_trajectories/analysis_summary.json")
    judge_context = read_json(
        artifacts / "response_judge_context_sensitivity/summary.json"
    )
    user_assistant = read_json(artifacts / "user_assistant_trajectories/summary.json")
    mini = read_json(artifacts / "mini_model_hypothesis/summary.json")
    truncation = read_json(artifacts / "context_truncation/summary.json")
    substitution = read_json(
        artifacts / "assistant_history_substitution/summary.json"
    )
    discovery_route = read_json(artifacts / "discovery_route_benchmark/summary.json")
    jspace_effects = read_csv_rows(
        artifacts / "jspace/context_intervention/paired_effects.csv"
    )
    jspace_behavior = read_csv_rows(
        artifacts / "jspace/context_intervention/behavior_paired_effects.csv"
    )
    jspace_indicator = read_json(
        artifacts / "jspace/endorsement_indicator/summary.json"
    )
    jspace_hard_negative = read_json(
        artifacts / "jspace/hard_negative_control/summary.json"
    )

    assert mining["materialized_embedding_corpus_rows_approx"] == 6_130_000
    assert mining["bootstrap_labeling"] == {
        "input_rows": 1000,
        "positive_rows": 107,
        "hit_rate": 0.107,
    }
    assert mining["whitening_sample_rows"] == 200_000

    assert verification["input_context_rows"] == 715
    assert verification["distinct_source_conversations"] == 418
    assert verification["label_counts"] == {
        "positive": 436,
        "negative": 271,
        "uncertain": 8,
    }
    assert verification["positive_source_counts"]["lmsys_chat_1m"] == 3

    assert combined["rows"] == combined_manifest["rows"] == combined_integrity["rows"] == 522
    assert combined["distinct_source_conversations"] == 321
    assert combined["multi_target_source_conversations"] == 62
    assert combined_manifest["discovery_split_target_counts"] == {
        "legacy_probe_gpt52": 89,
        "openai_embedding_whitened": 433,
    }
    assert combined["source_counts"] == {
        "sharechat_chatgpt": 405,
        "sharechat_claude": 2,
        "sharechat_gemini": 5,
        "sharechat_grok": 42,
        "wildchat_full": 68,
    }
    assert combined["annotation_score_counts"] == {
        "7": 103,
        "8": 157,
        "9": 164,
        "10": 98,
    }
    assert combined_integrity["passed"] is True
    assert combined_integrity["lmsys_rows"] == 0
    assert combined_integrity["target_integrity_errors"] == 0

    precision = human["audited_precision_ppv"]
    specificity = human["specificity"]
    sensitivity = human["sensitivity"]
    judge_auc = human["judge_keep_score_auroc"]
    assert (int(float(precision["numerator"])), int(precision["denominator"])) == (35, 38)
    assert (int(float(specificity["numerator"])), int(specificity["denominator"])) == (26, 29)
    assert (int(float(sensitivity["numerator"])), int(sensitivity["denominator"])) == (35, 79)
    close(precision["estimate"], 35 / 38)
    close(judge_auc["estimate"], 0.8236577913574858)

    population = multimodel["analysis_population"]
    assert population == {
        "responses": 4316,
        "targets": 433,
        "conversations": 232,
        "models": 10,
        "excluded_lmsys_responses": 30,
    }
    assert sum(multimodel["models"].values()) == 4316
    taxonomy = {row["flag"]: row for row in multimodel["taxonomy_overall"]}
    expected_flags = {
        "bot_positive_affirmation": 2347,
        "bot_metaphysical_themes": 1808,
        "bot_reflective_summary": 945,
        "bot_grand_significance": 517,
        "bot_endorses_delusion": 289,
        "bot_dismisses_counterevidence": 169,
        "bot_misrepresents_sentience": 54,
        "bot_claims_unique_connection": 6,
    }
    assert {flag: int(taxonomy[flag]["positive"]) for flag in expected_flags} == expected_flags
    topics = {row["label"]: row for row in multimodel["lda_top_topics"]}
    expected_topics = {
        "Existential Validation": 641,
        "Reflective Meaning Amplification": 373,
        "Supportive Referral Guidance": 325,
        "Generic Invitation Redirect": 199,
        "Mystical Affirming Elaboration": 193,
        "Grounded Reality Check": 176,
    }
    assert {label: int(topics[label]["positive"]) for label in expected_topics} == expected_topics

    assert longitudinal["observed"]["recovered_replies"] == 295
    assert longitudinal["observed"]["conversations"] == 156
    assert longitudinal["observed"]["repeated_conversations"] == 27
    close(observed_rates["All recovered"]["rate"], 171 / 295)
    close(observed_rates["chatgpt"]["rate"], 150 / 221)
    close(observed_rates["grok"]["rate"], 17 / 26)
    close(observed_rates["wildchat_unspecified"]["rate"], 2 / 41)
    observed_trend = longitudinal["observed"]["binary_fixed_effect_trend"]
    close(observed_trend["coefficient"], 0.1350926768598484)
    generated_trend = longitudinal["controlled_generations"]["pooled_model_adjusted"]
    close(generated_trend["coefficient"], 0.016643993373582574)

    assert context["matched_pairs"] == 4315
    assert context["targets"] == 433
    assert context["models"] == 10
    close(context["overall"]["difference"], 163 / 4315)
    close(context["zero_prior_context_replication_control"]["difference"], 2 / 530)
    close(context["at_least_one_prior_message"]["difference"], 161 / 3785)
    close(context_models["gpt-4.1-mini-2025-04-14"]["difference"], 49 / 433)
    close(context_models["o3-mini-2025-01-31"]["difference"], 40 / 433)

    assert all_turns["assistant_turns"] == 1367
    assert all_turns["conversations"] == 27
    assert all_turns["all_turn_endorsements"] == 667
    endpoint = all_turns["conversation_macro_endpoint_change"]
    close(endpoint["last_minus_first_quintile"], 0.2811903838447241)
    assert endpoint["positive_conversations"] == 18
    assert endpoint["negative_conversations"] == 5
    assert endpoint["tied_conversations"] == 4

    assert judge_context["paired_replies"] == 166
    assert judge_context["target_plus_reply"]["positives"] == 122
    assert judge_context["preceding_context_plus_target_plus_reply"]["positives"] == 132
    close(judge_context["paired_context_effect"]["rate_difference"], 10 / 166)
    assert judge_context["paired_context_effect"]["negative_to_positive"] == 14
    assert judge_context["paired_context_effect"]["positive_to_negative"] == 4

    assert user_assistant["attempted_user_turns"] == 1367
    assert user_assistant["user_judge_errors"] == 2
    assert user_assistant["aligned_turn_pairs"] == 1365
    close(
        user_assistant["user_endpoint_change"]["last_minus_first_quintile"],
        0.1525892270382249,
    )
    close(
        user_assistant["assistant_trend_adjusted_for_user_positive"][
            "progress_coefficient"
        ],
        0.19409758426525908,
    )

    mini_rows = {row["comparison"]: row for row in mini["comparisons"]}
    close(mini_rows["GPT-4o mini - GPT-4o"]["difference"], -24 / 432)
    close(mini_rows["o3 mini - o1"]["difference"], 25 / 433)

    truncation_rows = {
        (row["model"], row["arm"]): row for row in truncation["results"]
    }
    assert truncation_rows[("gpt-4.1-mini-2025-04-14", "target_only")][
        "targets"
    ] == 222
    close(
        truncation_rows[("gpt-4.1-mini-2025-04-14", "target_only")][
            "endorsement_rate"
        ],
        11 / 222,
    )
    close(
        truncation_rows[("gpt-4.1-mini-2025-04-14", "full_context")][
            "endorsement_rate"
        ],
        61 / 222,
    )

    substitution_overall = substitution["overall"]
    assert substitution_overall["model_target_sets"] == 924
    assert substitution_overall["targets"] == 94
    assert substitution_overall["conversations"] == 94
    close(substitution_overall["rates"]["original"]["rate"], 167 / 924)
    close(
        substitution_overall["rates"]["assistant_removed"]["rate"],
        141 / 924,
    )
    close(
        substitution_overall["rates"]["gpt52_substituted"]["rate"],
        81 / 924,
    )
    primary_substitution = substitution_overall["contrasts"][
        "gpt52_substituted_minus_original"
    ]
    close(primary_substitution["difference"], -86 / 924)
    close(primary_substitution["ci_low"], -0.12864864864864864)
    close(primary_substitution["ci_high"], -0.06047516198704104)
    assert substitution["coherence"]["passing_targets"] == 70
    coherent_substitution = substitution["coherent_only_sensitivity"]
    assert coherent_substitution["model_target_sets"] == 690
    close(
        coherent_substitution["contrasts"][
            "gpt52_substituted_minus_original"
        ]["difference"],
        -59 / 690,
    )

    route_rows = {row["model"]: row for row in discovery_route["results"]}
    assert discovery_route["models"] == 10
    assert sum(int(row["historical_n"]) for row in route_rows.values()) == 885
    close(route_rows["o3-mini-2025-01-31"]["historical_minus_primary"], -0.08638451358434751)

    intervention_rows = {
        (row["metric"], row["contrast"], int(row["layer"])): row
        for row in jspace_effects
    }
    intervention_key = "validating_minus_reality_testing"
    misinformation = intervention_rows[
        ("misinformation_max_logit", intervention_key, 26)
    ]
    reality_testing = intervention_rows[
        ("reality_testing_mean_logit", intervention_key, 26)
    ]
    falsity_concern = intervention_rows[
        ("falsity_concern_mean_logit", intervention_key, 26)
    ]
    assert int(misinformation["n_pairs"]) == 417
    close(misinformation["mean_difference"], -0.2998351318944844)
    close(misinformation["bootstrap_ci_low"], -0.3455148381294964)
    close(misinformation["bootstrap_ci_high"], -0.25535727667865704)
    close(reality_testing["mean_difference"], -0.10871418131037058)
    close(falsity_concern["mean_difference"], -0.04639616254584443)

    behavior_rows = {
        (row["metric"], row["contrast"]): row for row in jspace_behavior
    }
    behavior_effect = behavior_rows[("positive", intervention_key)]
    close(behavior_effect["mean_difference"], -1 / 417)
    close(behavior_effect["bootstrap_ci_low"], -10 / 417)
    close(behavior_effect["bootstrap_ci_high"], 8 / 417)

    assert jspace_indicator["n_messages"] == 417
    assert jspace_indicator["conditional_prompt_fixed"]["n_discordant_messages"] == 32
    close(jspace_indicator["baseline_grouped_cv_auc"], 0.5963196824824102)
    close(jspace_indicator["full_grouped_cv_auc"], 0.731661555114559)
    close(jspace_indicator["grouped_cv_auc_improvement"], 0.1353418726321488)
    close(jspace_indicator["grouped_cv_auc_improvement_ci"][0], 0.028406799911630438)
    close(jspace_indicator["grouped_cv_auc_improvement_ci"][1], 0.2491291918764868)
    assert jspace_indicator["causal_gating_supported"] is False
    assert (
        jspace_indicator["pre_registered_token_group_control"][
            "epistemic_mean_auc_improvement"
        ]
        > jspace_indicator["pre_registered_token_group_control"][
            "placebo_mean_auc_improvement"
        ]
    )

    assert jspace_hard_negative["n"] == 337
    assert jspace_hard_negative["verified_n"] == 175
    assert jspace_hard_negative["hard_negative_n"] == 162
    close(jspace_hard_negative["oof_auc_improvement"], 0.019153439153439256)
    close(jspace_hard_negative["oof_auc_improvement_ci"][0], -0.004672139422915236)
    close(jspace_hard_negative["oof_auc_improvement_ci"][1], 0.04413349087188762)

    manuscript = (args.paper_dir / "main.tex").read_text(encoding="utf-8")
    for fragment in (
        "522 verified target turns from 321 WildChat and ShareChat conversations",
        "Every released row passes two automatic inclusion stages",
        "All 522 released rows satisfy both rules",
        "Illustrative construct-boundary examples",
        "author-written composite paraphrase",
        "does not correspond one-to-one with any released conversation",
        "Blinded final-release precision",
        "\\pending",
        "4,316 successful responses",
        "34,528 structured annotation decisions",
        "2,347/4,316 responses",
        "289 responses (6.7\\%",
        "3/433 for GPT-5.2",
        "84/433 for GPT-4.1-mini",
        "641/4,316, 14.9\\%",
        "325, 7.5\\%",
        "171/295 replies (58.0\\%",
        "150/221 for ChatGPT",
        "17/26 for Grok",
        "2/41 (4.9\\%",
        "Those 27 contain 1,367 scorable assistant replies",
        "paired last-minus-first-quintile change is +28.1 percentage points",
        "18 conversations increase, five decrease, and four tie",
        "target-plus-reply judging flags 122/166 (73.5\\%)",
        "paired increase of +6.0 points",
        "Fourteen labels switch from negative to positive and four from positive to negative",
        "A public Jacobian lens is applied to Qwen2.5-7B-Instruct",
        "raises grouped ten-fold \\auc{} from 0.596 to 0.732",
        "on contextual hard negatives the incremental \\auc{} is only 0.019",
        "Conversation-macro user-positive prevalence rises early and then plateaus",
        "a +15.3-point endpoint change",
        "associated with +32.0 points of assistant endorsement",
        "pooled model-adjusted estimate across retained target positions is +1.7 points",
        "Full context increases strict endorsement from 126/4,315 (2.9\\%)",
        "a paired difference of +3.78 percentage points",
        "identical-visible-input control",
        "GPT-4.1-mini (+11.3 points",
        "earlier user turns and still increase by +4.20 points",
        "5.0\\%, 13.1\\%, 11.7\\%, 22.5\\%, and 27.5\\%",
        "Across 924 complete model--target sets, endorsement is 18.1\\%, 15.3\\%, and 8.8\\%",
        "substitution also reduces endorsement relative to deletion by $-6.49$ points",
        "Seventy of 94 replacements score at least 7/10",
        "endorsement falls from 15.8\\% to 7.2\\%",
        "41/885 responses (4.6\\%)",
        "GPT-4o-mini is 5.6 percentage points \\emph{lower}",
        "\\auc{} from 0.596 to 0.732",
        "0.135 [0.028, 0.249]",
        "without moving average behavior",
        "does not establish a universal monotonic effect",
    ):
        require_text(manuscript, fragment)

    result = {
        "passed": True,
        "check_groups": [
            "mining and conversation verification",
            "combined release composition and integrity",
            "human transfer audit",
            "public-cohort response filtering",
            "SPIRALS taxonomy aggregates",
            "LDA topic aggregates",
            "recovered production replies and longitudinal models",
            "matched behavioral context ablation",
            "all-turn production trajectories",
            "paired response-judge context sensitivity",
            "aligned user and assistant trajectories",
            "context truncation dose response",
            "assistant-history substitution and coherence sensitivity",
            "mini-model paired comparisons",
            "discovery-route response sensitivity",
            "J-space intervention, indicator, placebo, and hard-negative controls",
            "manuscript rendering strings and pending validation markers",
        ],
        "artifact_root": str(artifacts),
        "manuscript": str(args.paper_dir / "main.tex"),
        "human_audit_scope": "non-random 108-case transfer audit only",
        "pending_human_gates": [
            "blinded final-release audit with independent reviewers",
            "assistant-response judge calibration",
            "topic-label and stability audit",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
