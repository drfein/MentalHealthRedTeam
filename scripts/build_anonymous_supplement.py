#!/usr/bin/env python3
"""Build and validate the anonymous paper supplement.

The archive is assembled from an explicit allowlist. It contains no Git history
and filters the non-redistributable LMSYS rows from behavioral artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path


PUBLIC_ROWS = 522
PRIMARY_BEHAVIOR_TARGETS = 433
PUBLIC_BEHAVIOR_ROWS = 3_464
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
RELEASE_JSONL = Path("data/releases/WildDelusionCombined/train.jsonl")
RELEASE_MANIFEST = Path("data/releases/WildDelusionCombined/manifest.json")

ROOT_FILES = ("pyproject.toml", "uv.lock")
COPY_TREES = (
    "src/wild_delusion_miner",
    "paper/iclr2026/artifacts/behavior/full_public",
)
ARTIFACT_FILES = (
    "paper/iclr2026/artifacts/README.md",
    "paper/iclr2026/artifacts/claim_verification.json",
    "paper/iclr2026/artifacts/combined_release_characterization.json",
    "paper/iclr2026/artifacts/combined_release_integrity.json",
    "paper/iclr2026/artifacts/combined_release_manifest.json",
    "paper/iclr2026/artifacts/human_validation_metrics.csv",
    "paper/iclr2026/artifacts/mining_settings.json",
    "paper/iclr2026/artifacts/release_characterization.json",
    "paper/iclr2026/artifacts/release_integrity.json",
    "paper/iclr2026/artifacts/release_manifest.json",
    "paper/iclr2026/artifacts/retrieval_ablation_results.csv",
    "paper/iclr2026/artifacts/verification_summary.json",
)
CONFIG_FILES = (
    "configs/default.yaml",
    "configs/wilddelusion_benchmark_protocol.json",
    "configs/wilddelusion_combined_release_summary.json",
    "configs/wilddelusion_release_summary.json",
)
PAPER_FILES = (
    "paper/iclr2026/CLAIM_AUDIT.md",
    "paper/iclr2026/DATASET_HUMAN_AUDIT_PROTOCOL.md",
    "paper/iclr2026/HUMAN_AUDIT_PROTOCOL.md",
    "paper/iclr2026/HUMAN_AUDIT_REVIEWER_INSTRUCTIONS.md",
    "paper/iclr2026/REPRODUCE.md",
    "paper/iclr2026/iclr2026_conference.bst",
    "paper/iclr2026/iclr2026_conference.sty",
    "paper/iclr2026/main.tex",
    "paper/iclr2026/math_commands.tex",
    "paper/iclr2026/references.bib",
    "paper/iclr2026/figures/full_behavior_main.png",
    "paper/iclr2026/figures/wilddelusion_dataset_overview.png",
    "output/pdf/wild_delusion_iclr2026.pdf",
)
SCRIPT_FILES = (
    "scripts/analyze_conversation_verifier_human_validation.py",
    "scripts/analyze_full_behavior_human_audit.py",
    "scripts/analyze_full_counterfactual_behavior.py",
    "scripts/analyze_wilddelusion_dataset.py",
    "scripts/analyze_wilddelusion_release_human_audit.py",
    "scripts/benchmark_openai_embeddings.py",
    "scripts/bootstrap_embedding_mechanisms.py",
    "scripts/build_counterfactual_judge_audit.py",
    "scripts/build_human_audit_adjudication.py",
    "scripts/build_legacy_expansion_candidates.py",
    "scripts/build_wilddelusion_combined_release.py",
    "scripts/build_wilddelusion_public_release.py",
    "scripts/build_wilddelusion_release_human_audit.py",
    "scripts/filter_positive_annotations.py",
    "scripts/finalize_openai_embeddings_and_retrieve.py",
    "scripts/judge_generated_responses_with_package.py",
    "scripts/judge_semantic_counterfactual_responses.py",
    "scripts/judge_semantic_counterfactual_responses_openai.py",
    "scripts/make_wilddelusion_release_audit_html.py",
    "scripts/make_counterfactual_audit_html.py",
    "scripts/plot_full_counterfactual_behavior.py",
    "scripts/plot_wilddelusion_dataset_overview.py",
    "scripts/populate_verification_contexts.py",
    "scripts/rebuild_paper.sh",
    "scripts/run_hypothetical_bucket_calibration.py",
    "scripts/verify_hf_combined_release.py",
    "scripts/verify_wilddelusion_release.py",
    "scripts/verify_paper_claims.py",
)
TEST_FILES = (
    "tests/test_assistant_responses.py",
    "tests/test_counterfactual_judge.py",
    "tests/test_dataset_adapters.py",
    "tests/test_full_counterfactual_behavior.py",
    "tests/test_human_audit_analysis.py",
    "tests/test_legacy_expansion.py",
    "tests/test_release_audit_tools.py",
    "tests/test_response_audit_html.py",
    "tests/test_retrieval.py",
    "tests/test_verify.py",
)
BEHAVIOR_ROOT = Path(
    "results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals"
)
PROMPTS_SOURCE = Path("data/jspace/semantic_counterfactuals.jsonl")
BEHAVIOR_FILES = (
    "generations_complete.jsonl",
    "all_openai_framing_judgments.jsonl",
    "all_openai_package_judgments.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/submission/wilddelusion_anonymous_supplement"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("output/submission/wilddelusion_anonymous_supplement.zip"),
    )
    return parser.parse_args()


def copy_file(root: Path, stage: Path, relative: str | Path) -> Path:
    relative = Path(relative)
    source = root / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = stage / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def copy_tree(root: Path, stage: Path, relative: str) -> None:
    source = root / relative
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(
        source,
        stage / relative,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )


def redact_text_files(stage: Path) -> None:
    substitutions = (
        (
            "results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals/full_public_behavior",
            "results/full_public_behavior",
        ),
        (
            "results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals",
            "results/full_public_behavior",
        ),
        (
            "data/jspace/semantic_counterfactuals.jsonl",
            "results/full_public_behavior/prompts.jsonl",
        ),
        (
            'dataset = load_dataset("danielfein/WildDelusionCombined", split="train")',
            'dataset = load_dataset("json", data_files="data/releases/WildDelusionCombined/train.jsonl", split="train")',
        ),
        (
            "https://huggingface.co/datasets/danielfein/WildDelusionCombined",
            "ANONYMOUS_DATASET_URL_ADDED_AFTER_REVIEW",
        ),
        ("danielfein/WildDelusionCombined", "anonymous/WildDelusionCombined"),
        ("danielfein/WildDelusionVerified", "anonymous/WildDelusionVerified"),
        ("danielfein/WildDelusion", "anonymous/WildDelusion"),
        (
            "[`drfein/MentalHealthRedTeam`](https://github.com/drfein/MentalHealthRedTeam)",
            "the anonymized code in this supplement",
        ),
    )
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.suffix.lower() in {".pdf", ".png", ".parquet"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for old, new in substitutions:
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def filter_public_jsonl(source: Path, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(encoding="utf-8") as reader, destination.open(
        "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            row = json.loads(line)
            if row.get("source") == "lmsys_chat_1m":
                continue
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_readme(stage: Path) -> None:
    (stage / "README.md").write_text(
        """# WildDelusion anonymous supplement

This archive accompanies an anonymous submission. It contains the 522-row
redistributable benchmark, the 3,464 matched-frame generations and both judge
outputs for the 433-row primary discovery cohort, aggregate paper artifacts,
and the code needed to reproduce the mining and evaluation analyses. No
LMSYS-Chat-1M conversation text is included.

Install with `uv sync --extra paper --extra dev`, run tests with `uv run pytest`,
and rebuild the manuscript with `scripts/rebuild_paper.sh`. See
`docs/WILDDELUSION_COMBINED_DATASET_CARD.md` for licensing, privacy, and intended-use
constraints. The permanent public repository and dataset URLs are withheld
during double-blind review.
""",
        encoding="utf-8",
    )


def validate_anonymity(stage: Path) -> None:
    # Split literals keep author identifiers out of the generated supplement.
    blocked = (
        "dani" + "elfein",
        "dr" + "fein",
        "mentalhealth" + "redteam",
        "/users/" + "danielfein",
    )
    secret_patterns = (
        re.compile(rb"sk-proj-[A-Za-z0-9_-]{20,}"),
        re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    )
    failures: list[str] = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        content = path.read_bytes()
        lowered = content.lower()
        for token in blocked:
            if token.encode() in lowered:
                failures.append(f"{path.relative_to(stage)} contains {token!r}")
        for pattern in secret_patterns:
            if pattern.search(content):
                failures.append(
                    f"{path.relative_to(stage)} contains credential-shaped text"
                )
    if failures:
        raise RuntimeError("Anonymous supplement validation failed:\n" + "\n".join(failures))


def validate_contents(stage: Path) -> None:
    """Fail closed when the staged archive does not match its documented contract."""
    required_jsonl = {
        RELEASE_JSONL: PUBLIC_ROWS,
        Path("results/full_public_behavior/prompts.jsonl"): PUBLIC_BEHAVIOR_ROWS,
        Path("results/full_public_behavior/generations_complete.jsonl"): PUBLIC_BEHAVIOR_ROWS,
        Path(
            "results/full_public_behavior/all_openai_framing_judgments.jsonl"
        ): PUBLIC_BEHAVIOR_ROWS,
        Path(
            "results/full_public_behavior/all_openai_package_judgments.jsonl"
        ): PUBLIC_BEHAVIOR_ROWS,
    }
    for relative, expected_rows in required_jsonl.items():
        path = stage / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_rows = count_jsonl(path)
        if actual_rows != expected_rows:
            raise ValueError(f"{relative}: expected {expected_rows} rows, got {actual_rows}")

    for relative in (RELEASE_JSONL, RELEASE_MANIFEST):
        if not (stage / relative).is_file():
            raise FileNotFoundError(stage / relative)

    for relative in required_jsonl:
        with (stage / relative).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if row.get("source") == "lmsys_chat_1m":
                    raise ValueError(f"{relative}:{line_number}: contains excluded LMSYS text")

    documented_paths = (
        "data/releases/WildDelusionCombined/train.jsonl",
        "results/full_public_behavior/prompts.jsonl",
        "results/full_public_behavior/generations_complete.jsonl",
        "results/full_public_behavior/all_openai_framing_judgments.jsonl",
        "results/full_public_behavior/all_openai_package_judgments.jsonl",
    )
    corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in stage.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".json", ".py", ".toml"}
    )
    stale_path = "data/WildDelusionVerified/train.jsonl"
    if stale_path in corpus:
        raise ValueError(f"Staged supplement contains stale path {stale_path!r}")
    for documented_path in documented_paths:
        if not (stage / documented_path).is_file():
            raise FileNotFoundError(stage / documented_path)


def write_manifest(stage: Path) -> None:
    files = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files.append(
                {
                    "path": path.relative_to(stage).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    payload = {
        "archive_role": "anonymous_submission_supplement",
        "public_dataset_rows": PUBLIC_ROWS,
        "public_behavior_rows": PUBLIC_BEHAVIOR_ROWS,
        "lmsys_conversation_text_included": False,
        "files": files,
    }
    (stage / "MANIFEST.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_deterministic_zip(stage: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in sorted(stage.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.relative_to(stage).as_posix(), FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o100755 if path.relative_to(stage).parts[0] == "scripts" else 0o100644
            info.external_attr = mode << 16
            info.create_system = 3
            output.writestr(info, path.read_bytes(), compresslevel=9)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    stage = (root / args.output_dir).resolve()
    archive = (root / args.archive).resolve()
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    for relative in (
        ROOT_FILES + CONFIG_FILES + PAPER_FILES + ARTIFACT_FILES + SCRIPT_FILES + TEST_FILES
    ):
        copy_file(root, stage, relative)
    for relative in COPY_TREES:
        copy_tree(root, stage, relative)
    copy_file(root, stage, "docs/WILDDELUSION_COMBINED_DATASET_CARD.md")
    copy_file(root, stage, RELEASE_JSONL)
    copy_file(root, stage, RELEASE_MANIFEST)

    behavior_destination = stage / "results/full_public_behavior"
    prompt_count = filter_public_jsonl(
        root / PROMPTS_SOURCE, behavior_destination / "prompts.jsonl"
    )
    if prompt_count != PUBLIC_BEHAVIOR_ROWS:
        raise ValueError(
            f"prompts.jsonl: expected {PUBLIC_BEHAVIOR_ROWS} public rows, got {prompt_count}"
        )
    for name in BEHAVIOR_FILES:
        source = root / BEHAVIOR_ROOT / name
        destination = behavior_destination / name
        if name.endswith(".jsonl"):
            count = filter_public_jsonl(source, destination)
            if count != PUBLIC_BEHAVIOR_ROWS:
                raise ValueError(f"{name}: expected {PUBLIC_BEHAVIOR_ROWS} public rows, got {count}")
    (behavior_destination / "manifest.json").write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "decoding": "greedy",
                "max_new_tokens": 192,
                "target_turns": PRIMARY_BEHAVIOR_TARGETS,
                "source_conversations": 232,
                "intervention_arms": 8,
                "rows_per_artifact": PUBLIC_BEHAVIOR_ROWS,
                "excluded_sources": ["lmsys_chat_1m"],
                "framing_judge": "gpt-5.4-mini",
                "framing_judge_reasoning_effort": "low",
                "framing_positive_threshold": 4,
                "package_judge": "gpt-5.4-mini",
                "package_judge_reasoning_effort": "low",
                "package_positive_threshold": 7,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    write_readme(stage)
    redact_text_files(stage)
    validate_contents(stage)
    validate_anonymity(stage)
    write_manifest(stage)
    validate_anonymity(stage)
    write_deterministic_zip(stage, archive)
    print(f"Built {archive} ({archive.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
