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


PUBLIC_ROWS = 433
PUBLIC_BEHAVIOR_ROWS = 3_464
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)

ROOT_FILES = ("pyproject.toml", "uv.lock")
COPY_TREES = (
    "src/wild_delusion_miner",
    "tests",
    "paper/iclr2026/artifacts",
    "paper/iclr2026/figures",
)
CONFIG_FILES = (
    "configs/default.yaml",
    "configs/wilddelusion_benchmark_protocol.json",
    "configs/wilddelusion_release_summary.json",
)
PAPER_FILES = (
    "paper/iclr2026/CLAIM_AUDIT.md",
    "paper/iclr2026/DATASET_HUMAN_AUDIT_PROTOCOL.md",
    "paper/iclr2026/HUMAN_AUDIT_PROTOCOL.md",
    "paper/iclr2026/REPRODUCE.md",
    "paper/iclr2026/iclr2026_conference.bst",
    "paper/iclr2026/iclr2026_conference.sty",
    "paper/iclr2026/main.tex",
    "paper/iclr2026/math_commands.tex",
    "paper/iclr2026/references.bib",
    "output/pdf/wild_delusion_iclr2026.pdf",
)
SCRIPT_FILES = (
    "scripts/analyze_conversation_verifier_human_validation.py",
    "scripts/analyze_counterfactual_human_audit.py",
    "scripts/analyze_full_behavior_human_audit.py",
    "scripts/analyze_full_counterfactual_behavior.py",
    "scripts/analyze_jspace_conversation_disjoint_holdout.py",
    "scripts/analyze_jspace_headline_probe_diagnostics.py",
    "scripts/analyze_jspace_primary_endpoint_robustness.py",
    "scripts/analyze_jspace_semantic_specificity.py",
    "scripts/analyze_jspace_true_neutral_control.py",
    "scripts/analyze_wilddelusion_dataset.py",
    "scripts/analyze_wilddelusion_release_human_audit.py",
    "scripts/benchmark_openai_embeddings.py",
    "scripts/bootstrap_embedding_mechanisms.py",
    "scripts/build_counterfactual_judge_audit.py",
    "scripts/build_jspace_group_split.py",
    "scripts/build_jspace_semantic_counterfactuals.py",
    "scripts/build_jspace_true_neutral_controls.py",
    "scripts/build_wilddelusion_public_release.py",
    "scripts/build_wilddelusion_release_human_audit.py",
    "scripts/finalize_openai_embeddings_and_retrieve.py",
    "scripts/generate_jspace_matched_responses.py",
    "scripts/judge_generated_responses_with_package.py",
    "scripts/judge_semantic_counterfactual_responses.py",
    "scripts/judge_semantic_counterfactual_responses_openai.py",
    "scripts/make_wilddelusion_release_audit_html.py",
    "scripts/plot_full_counterfactual_behavior.py",
    "scripts/plot_jspace_paper_main_figure_v2.py",
    "scripts/plot_wilddelusion_dataset_overview.py",
    "scripts/populate_verification_contexts.py",
    "scripts/rebuild_paper.sh",
    "scripts/run_hypothetical_bucket_calibration.py",
    "scripts/run_jspace_semantic_counterfactuals.py",
    "scripts/verify_wilddelusion_release.py",
    "scripts/verify_paper_claims.py",
)
BEHAVIOR_ROOT = Path(
    "results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals"
)
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
            'dataset = load_dataset("danielfein/WildDelusionVerified", split="train")',
            'dataset = load_dataset("json", data_files="data/WildDelusionVerified/train.jsonl", split="train")',
        ),
        (
            "https://huggingface.co/datasets/danielfein/WildDelusionVerified",
            "ANONYMOUS_DATASET_URL_ADDED_AFTER_REVIEW",
        ),
        (
            "hf://datasets/danielfein/WildDelusionVerified@7639c02b593e437bcea32259db4c0a5da29bd79d/train.jsonl",
            "supplement://data/WildDelusionVerified/train.jsonl",
        ),
        ("danielfein/WildDelusionVerified", "anonymous/WildDelusionVerified"),
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

This archive accompanies an anonymous submission. It contains the 433-row
redistributable benchmark, the 3,464 public matched-frame generations and both
judge outputs, aggregate paper artifacts, and the code needed to reproduce the
mining and evaluation analyses. No LMSYS-Chat-1M conversation text is included.

Install with `uv sync --extra paper --extra dev`, run tests with `uv run pytest`,
and rebuild the manuscript with `scripts/rebuild_paper.sh`. See
`docs/WILDDELUSION_DATASET_CARD.md` for licensing, privacy, and intended-use
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

    for relative in ROOT_FILES + CONFIG_FILES + PAPER_FILES + SCRIPT_FILES:
        copy_file(root, stage, relative)
    for relative in COPY_TREES:
        copy_tree(root, stage, relative)
    copy_file(root, stage, "docs/WILDDELUSION_DATASET_CARD.md")
    copy_file(root, stage, "data/releases/WildDelusionVerified/train.jsonl")
    copy_file(root, stage, "data/releases/WildDelusionVerified/manifest.json")

    behavior_destination = stage / "results/full_public_behavior"
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
                "target_turns": PUBLIC_ROWS,
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

    if count_jsonl(stage / "data/releases/WildDelusionVerified/train.jsonl") != PUBLIC_ROWS:
        raise ValueError("Release JSONL does not contain 433 rows")

    write_readme(stage)
    redact_text_files(stage)
    validate_anonymity(stage)
    write_manifest(stage)
    validate_anonymity(stage)
    write_deterministic_zip(stage, archive)
    print(f"Built {archive} ({archive.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
