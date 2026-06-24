from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    split: str = "train"
    config: str | None = None
    gated: bool = False


@dataclass(frozen=True)
class Paths:
    data_dir: Path = Path("data")
    sqlite_path: Path = Path("data/dedupe.sqlite")
    synthetic_path: Path = Path("data/synthetic_delusion_messages.jsonl")
    corpus_embedding_dir: Path = Path("data/embeddings/corpus")
    synthetic_embedding_dir: Path = Path("data/embeddings/synthetic")
    retrieval_dir: Path = Path("data/retrieval")
    annotation_dir: Path = Path("data/annotations")
    verification_dir: Path = Path("data/verification")


@dataclass(frozen=True)
class ModelConfig:
    synthetic_model: str = "gpt-5.5"
    annotation_model: str = "openai/gpt-5.1-2025-11-13"
    verifier_model: str = "gpt-5.5"
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"


@dataclass(frozen=True)
class OpenAIConfig:
    reasoning_effort: str = "low"
    timeout_seconds: int = 300


@dataclass(frozen=True)
class AnnotationConfig:
    annotation_id: str = "user-endorses-delusion"
    cutoff: int = 7
    preceding_count: int = 0
    batch_size: int = 256
    max_workers: int = 32
    timeout_seconds: int = 120


@dataclass(frozen=True)
class RetrievalConfig:
    initial_top_k: int = 500_000
    calibration_bins: int = 20
    calibration_sample_per_bin: int = 100
    target_hit_rate: float = 0.10
    above_threshold_margin: float = 0.0
    second_pass_top_k: int = 500_000


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "openai_batch"
    batch_request_size: int = 50_000
    poll_seconds: int = 120
    shard_size: int = 100_000
    normalize: bool = True
    dtype: str = "float16"
    dimensions: int | None = None
    query_instruction: str = (
        "Retrieve user messages that explicitly endorse or demonstrate genuine "
        "delusional beliefs rather than fiction, hypotheticals, jokes, or commonly "
        "held beliefs."
    )


@dataclass(frozen=True)
class PipelineConfig:
    paths: Paths = field(default_factory=Paths)
    models: ModelConfig = field(default_factory=ModelConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    datasets: tuple[DatasetSpec, ...] = field(default_factory=tuple)


def _coerce_path_dict(raw: dict[str, Any]) -> dict[str, Path]:
    return {key: Path(value) for key, value in raw.items()}


def load_config(path: str | Path) -> PipelineConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return PipelineConfig(
        paths=Paths(**_coerce_path_dict(raw.get("paths", {}))),
        models=ModelConfig(**raw.get("models", {})),
        openai=OpenAIConfig(**raw.get("openai", {})),
        annotation=AnnotationConfig(**raw.get("annotation", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        datasets=tuple(DatasetSpec(**item) for item in raw.get("datasets", [])),
    )


def ensure_dirs(config: PipelineConfig) -> None:
    for path in (
        config.paths.data_dir,
        config.paths.corpus_embedding_dir,
        config.paths.synthetic_embedding_dir,
        config.paths.retrieval_dir,
        config.paths.annotation_dir,
        config.paths.verification_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
