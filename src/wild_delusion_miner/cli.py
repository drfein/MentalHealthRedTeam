from __future__ import annotations

import json
from pathlib import Path

import typer
from tqdm import tqdm

from wild_delusion_miner.config import ensure_dirs, load_config
from wild_delusion_miner.datasets import iter_user_messages, stream_conversations
from wild_delusion_miner.embed import (
    download_embedding_batch_outputs as download_embedding_batch_outputs_impl,
    materialize_corpus_embeddings,
    materialize_synthetic_embeddings,
    materialize_text_mean_embeddings,
    prepare_corpus_embedding_batches,
    prepare_synthetic_embedding_batches,
    prepare_text_embedding_batches,
    refresh_embedding_batches as refresh_embedding_batches_impl,
    submit_embedding_batches as submit_embedding_batches_impl,
    wait_for_embedding_batches as wait_for_embedding_batches_impl,
)
from wild_delusion_miner.monitor import pipeline_status, write_inspection_html
from wild_delusion_miner.retrieval import (
    build_query_from_means,
    choose_threshold_from_calibration,
    export_above_threshold,
    sample_calibration_set,
    score_corpus,
)
from wild_delusion_miner.store import DedupeStore
from wild_delusion_miner.synthetic import generate_synthetic_messages

app = typer.Typer(no_args_is_help=True)


def _config(path: Path):
    config = load_config(path)
    ensure_dirs(config)
    return config


@app.command()
def generate_synthetic(config_path: Path = Path("configs/default.yaml"), count_hint: int = 1000) -> None:
    config = _config(config_path)
    count = generate_synthetic_messages(config, count_hint=count_hint)
    typer.echo(f"wrote {count} synthetic messages to {config.paths.synthetic_path}")


@app.command()
def extract_users(
    config_path: Path = Path("configs/default.yaml"),
    limit_rows_per_dataset: int | None = None,
    flush_every: int = 10_000,
) -> None:
    config = _config(config_path)
    store = DedupeStore(config.paths.sqlite_path)
    try:
        for spec in config.datasets:
            buffer = []
            with tqdm(desc=f"extracting {spec.source}") as progress:
                for conversation in stream_conversations(spec, limit_rows=limit_rows_per_dataset):
                    buffer.extend(iter_user_messages(conversation))
                    progress.update(1)
                    if len(buffer) >= flush_every:
                        new_count, ref_count = store.add_many(buffer)
                        progress.set_postfix(unique=store.message_count(), new=new_count, refs=ref_count)
                        buffer = []
                if buffer:
                    new_count, ref_count = store.add_many(buffer)
                    progress.set_postfix(unique=store.message_count(), new=new_count, refs=ref_count)
    finally:
        store.close()


@app.command()
def prepare_embedding_batches(
    config_path: Path = Path("configs/default.yaml"),
    limit_messages: int | None = None,
) -> None:
    config = _config(config_path)
    corpus_count = prepare_corpus_embedding_batches(config, limit_messages=limit_messages)
    synthetic_count = prepare_synthetic_embedding_batches(config)
    typer.echo(f"prepared {corpus_count} corpus and {synthetic_count} synthetic embedding requests")


@app.command()
def submit_embedding_batches(config_path: Path = Path("configs/default.yaml")) -> None:
    config = _config(config_path)
    submit_embedding_batches_impl(config, batch_dir=config.paths.corpus_embedding_dir / "batches")
    submit_embedding_batches_impl(config, batch_dir=config.paths.synthetic_embedding_dir / "batches")
    typer.echo("submitted corpus and synthetic embedding batches")


@app.command()
def poll_embedding_batches(
    config_path: Path = Path("configs/default.yaml"),
    wait: bool = False,
) -> None:
    config = _config(config_path)
    refresh = wait_for_embedding_batches_impl if wait else refresh_embedding_batches_impl
    corpus = refresh(config, batch_dir=config.paths.corpus_embedding_dir / "batches")
    synthetic = refresh(config, batch_dir=config.paths.synthetic_embedding_dir / "batches")
    typer.echo(
        json.dumps(
            {
                "corpus": [(item.get("index"), item.get("status")) for item in corpus["batches"]],
                "synthetic": [(item.get("index"), item.get("status")) for item in synthetic["batches"]],
            },
            indent=2,
        )
    )


@app.command()
def download_embedding_outputs(config_path: Path = Path("configs/default.yaml")) -> None:
    config = _config(config_path)
    download_embedding_batch_outputs_impl(config, batch_dir=config.paths.corpus_embedding_dir / "batches")
    download_embedding_batch_outputs_impl(config, batch_dir=config.paths.synthetic_embedding_dir / "batches")
    typer.echo("downloaded available embedding batch outputs")


@app.command()
def materialize_embeddings(config_path: Path = Path("configs/default.yaml")) -> None:
    config = _config(config_path)
    corpus_count = materialize_corpus_embeddings(config)
    synthetic_count = materialize_synthetic_embeddings(config)
    typer.echo(f"materialized {corpus_count} corpus and {synthetic_count} synthetic embeddings")


@app.command()
def retrieve_initial(config_path: Path = Path("configs/default.yaml")) -> None:
    config = _config(config_path)
    query_path = config.paths.retrieval_dir / "initial_query.npy"
    build_query_from_means(
        config.paths.corpus_embedding_dir / "document_mean.npy",
        config.paths.synthetic_embedding_dir / "mean.npy",
        query_path,
    )
    out_dir = score_corpus(
        config,
        query_path=query_path,
        out_name="initial",
        top_k=config.retrieval.initial_top_k,
    )
    typer.echo(f"wrote initial retrieval to {out_dir}")


@app.command()
def make_calibration(config_path: Path = Path("configs/default.yaml")) -> None:
    from wild_delusion_miner.annotate import annotate_jsonl

    config = _config(config_path)
    out_path = config.paths.annotation_dir / "calibration_sample.jsonl"
    sample_calibration_set(
        config,
        scores_path=config.paths.retrieval_dir / "initial" / "scores.parquet",
        out_path=out_path,
    )
    annotated_path = config.paths.annotation_dir / "calibration_annotations.jsonl"
    annotate_jsonl(config, input_path=out_path, output_path=annotated_path)
    threshold = choose_threshold_from_calibration(
        annotated_path,
        target_hit_rate=config.retrieval.target_hit_rate,
    )
    (config.paths.annotation_dir / "initial_threshold.txt").write_text(str(threshold), encoding="utf-8")
    typer.echo(f"initial threshold: {threshold:.6f}")


@app.command()
def annotate_initial_above_threshold(config_path: Path = Path("configs/default.yaml")) -> None:
    from wild_delusion_miner.annotate import annotate_jsonl

    config = _config(config_path)
    threshold = float((config.paths.annotation_dir / "initial_threshold.txt").read_text(encoding="utf-8"))
    candidate_path = config.paths.annotation_dir / "initial_candidates.jsonl"
    count = export_above_threshold(
        config,
        scores_path=config.paths.retrieval_dir / "initial" / "scores.parquet",
        threshold=threshold + config.retrieval.above_threshold_margin,
        out_path=candidate_path,
    )
    annotated_path = config.paths.annotation_dir / "initial_candidates_annotated.jsonl"
    annotate_jsonl(config, input_path=candidate_path, output_path=annotated_path)
    typer.echo(f"annotated {count} initial above-threshold candidates")


@app.command()
def retrieve_from_true_positives(config_path: Path = Path("configs/default.yaml")) -> None:
    from wild_delusion_miner.annotate import positive_texts

    config = _config(config_path)
    true_positive_mean = config.paths.retrieval_dir / "true_positive_mean.npy"
    texts = positive_texts(config.paths.annotation_dir / "initial_candidates_annotated.jsonl")
    batch_dir = config.paths.retrieval_dir / "true_positive_embedding_batches"
    prepare_text_embedding_batches(config, texts, out_dir=batch_dir, kind="true_positive")
    submit_embedding_batches_impl(config, batch_dir=batch_dir)
    wait_for_embedding_batches_impl(config, batch_dir=batch_dir)
    download_embedding_batch_outputs_impl(config, batch_dir=batch_dir)
    materialize_text_mean_embeddings(config, batch_dir=batch_dir, out_path=true_positive_mean)
    query_path = config.paths.retrieval_dir / "true_positive_query.npy"
    build_query_from_means(
        config.paths.corpus_embedding_dir / "document_mean.npy",
        true_positive_mean,
        query_path,
    )
    out_dir = score_corpus(
        config,
        query_path=query_path,
        out_name="true_positive",
        top_k=config.retrieval.second_pass_top_k,
    )
    typer.echo(f"wrote true-positive retrieval to {out_dir}")


@app.command()
def annotate_true_positive_pass(config_path: Path = Path("configs/default.yaml")) -> None:
    from wild_delusion_miner.annotate import annotate_jsonl

    config = _config(config_path)
    out_path = config.paths.annotation_dir / "true_positive_calibration_sample.jsonl"
    sample_calibration_set(
        config,
        scores_path=config.paths.retrieval_dir / "true_positive" / "scores.parquet",
        out_path=out_path,
    )
    calibration_annotations = config.paths.annotation_dir / "true_positive_calibration_annotations.jsonl"
    annotate_jsonl(config, input_path=out_path, output_path=calibration_annotations)
    threshold = choose_threshold_from_calibration(
        calibration_annotations,
        target_hit_rate=config.retrieval.target_hit_rate,
    )
    (config.paths.annotation_dir / "true_positive_threshold.txt").write_text(
        str(threshold), encoding="utf-8"
    )
    candidate_path = config.paths.annotation_dir / "true_positive_candidates.jsonl"
    count = export_above_threshold(
        config,
        scores_path=config.paths.retrieval_dir / "true_positive" / "scores.parquet",
        threshold=threshold + config.retrieval.above_threshold_margin,
        out_path=candidate_path,
    )
    annotate_jsonl(
        config,
        input_path=candidate_path,
        output_path=config.paths.annotation_dir / "true_positive_candidates_annotated.jsonl",
    )
    typer.echo(f"annotated {count} true-positive-pass candidates")


@app.command()
def verify_final(config_path: Path = Path("configs/default.yaml")) -> None:
    from wild_delusion_miner.verify import collect_candidate_conversations, verify_conversations

    config = _config(config_path)
    conversations_path = config.paths.verification_dir / "candidate_conversations.jsonl"
    count = collect_candidate_conversations(
        config,
        candidates_path=config.paths.annotation_dir / "true_positive_candidates_annotated.jsonl",
        out_path=conversations_path,
    )
    verify_conversations(
        config,
        conversations_path=conversations_path,
        out_path=config.paths.verification_dir / "verified_candidates.jsonl",
    )
    typer.echo(f"verified {count} candidate conversations")


@app.command()
def status(config_path: Path = Path("configs/default.yaml")) -> None:
    config = _config(config_path)
    typer.echo(json.dumps(pipeline_status(config), indent=2, sort_keys=True))


@app.command()
def inspect_samples(
    config_path: Path = Path("configs/default.yaml"),
    out_path: Path = Path("artifacts/inspection.html"),
    per_source: int = 25,
    positives_only: bool = True,
) -> None:
    config = _config(config_path)
    count = write_inspection_html(
        config,
        out_path=out_path,
        per_source=per_source,
        positives_only=positives_only,
    )
    typer.echo(f"wrote {count} review samples to {out_path}")


@app.command()
def run_local_smoke(config_path: Path = Path("configs/default.yaml")) -> None:
    """Run extraction and embedding on tiny limits to validate plumbing."""
    extract_users(config_path=config_path, limit_rows_per_dataset=10, flush_every=100)
    prepare_embedding_batches(config_path=config_path, limit_messages=100)


if __name__ == "__main__":
    app()
