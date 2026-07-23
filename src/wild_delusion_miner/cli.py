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
    watch_corpus_embedding_batches as watch_corpus_embedding_batches_impl,
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
    typer.echo(f"generating about {count_hint} synthetic messages with {config.models.synthetic_model}")
    count = generate_synthetic_messages(config, count_hint=count_hint)
    typer.echo(f"wrote {count} synthetic messages to {config.paths.synthetic_path}")


@app.command()
def extract_users(
    config_path: Path = Path("configs/default.yaml"),
    limit_rows_per_dataset: int | None = None,
    flush_every: int = 10_000,
    resume: bool = True,
) -> None:
    config = _config(config_path)
    store = DedupeStore(config.paths.sqlite_path)
    try:
        for spec in config.datasets:
            buffer = []
            start_offset = 0
            if resume:
                max_offset = store.max_row_offset(source=spec.source, split=spec.split)
                if max_offset is not None:
                    start_offset = max_offset + 1
            with tqdm(desc=f"extracting {spec.source}", initial=start_offset) as progress:
                for conversation in stream_conversations(
                    spec,
                    limit_rows=limit_rows_per_dataset,
                    start_offset=start_offset,
                ):
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
def prepare_synthetic_embedding_batches_only(config_path: Path = Path("configs/default.yaml")) -> None:
    config = _config(config_path)
    synthetic_count = prepare_synthetic_embedding_batches(config)
    typer.echo(f"prepared {synthetic_count} synthetic embedding requests")


@app.command()
def submit_synthetic_embedding_batches_only(config_path: Path = Path("configs/default.yaml")) -> None:
    config = _config(config_path)
    submit_embedding_batches_impl(config, batch_dir=config.paths.synthetic_embedding_dir / "batches")
    typer.echo("submitted synthetic embedding batches")


@app.command()
def watch_corpus_embedding_batches(
    config_path: Path = Path("configs/default.yaml"),
    follow_pid: int | None = None,
    idle_exit_seconds: int = 600,
    poll_seconds: int = 30,
    submit: bool = False,
) -> None:
    config = _config(config_path)
    count = watch_corpus_embedding_batches_impl(
        config,
        follow_pid=follow_pid,
        idle_exit_seconds=idle_exit_seconds,
        poll_seconds=poll_seconds,
        submit=submit,
    )
    typer.echo(f"watched {count} new corpus embedding requests")


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
def build_rerank_candidates(
    config_path: Path = Path("configs/default.yaml"),
    out_path: Path = Path("data/annotations/rerank_candidates.jsonl"),
    top_per_retrieval: int = 750,
    bin_samples_per_retrieval: int = 10,
    max_candidates: int = 2500,
) -> None:
    from wild_delusion_miner.rerank import build_rerank_candidates as build_candidates

    config = _config(config_path)
    count = build_candidates(
        config,
        out_path=out_path,
        top_per_retrieval=top_per_retrieval,
        bin_samples_per_retrieval=bin_samples_per_retrieval,
        max_candidates=max_candidates,
    )
    typer.echo(f"wrote {count} rerank candidates to {out_path}")


@app.command()
def annotate_rerank_canonical(
    config_path: Path = Path("configs/default.yaml"),
    input_path: Path = Path("data/annotations/rerank_candidates.jsonl"),
    output_path: Path = Path("data/annotations/rerank_canonical_annotations.jsonl"),
    model: str = "gpt-5.5",
    max_rows: int = 1000,
    resume: bool = True,
) -> None:
    from wild_delusion_miner.annotate import annotate_jsonl

    config = _config(config_path)
    count = annotate_jsonl(
        config,
        input_path=input_path,
        output_path=output_path,
        model=model,
        max_rows=max_rows,
        resume=resume,
    )
    typer.echo(f"canonical annotations present={count} at {output_path}")


@app.command()
def annotate_rerank_canonical_direct(
    config_path: Path = Path("configs/default.yaml"),
    input_path: Path = Path("data/annotations/rerank_candidates.jsonl"),
    output_path: Path = Path("data/annotations/rerank_canonical_gpt55_direct.jsonl"),
    model: str = "gpt-5.5",
    budget_usd: float = 20.0,
    max_rows: int | None = None,
    max_workers: int = 8,
    resume: bool = True,
) -> None:
    from wild_delusion_miner.annotate import annotate_jsonl_openai_direct

    config = _config(config_path)
    summary = annotate_jsonl_openai_direct(
        config,
        input_path=input_path,
        output_path=output_path,
        model=model,
        budget_usd=budget_usd,
        max_rows=max_rows,
        max_workers=max_workers,
        resume=resume,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


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
def collect_verification_conversations(
    config_path: Path = Path("configs/default.yaml"),
    candidates_path: Path = Path("data/annotations/true_positive_candidates_annotated.jsonl"),
    out_path: Path = Path("data/verification/candidate_conversations.jsonl"),
) -> None:
    from wild_delusion_miner.verify import collect_candidate_conversations

    config = _config(config_path)
    count = collect_candidate_conversations(
        config,
        candidates_path=candidates_path,
        out_path=out_path,
    )
    typer.echo(f"collected {count} candidate conversations at {out_path}")


@app.command()
def verify_conversations(
    config_path: Path = Path("configs/default.yaml"),
    conversations_path: Path = Path("data/verification/candidate_conversations.jsonl"),
    out_path: Path = Path("data/verification/verified_candidates.jsonl"),
    model: str | None = None,
    max_rows: int | None = None,
    max_workers: int = 8,
    resume: bool = True,
) -> None:
    from wild_delusion_miner.verify import verify_conversations as verify_impl

    config = _config(config_path)
    count = verify_impl(
        config,
        conversations_path=conversations_path,
        out_path=out_path,
        model=model,
        max_rows=max_rows,
        max_workers=max_workers,
        resume=resume,
    )
    typer.echo(f"verified conversations present={count} at {out_path}")


@app.command()
def evaluate_review_agreement(
    review_csv_path: Path,
    judge_output_path: Path = Path("data/verification/verified_candidates.jsonl"),
    out_path: Path = Path("data/verification/review_agreement.json"),
) -> None:
    from wild_delusion_miner.verify import evaluate_review_agreement as evaluate_impl

    summary = evaluate_impl(
        review_csv_path=review_csv_path,
        judge_output_path=judge_output_path,
        out_path=out_path,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command()
def import_review_conversations(
    review_csv_path: Path,
    out_path: Path = Path("data/verification/review_conversations.jsonl"),
) -> None:
    from wild_delusion_miner.verify import import_review_conversations as import_impl

    count = import_impl(review_csv_path=review_csv_path, out_path=out_path)
    typer.echo(f"imported {count} reviewed conversations at {out_path}")


@app.command()
def verify_review_csv(
    review_csv_path: Path,
    conversations_path: Path = Path("data/verification/review_conversations.jsonl"),
    judge_output_path: Path = Path("data/verification/review_judged.jsonl"),
    agreement_path: Path = Path("data/verification/review_agreement.json"),
    config_path: Path = Path("configs/default.yaml"),
    model: str | None = None,
    max_rows: int | None = None,
    max_workers: int = 8,
    resume: bool = True,
) -> None:
    from wild_delusion_miner.verify import (
        evaluate_review_agreement as evaluate_impl,
        import_review_conversations as import_impl,
        verify_conversations as verify_impl,
    )

    config = _config(config_path)
    import_impl(review_csv_path=review_csv_path, out_path=conversations_path)
    verify_impl(
        config,
        conversations_path=conversations_path,
        out_path=judge_output_path,
        model=model,
        max_rows=max_rows,
        max_workers=max_workers,
        resume=resume,
    )
    summary = evaluate_impl(
        review_csv_path=review_csv_path,
        judge_output_path=judge_output_path,
        out_path=agreement_path,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command()
def verify_final(config_path: Path = Path("configs/default.yaml")) -> None:
    config = _config(config_path)
    conversations_path = config.paths.verification_dir / "candidate_conversations.jsonl"
    collect_verification_conversations(
        config_path=config_path,
        candidates_path=config.paths.annotation_dir / "true_positive_candidates_annotated.jsonl",
        out_path=conversations_path,
    )
    verify_conversations(
        config_path=config_path,
        conversations_path=conversations_path,
        out_path=config.paths.verification_dir / "verified_candidates.jsonl",
    )


@app.command()
def snapshot_openai_response_models(
    out_path: Path = Path("data/generations/openai_response_model_snapshot.json"),
    include_pattern: list[str] | None = typer.Option(None, "--include-pattern"),
    exclude_pattern: list[str] | None = typer.Option(None, "--exclude-pattern"),
) -> None:
    from wild_delusion_miner.assistant_responses import snapshot_openai_generation_models

    snapshot = snapshot_openai_generation_models(
        out_path=out_path,
        include_patterns=include_pattern,
        exclude_patterns=exclude_pattern,
    )
    typer.echo(f"wrote {len(snapshot['models'])} OpenAI generation-model candidates to {out_path}")


@app.command()
def generate_post_delusion_responses(
    config_path: Path = Path("configs/default.yaml"),
    input_path: Path = Path("data/verification/whitened_top3k_verified_positive_contexts.jsonl"),
    out_path: Path = Path("data/generations/post_delusion_openai_responses.jsonl"),
    manifest_path: Path | None = None,
    model_snapshot_path: Path | None = Path("data/generations/openai_response_model_snapshot.json"),
    model: list[str] | None = typer.Option(None, "--model"),
    max_rows: int | None = None,
    max_models: int | None = None,
    max_workers: int = 8,
    max_output_tokens: int = 800,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    system_prompt: str | None = None,
    no_system_prompt: bool = typer.Option(False, "--no-system-prompt"),
    resume: bool = True,
    retry_errors: bool = False,
) -> None:
    from wild_delusion_miner.assistant_responses import (
        DEFAULT_ASSISTANT_RESPONSE_SYSTEM_PROMPT,
        generate_post_delusion_responses as generate_impl,
    )

    config = _config(config_path)
    summary = generate_impl(
        config,
        input_path=input_path,
        out_path=out_path,
        manifest_path=manifest_path,
        models=model,
        model_snapshot_path=model_snapshot_path if not model else None,
        max_rows=max_rows,
        max_models=max_models,
        max_workers=max_workers,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        system_prompt=(
            None
            if no_system_prompt
            else system_prompt or DEFAULT_ASSISTANT_RESPONSE_SYSTEM_PROMPT
        ),
        resume=resume,
        retry_errors=retry_errors,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command()
def analyze_generated_responses(
    input_path: Path = Path("data/generations/post_delusion_openai_responses_10model_low_reasoning.jsonl"),
    out_dir: Path = Path("results/response_analysis/10model_low_reasoning"),
    embedding_model: str = "text-embedding-3-small",
    label_model: str = "gpt-5.4-mini",
    label_reasoning_effort: str | None = "low",
    embedding_batch_size: int = 128,
    cluster_count: int = 16,
    topic_count: int = 16,
    random_state: int = 13,
    min_df: int = 3,
    max_df: float = 0.85,
    max_features: int = 6000,
    representative_count: int = 8,
) -> None:
    from wild_delusion_miner.response_analysis import (
        ResponseAnalysisParams,
        analyze_generated_responses as analyze_impl,
    )

    summary = analyze_impl(
        input_path=input_path,
        out_dir=out_dir,
        params=ResponseAnalysisParams(
            embedding_model=embedding_model,
            label_model=label_model,
            label_reasoning_effort=label_reasoning_effort,
            embedding_batch_size=embedding_batch_size,
            cluster_count=cluster_count,
            topic_count=topic_count,
            random_state=random_state,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            representative_count=representative_count,
        ),
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


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
