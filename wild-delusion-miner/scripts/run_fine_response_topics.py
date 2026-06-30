from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from openai import APIStatusError, OpenAI, OpenAIError
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


MODEL_DATES = {
    "gpt-3.5-turbo-0125": "2024-01-23",
    "gpt-4-turbo-2024-04-09": "2024-04-08",
    "gpt-4o-2024-05-13": "2024-05-10",
    "gpt-4o-mini-2024-07-18": "2024-07-16",
    "o1-2024-12-17": "2024-12-16",
    "o3-mini-2025-01-31": "2025-01-27",
    "gpt-4.1-mini-2025-04-14": "2025-04-10",
    "gpt-5-mini-2025-08-07": "2025-08-05",
    "gpt-5.2-2025-12-11": "2025-12-09",
    "gpt-5.5-2026-04-23": "2026-04-22",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--topic-count", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--label-model", default="gpt-5.4-mini")
    parser.add_argument("--label-reasoning-effort", default="low")
    parser.add_argument("--max-features", type=int, default=14_000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-df", type=float, default=0.8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.input_path.open(encoding="utf-8") if line.strip()]
    texts = [str(row.get("response_text") or "") for row in rows]
    print(f"loaded {len(rows)} successful responses", flush=True)

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z][a-zA-Z]+\b",
    )
    counts = vectorizer.fit_transform(texts)
    print(f"count matrix {counts.shape}", flush=True)

    lda = LatentDirichletAllocation(
        n_components=args.topic_count,
        random_state=args.random_state,
        learning_method="batch",
        max_iter=35,
        evaluate_every=5,
        verbose=1,
    )
    doc_topics = lda.fit_transform(counts)
    topic_ids = doc_topics.argmax(axis=1)
    feature_names = np.asarray(vectorizer.get_feature_names_out())

    write_assignments(args.out_dir / "fine_topic_assignments.csv", rows, topic_ids, doc_topics)
    topics = build_topics(rows, lda.components_, doc_topics, topic_ids, feature_names)
    (args.out_dir / "fine_topics.json").write_text(
        json.dumps(topics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_over_time(args.out_dir / "fine_topic_over_time.csv", rows, topic_ids, args.topic_count)

    labels = []
    for topic in tqdm(topics, desc="labeling fine topics"):
        labels.append(
            {
                "id": topic["id"],
                "size": topic["size"],
                "share": topic["share"],
                "label": label_topic(
                    topic,
                    model=args.label_model,
                    reasoning_effort=args.label_reasoning_effort,
                ),
            }
        )
    (args.out_dir / "fine_topic_labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "input_path": str(args.input_path),
        "out_dir": str(args.out_dir),
        "input_rows": len(rows),
        "topic_count": args.topic_count,
        "random_state": args.random_state,
        "vectorizer": {
            "ngram_range": [1, 3],
            "min_df": args.min_df,
            "max_df": args.max_df,
            "max_features": args.max_features,
        },
        "label_model": args.label_model,
        "label_reasoning_effort": args.label_reasoning_effort,
        "outputs": [path.name for path in sorted(args.out_dir.iterdir())],
        "labels": labels,
    }
    (args.out_dir / "fine_topic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out_dir), "topics": args.topic_count, "rows": len(rows)}, indent=2))


def write_assignments(
    path: Path,
    rows: list[dict[str, Any]],
    topic_ids: np.ndarray,
    doc_topics: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "generation_id",
                "model",
                "date",
                "topic",
                "topic_probability",
                "target_text",
                "response_text",
            ],
        )
        writer.writeheader()
        for row, topic_id, probabilities in zip(rows, topic_ids, doc_topics, strict=True):
            writer.writerow(
                {
                    "generation_id": row["generation_id"],
                    "model": row["model"],
                    "date": MODEL_DATES.get(row["model"], ""),
                    "topic": int(topic_id),
                    "topic_probability": float(probabilities[int(topic_id)]),
                    "target_text": row.get("target_text") or "",
                    "response_text": row.get("response_text") or "",
                }
            )


def build_topics(
    rows: list[dict[str, Any]],
    components: np.ndarray,
    doc_topics: np.ndarray,
    topic_ids: np.ndarray,
    feature_names: np.ndarray,
) -> list[dict[str, Any]]:
    topics = []
    for topic_id, weights in enumerate(components):
        indices = np.flatnonzero(topic_ids == topic_id)
        if len(indices):
            representative_indices = indices[np.argsort(doc_topics[indices, topic_id])[::-1][:7]]
        else:
            representative_indices = []
        topics.append(
            {
                "id": int(topic_id),
                "size": int(len(indices)),
                "share": float(len(indices) / len(rows)),
                "top_terms": feature_names[np.argsort(weights)[-25:][::-1]].tolist(),
                "representatives": [_representative_row(rows[int(index)]) for index in representative_indices],
            }
        )
    return sorted(topics, key=lambda topic: topic["size"], reverse=True)


def write_over_time(path: Path, rows: list[dict[str, Any]], topic_ids: np.ndarray, topic_count: int) -> None:
    models = sorted(MODEL_DATES, key=lambda model: MODEL_DATES[model])
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "date", "n"] + [f"topic_{topic_id}" for topic_id in range(topic_count)]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in models:
            indices = [index for index, row in enumerate(rows) if row["model"] == model]
            n = len(indices)
            record = {"model": model, "date": MODEL_DATES[model], "n": n}
            for topic_id in range(topic_count):
                record[f"topic_{topic_id}"] = (
                    sum(1 for index in indices if int(topic_ids[index]) == topic_id) / n if n else 0
                )
            writer.writerow(record)


def label_topic(topic: dict[str, Any], *, model: str, reasoning_effort: str | None) -> dict[str, Any]:
    client = OpenAI()
    examples = []
    for index, row in enumerate(topic["representatives"][:5], start=1):
        target = _clip(row.get("target_text", ""), 280)
        response = _clip(row.get("response_text", ""), 700)
        examples.append(
            f"Example {index}\n"
            f"Flagged user message: {target}\n"
            f"Assistant response: {response}"
        )
    prompt = (
        "Give a neutral descriptive label for this topic of assistant responses. "
        "Do not use normative words such as safe, unsafe, harmful, good, bad, delusion, "
        "or hallucination. Return compact JSON with keys: title, description, "
        "response_moves, boundary_style. Title should be 2-6 words and describe the "
        "response pattern.\n"
        f"Topic size: {topic['size']}\n"
        f"Top terms: {', '.join(topic['top_terms'][:25])}\n\n"
        + "\n\n".join(examples)
    )
    payload: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": "You create neutral qualitative topic labels."},
            {"role": "user", "content": prompt},
        ],
        "max_output_tokens": 500,
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    for attempt in range(1, 5):
        try:
            response = client.responses.create(**payload)
            parsed = _loads_json_object(_response_text(response).strip())
            if parsed is None:
                parsed = {"title": "Unparsed topic", "description": _response_text(response).strip()}
            parsed["model"] = model
            parsed["reasoning_effort"] = reasoning_effort
            return parsed
        except APIStatusError as error:
            if _is_unsupported_reasoning_error(error) and "reasoning" in payload:
                payload.pop("reasoning")
                continue
            if error.status_code not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(min(10 * attempt, 60))
        except OpenAIError:
            raise
    raise RuntimeError("Topic label request failed.")


def _representative_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "generation_id": row.get("generation_id"),
        "model": row.get("model"),
        "target_text": row.get("target_text") or "",
        "response_text": row.get("response_text") or "",
        "response_chars": len(str(row.get("response_text") or "")),
    }


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks)


def _loads_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_unsupported_reasoning_error(error: APIStatusError) -> bool:
    text = str(error).lower()
    return "unsupported parameter" in text and "reasoning.effort" in text


def _clip(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


if __name__ == "__main__":
    main()
