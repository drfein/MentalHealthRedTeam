#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sqlite3
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError, RateLimitError


@dataclass
class BenchResult:
    concurrency: int
    batch_size: int
    requested_messages: int
    completed_messages: int
    completed_requests: int
    failed_requests: int
    rate_limit_events: int
    retry_events: int
    elapsed_seconds: float
    messages_per_second: float
    requests_per_minute: float
    median_request_seconds: float | None
    p95_request_seconds: float | None
    mean_chars_per_message: float
    estimated_tokens_per_second: float


def _load_api_key_from_pid(pid: int) -> str | None:
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    prefix = b"OPENAI_API_KEY="
    for item in environ:
        if item.startswith(prefix):
            return item[len(prefix) :].decode("utf-8")
    return None


def _load_api_key(pid: int | None) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    if pid is not None:
        key = _load_api_key_from_pid(pid)
        if key:
            return key
    raise RuntimeError("OPENAI_API_KEY is not set and no key was found in the provided process env")


def _sample_messages(db_path: Path, *, sample_size: int, min_len: int, max_len: int, seed: int) -> list[str]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    rng = random.Random(seed)
    starts = [1, 100_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000]
    rng.shuffle(starts)
    messages: list[str] = []
    per_start = max(250, sample_size // len(starts) + 1)
    for start in starts:
        rows = conn.execute(
            """
            SELECT text
            FROM messages
            WHERE rowid >= ? AND text_len BETWEEN ? AND ?
            ORDER BY rowid
            LIMIT ?
            """,
            (start, min_len, max_len, per_start),
        ).fetchall()
        messages.extend(str(row[0]) for row in rows)
        if len(messages) >= sample_size:
            break
    if len(messages) < sample_size:
        rows = conn.execute(
            """
            SELECT text
            FROM messages
            WHERE text_len BETWEEN ? AND ?
            ORDER BY rowid
            LIMIT ?
            """,
            (min_len, max_len, sample_size - len(messages)),
        ).fetchall()
        messages.extend(str(row[0]) for row in rows)
    conn.close()
    rng.shuffle(messages)
    return messages[:sample_size]


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _retry_after_seconds(exc: RateLimitError) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    value = response.headers.get("retry-after") or response.headers.get("retry-after-ms")
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if "retry-after-ms" in response.headers and seconds > 10:
        return seconds / 1000
    return seconds


async def _embed_with_retries(
    client: AsyncOpenAI,
    *,
    model: str,
    inputs: list[str],
    max_retries: int,
    timeout_seconds: float,
    dimensions: int | None,
) -> tuple[int, float, int, int]:
    retry_events = 0
    rate_limit_events = 0
    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "input": inputs,
                "encoding_format": "float",
                "timeout": timeout_seconds,
            }
            if dimensions is not None:
                kwargs["dimensions"] = dimensions
            response = await client.embeddings.create(**kwargs)
            elapsed = time.perf_counter() - started
            if len(response.data) != len(inputs):
                raise RuntimeError(f"expected {len(inputs)} embeddings, got {len(response.data)}")
            return len(inputs), elapsed, retry_events, rate_limit_events
        except BadRequestError:
            if len(inputs) == 1:
                raise
            midpoint = len(inputs) // 2
            left = await _embed_with_retries(
                client,
                model=model,
                inputs=inputs[:midpoint],
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                dimensions=dimensions,
            )
            right = await _embed_with_retries(
                client,
                model=model,
                inputs=inputs[midpoint:],
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                dimensions=dimensions,
            )
            return (
                left[0] + right[0],
                left[1] + right[1],
                left[2] + right[2] + 1,
                left[3] + right[3],
            )
        except RateLimitError as exc:
            rate_limit_events += 1
            retry_events += 1
            if attempt >= max_retries:
                raise
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = min(60.0, 1.5 * (2**attempt)) + random.random()
            await asyncio.sleep(delay)
        except (APITimeoutError, APIConnectionError):
            retry_events += 1
            if attempt >= max_retries:
                raise
            await asyncio.sleep(min(30.0, 1.0 * (2**attempt)) + random.random())
    raise RuntimeError("unreachable retry loop exit")


async def _run_one(
    *,
    api_key: str,
    model: str,
    messages: list[str],
    batch_size: int,
    concurrency: int,
    max_retries: int,
    timeout_seconds: float,
    dimensions: int | None,
) -> BenchResult:
    client = AsyncOpenAI(api_key=api_key, max_retries=0)
    batches = _chunks(messages, batch_size)
    queue: asyncio.Queue[list[str]] = asyncio.Queue()
    for batch in batches:
        queue.put_nowait(batch)

    completed_messages = 0
    completed_requests = 0
    failed_requests = 0
    rate_limit_events = 0
    retry_events = 0
    request_times: list[float] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal completed_messages, completed_requests, failed_requests, rate_limit_events, retry_events
        while True:
            try:
                batch = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                count, elapsed, retries, rate_limits = await _embed_with_retries(
                    client,
                    model=model,
                    inputs=batch,
                    max_retries=max_retries,
                    timeout_seconds=timeout_seconds,
                    dimensions=dimensions,
                )
                async with lock:
                    completed_messages += count
                    completed_requests += 1
                    retry_events += retries
                    rate_limit_events += rate_limits
                    request_times.append(elapsed)
            except Exception:
                async with lock:
                    failed_requests += 1
            finally:
                queue.task_done()

    started = time.perf_counter()
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers)
    elapsed = time.perf_counter() - started

    chars = [len(message) for message in messages[:completed_messages or len(messages)]]
    mean_chars = statistics.mean(chars) if chars else 0.0
    median = statistics.median(request_times) if request_times else None
    p95 = None
    if len(request_times) >= 2:
        p95 = statistics.quantiles(request_times, n=20)[18]

    return BenchResult(
        concurrency=concurrency,
        batch_size=batch_size,
        requested_messages=len(messages),
        completed_messages=completed_messages,
        completed_requests=completed_requests,
        failed_requests=failed_requests,
        rate_limit_events=rate_limit_events,
        retry_events=retry_events,
        elapsed_seconds=elapsed,
        messages_per_second=completed_messages / elapsed if elapsed > 0 else 0.0,
        requests_per_minute=(completed_requests / elapsed * 60) if elapsed > 0 else 0.0,
        median_request_seconds=median,
        p95_request_seconds=p95,
        mean_chars_per_message=mean_chars,
        estimated_tokens_per_second=(completed_messages * mean_chars / 4 / elapsed) if elapsed > 0 else 0.0,
    )


async def main_async(args: argparse.Namespace) -> None:
    api_key = _load_api_key(args.key_pid)
    messages = _sample_messages(
        Path(args.db),
        sample_size=args.sample_size,
        min_len=args.min_len,
        max_len=args.max_len,
        seed=args.seed,
    )
    if not messages:
        raise RuntimeError("no messages sampled")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[BenchResult] = []
    for concurrency in args.concurrency:
        run_messages = messages[: args.per_level_messages]
        result = await _run_one(
            api_key=api_key,
            model=args.model,
            messages=run_messages,
            batch_size=args.batch_size,
            concurrency=concurrency,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout_seconds,
            dimensions=args.dimensions,
        )
        results.append(result)
        print(json.dumps(asdict(result), sort_keys=True), flush=True)
        if result.rate_limit_events and args.stop_after_rate_limit:
            break
        await asyncio.sleep(args.cooldown_seconds)

    payload = {
        "model": args.model,
        "sample_size": len(messages),
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "results": [asdict(result) for result in results],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark direct OpenAI embedding throughput.")
    parser.add_argument("--db", default="/workspace/wild-delusion-miner/data/dedupe.sqlite")
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--output", default="/workspace/wild-delusion-miner/data/benchmarks/openai_embeddings.json")
    parser.add_argument("--key-pid", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=12000)
    parser.add_argument("--per-level-messages", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--cooldown-seconds", type=float, default=5)
    parser.add_argument("--dimensions", type=int, default=None)
    parser.add_argument("--stop-after-rate-limit", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
