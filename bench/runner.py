"""Async trace replay against a router instance, with metrics scraping.

The runner replays a trace at a Poisson arrival rate, measures per-request
TTFT (first streamed chunk) and end-to-end latency, samples the router's
`/metrics` during the run (for per-worker in-flight load), and scrapes
cumulative counters before and after so deltas (cache-hit blocks, prompt
blocks, tokens, requests) are exact per run.
"""

import asyncio
import math
import random
import re
import time

import httpx

METRIC_LINE = re.compile(r"^(\w+)(?:\{([^}]*)\})?\s+([\d.eE+\-]+|NaN)$")

MAX_TOKENS = 32
MODEL = "mock-vllm"


def parse_metrics(text):
    """Parse Prometheus text exposition into {(name, (label, value)...): float}."""
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        match = METRIC_LINE.match(line)
        if not match:
            continue
        name, labels_text, value = match.groups()
        labels = ()
        if labels_text:
            pairs = []
            for pair in labels_text.split(","):
                key, val = pair.split("=", 1)
                pairs.append((key.strip(), val.strip().strip('"')))
            labels = tuple(sorted(pairs))
        out[(name, labels)] = float(value)
    return out


def counter_sum(metrics, name):
    return sum(value for (metric_name, _), value in metrics.items() if metric_name == name)


def metric_by_label(metrics, name):
    """{(labels tuple) -> value} for one metric name."""
    return {
        labels: value
        for (metric_name, labels), value in metrics.items()
        if metric_name == name
    }


async def wait_ready(base_url, timeout=60.0):
    async with httpx.AsyncClient(timeout=10.0) as client:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{base_url}/ready")
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
    return False


def percentile(values, p):
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return ordered[index]


async def run_trace(base_url, trace, rate, seed, sampler_interval=0.2):
    """Replay `trace` against `base_url` at a Poisson arrival rate.

    Returns a dict with per-request samples, percentiles, throughput, hit
    rates, and the sampled per-worker load imbalance.
    """
    assert await wait_ready(base_url), f"router at {base_url} never became ready"

    async with httpx.AsyncClient(timeout=180.0) as client:
        before_text = (await client.get(f"{base_url}/metrics")).text
        before = parse_metrics(before_text)

        inflight_samples = {}
        sampler_done = asyncio.Event()

        async def sampler():
            while not sampler_done.is_set():
                try:
                    text = (await client.get(f"{base_url}/metrics")).text
                    for labels, value in metric_by_label(parse_metrics(text), "kvrouter_inflight_requests").items():
                        worker = dict(labels).get("worker", "?")
                        inflight_samples.setdefault(worker, []).append(value)
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(sampler_interval)

        sampler_task = asyncio.create_task(sampler())

        per_request = [None] * len(trace)

        rng = random.Random(seed)
        arrivals = [0.0]
        for _ in range(1, len(trace)):
            arrivals.append(arrivals[-1] + rng.expovariate(rate))

        async def issue(index, item, due):
            # `due` is an absolute monotonic timestamp; sleep the remaining gap.
            await asyncio.sleep(max(0.0, due - time.monotonic()))
            body = {
                "model": MODEL,
                "messages": item["messages"],
                "max_tokens": MAX_TOKENS,
                "stream": True,
            }
            record = {
                "index": index,
                "ttft": float("nan"),
                "e2e": float("nan"),
                "tokens": 0,
                "matched": -1,
                "engine_hit": -1,
                "status": None,
            }
            try:
                async with client.stream(
                    "POST", f"{base_url}/v1/chat/completions", json=body
                ) as response:
                    headers = response.headers
                    record["matched"] = int(headers.get("x-router-matched-prefix", -1))
                    record["engine_hit"] = int(headers.get("x-kv-prefix-hit", -1))
                    ttft = None
                    tokens = 0
                    async for line in response.aiter_lines():
                        if line.startswith("data:") and ttft is None:
                            ttft = time.monotonic() - due
                        if '"content"' in line:
                            tokens += 1
                    end = time.monotonic()
                    record["ttft"] = ttft if ttft is not None else end - due
                    record["e2e"] = end - due
                    record["tokens"] = tokens
                    record["status"] = response.status_code
            except httpx.HTTPError as error:
                record["status"] = f"error: {type(error).__name__}"
            # Results are written by request index so per-request records stay
            # aligned with the trace regardless of completion order.
            per_request[index] = record

        start_time = time.monotonic()
        tasks = [
            asyncio.create_task(issue(i, item, start_time + arrivals[i]))
            for i, item in enumerate(trace)
        ]
        await asyncio.gather(*tasks)
        wall = time.monotonic() - start_time

        sampler_done.set()
        await sampler_task

        after_text = (await client.get(f"{base_url}/metrics")).text
        after = parse_metrics(after_text)

    records = [record for record in per_request if record is not None]
    ok = sum(1 for record in records if record["status"] == 200)
    ttfts = [record["ttft"] for record in records]
    e2es = [record["e2e"] for record in records]
    tokens_per_request = [record["tokens"] for record in records]
    statuses = [record["status"] for record in records]
    hit_blocks = counter_sum(after, "kvrouter_cache_hit_blocks") - counter_sum(
        before, "kvrouter_cache_hit_blocks"
    )
    prompt_blocks = counter_sum(after, "kvrouter_prompt_blocks_total") - counter_sum(
        before, "kvrouter_prompt_blocks_total"
    )
    tokens_total = counter_sum(after, "kvrouter_tokens_generated_total") - counter_sum(
        before, "kvrouter_tokens_generated_total"
    )
    requests_total = counter_sum(after, "kvrouter_requests_total") - counter_sum(
        before, "kvrouter_requests_total"
    )

    # Per-worker mean in-flight, sampled during the run.
    worker_means = {
        worker: sum(samples) / len(samples)
        for worker, samples in inflight_samples.items()
        if samples
    }
    if worker_means:
        mean_of_means = sum(worker_means.values()) / len(worker_means)
        imbalance = (
            max(worker_means.values()) / mean_of_means if mean_of_means > 1e-9 else 1.0
        )
    else:
        imbalance = 1.0

    engine_hit_fraction = (
        sum(
            record["engine_hit"] / trace[record["index"]]["blocks"]
            for record in records
            if record["engine_hit"] >= 0 and trace[record["index"]]["blocks"] > 0
        )
        / max(1, len(trace))
    )

    matched_blocks = [record["matched"] for record in records]

    return {
        "base_url": base_url,
        "requests": len(trace),
        "ok": ok,
        "statuses": {str(status): statuses.count(status) for status in set(statuses)},
        "wall_seconds": wall,
        "req_per_s": len(trace) / wall if wall > 0 else 0.0,
        "tokens_total": tokens_total,
        "tokens_per_s": tokens_total / wall if wall > 0 else 0.0,
        "ttft": {
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
            "p99": percentile(ttfts, 0.99),
            "mean": sum(t for t in ttfts if t == t) / max(1, ok),
        },
        "e2e": {
            "p50": percentile(e2es, 0.50),
            "p95": percentile(e2es, 0.95),
            "mean": sum(t for t in e2es if t == t) / max(1, ok),
        },
        "hit_rate_router": hit_blocks / prompt_blocks if prompt_blocks else 0.0,
        "hit_rate_engine": engine_hit_fraction,
        "imbalance": imbalance,
        "worker_inflight_means": worker_means,
        "requests_total_delta": requests_total,
        "ttft_samples": ttfts,
        "e2e_samples": e2es,
        "matched_blocks": matched_blocks,
        "tokens_per_request": tokens_per_request,
    }
