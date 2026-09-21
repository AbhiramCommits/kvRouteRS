#!/usr/bin/env python3
"""A/B benchmark driver for kvRouteRS.

Replays identical traces against `round_robin`, `cache_aware`, and
`disaggregated` router instances, then prints a comparison table (cache hit
rate, TTFT p50/p95/p99, e2e p50/p95, throughput, per-worker load imbalance)
and emits `bench/results/*.json` plus charts:

- `ttft_cdf.png`  — TTFT CDF per policy, one subplot per workload shape
- `hit_rate_vs_replicas.png` — hit rate vs. number of replicas (sweep)

The engine-side mocks and routers are expected to run via docker-compose; the
driver can restart them between phases to guarantee cold caches
(`--reset`). Honest numbers: the `random` shape has no shared prefixes, so
cache-aware routing is expected to show no win there.

Usage:
    python bench/ab.py [--requests N] [--rate R] [--seed S] [--reset]
                       [--skip-sweep] [--router-binary PATH]
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner import run_trace, wait_ready  # noqa: E402
from tracegen import TraceGenerator  # noqa: E402

POLICIES = (
    ("round_robin", os.environ.get("KVROUTER_RR", "http://127.0.0.1:18080")),
    ("cache_aware", os.environ.get("KVROUTER_CA", "http://127.0.0.1:18081")),
    ("disaggregated", os.environ.get("KVROUTER_DG", "http://127.0.0.1:18082")),
)

SHAPES = ("shared_prefix", "multi_turn", "random")

COMPOSE = ["docker", "compose", "-f", "docker-compose.yml"]
MOCK_SERVICES = ("mock1", "mock2", "mock3", "mock4")
ROUTER_SERVICES = ("router-round-robin", "router-cache-aware", "router-disaggregated")
# Host ports the compose file exposes the mocks on (configurable: the local
# machine may have other stacks on 8001-8004).
MOCK_PORTS = [
    int(port) for port in os.environ.get("KVROUTER_MOCK_PORTS", "8001,8002,8003,8004").split(",")
]


def docker_available():
    return shutil.which("docker") is not None


def restart_services(services):
    if not docker_available():
        print("WARN: docker not available; skipping service restart (warm caches may contaminate results)")
        return
    subprocess.run(
        COMPOSE + ["restart", *services],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def wait_mocks(mocks):
    for port in mocks:
        if not await wait_health(f"http://127.0.0.1:{port}", timeout=30.0):
            print(f"WARN: mock on port {port} did not come up")
    time.sleep(2.0)


async def wait_health(base_url, timeout=60.0):
    async with httpx.AsyncClient(timeout=10.0) as client:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
    return False


async def reset_stack():
    """Restart mocks (clears engine KV cache) and routers (clears router-side
    prefix index) so every benchmark phase starts from a cold state."""
    print("== resetting mocks and routers for cold-cache runs ==")
    restart_services((*MOCK_SERVICES, *ROUTER_SERVICES))
    await wait_mocks(MOCK_PORTS)
    for _, url in POLICIES:
        await wait_ready(url, timeout=60.0)


def build_traces(generator, args):
    return {
        "shared_prefix": generator.shared_prefix(
            requests=args.requests,
            prefixes=args.prefixes,
            prefix_chars=args.prefix_chars,
            suffix_chars=args.suffix_chars,
        ),
        "multi_turn": generator.multi_turn(
            requests=args.mt_requests, conversations=args.conversations, turn_chars=args.turn_chars
        ),
        "random": generator.random(requests=args.random_requests, prompt_chars=args.prompt_chars),
    }


async def run_ab(generator, traces, args):
    results = {}
    for shape in SHAPES:
        trace = traces[shape]
        for name, url in POLICIES:
            # Restart the engine mocks before every run: their KV-cache LRUs
            # persist across runs, and all three policies replay the SAME
            # trace — without a cold engine, an earlier policy's warm-up would
            # leak hits into the next policy's numbers.
            print(f"== running {shape} against {name} ({len(trace)} requests @ {args.rate}/s) ==")
            restart_services(MOCK_SERVICES)
            await wait_mocks(MOCK_PORTS)
            await wait_ready(url, timeout=60.0)
            seed = args.seed + hash(shape) % 1000
            results[(shape, name)] = await run_trace(url, trace, args.rate, seed)
    return results


def fmt_seconds(value):
    return "n/a" if value != value else f"{value:.3f}"


def render_table(results, shape):
    header = (
        "| policy | hit rate (router) | hit rate (engine) | ttft p50 | ttft p95 | ttft p99 | "
        "e2e p50 | e2e p95 | req/s | tok/s | imbalance |"
    )
    separator = "|" + "---|" * 10 + "---|"
    rows = [header, separator]
    for name, _ in POLICIES:
        r = results[(shape, name)]
        rows.append(
            "| {name} | {hr:.3f} | {he:.3f} | {t50} | {t95} | {t99} | {e50} | {e95} | "
            "{rps:.1f} | {tps:.1f} | {imb:.2f} |".format(
                name=name,
                hr=r["hit_rate_router"],
                he=r["hit_rate_engine"],
                t50=fmt_seconds(r["ttft"]["p50"]),
                t95=fmt_seconds(r["ttft"]["p95"]),
                t99=fmt_seconds(r["ttft"]["p99"]),
                e50=fmt_seconds(r["e2e"]["p50"]),
                e95=fmt_seconds(r["e2e"]["p95"]),
                rps=r["req_per_s"],
                tps=r["tokens_per_s"],
                imb=r["imbalance"],
            )
        )
    return "\n".join(rows)


def save_results(outdir, results, sweep):
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": {
            f"{shape}/{policy}": {key: value for key, value in run.items() if key not in ("ttft_samples", "e2e_samples")}
            for (shape, policy), run in results.items()
        },
        "sweep": sweep,
    }
    (outdir / "ab_results.json").write_text(json.dumps(payload, indent=2))
    table = "\n\n".join(
        f"## {shape}\n\n{render_table(results, shape)}" for shape in SHAPES
    )
    (outdir / "report.md").write_text(table + "\n")
    print(f"\nresults written to {outdir}/")


def render_charts(outdir, results, sweep):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # TTFT CDF per policy, one subplot per shape.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, shape in zip(axes, SHAPES):
        for name, _ in POLICIES:
            samples = [t for t in results[(shape, name)]["ttft_samples"] if t == t]
            if not samples:
                continue
            ordered = sorted(samples)
            cdf = [i / len(ordered) for i in range(1, len(ordered) + 1)]
            ax.plot(ordered, cdf, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("TTFT (s)")
        ax.set_ylabel("CDF")
        ax.set_title(shape)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("TTFT CDF per policy")
    fig.tight_layout()
    fig.savefig(outdir / "ttft_cdf.png", dpi=120)
    plt.close(fig)

    # Hit rate vs. number of replicas.
    if sweep:
        fig, ax = plt.subplots(figsize=(9, 5))
        replicas = sorted(sweep.keys())
        for metric in ("router", "engine"):
            for policy in ("round_robin", "cache_aware"):
                values = [sweep[n][policy][f"hit_rate_{metric}"] for n in replicas]
                ax.plot(replicas, values, marker="o", label=f"{policy} ({metric})")
        ax.set_xlabel("replicas")
        ax.set_ylabel("cache hit rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Cache hit rate vs. number of replicas (shared_prefix trace)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "hit_rate_vs_replicas.png", dpi=120)
        plt.close(fig)

    print(f"charts written to {outdir}/")


async def replica_sweep(args, generator, shared_trace):
    """Spawn local router processes with 1..N mock replicas and measure hit
    rate as a function of replica count, for round_robin and cache_aware."""
    binary = args.router_binary
    if not binary:
        for candidate in ("target/release/router-server", "target/debug/router-server"):
            if Path(candidate).exists():
                binary = candidate
                break
    if not binary:
        print("WARN: router binary not found; skipping replica sweep (use --router-binary)")
        return {}

    sweep = {}
    base_port = 19100
    for n in (1, 2, 3, 4):
        for policy in ("round_robin", "cache_aware"):
            workers = "\n".join(
                f"  - url: http://127.0.0.1:{MOCK_PORTS[i]}\n    pool: both"
                for i in range(n)
            )
            config = (
                f"routing_policy: {policy}\n"
                "health_check_interval_secs: 2\n"
                "prompt_block_chars: 512\n"
                "cache_weight: 1.0\n"
                "load_weight: 0.5\n"
                "prefix_index_ttl_secs: 300\n"
                "prefix_index_max_entries: 100000\n"
                "kv_transfer_cost_ms: 20\n"
                f"workers:\n{workers}"
            )
            config_path = Path(f"/tmp/kvrouter-sweep-{n}-{policy}.yaml")
            config_path.write_text(config)
            port = base_port + n * 10 + (0 if policy == "round_robin" else 1)
            env = dict(os.environ, ROUTER_CONFIG=str(config_path), ROUTER_BIND=f"127.0.0.1:{port}")
            print(f"== sweep: {policy} with {n} replica(s) on :{port} ==")
            process = subprocess.Popen([binary], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                restart_services(MOCK_SERVICES)
                await wait_mocks(MOCK_PORTS)
                if not await wait_ready(f"http://127.0.0.1:{port}", timeout=30.0):
                    print("WARN: sweep router never became ready; skipping point")
                    continue
                result = await run_trace(
                    f"http://127.0.0.1:{port}", shared_trace, args.rate, args.seed + n * 100
                )
                sweep.setdefault(n, {})[policy] = {
                    "hit_rate_router": result["hit_rate_router"],
                    "hit_rate_engine": result["hit_rate_engine"],
                }
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
    return sweep


async def main():
    parser = argparse.ArgumentParser(description="kvRouteRS A/B benchmark")
    parser.add_argument("--requests", type=int, default=472)
    # `prefixes` must not be divisible by the number of replicas (4): if it
    # were, round-robin would pin each prefix to the same replica on every
    # revisit and silently behave like cache-aware routing.
    parser.add_argument("--prefixes", type=int, default=59)
    parser.add_argument("--prefix-chars", type=int, default=1536)
    parser.add_argument("--suffix-chars", type=int, default=40)
    parser.add_argument("--mt-requests", type=int, default=248)
    # Same divisibility caveat applies to conversations.
    parser.add_argument("--conversations", type=int, default=31)
    parser.add_argument("--turn-chars", type=int, default=110)
    parser.add_argument("--random-requests", type=int, default=256)
    parser.add_argument("--prompt-chars", type=int, default=200)
    parser.add_argument("--rate", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true", help="restart compose stack for cold caches")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--router-binary", default=None)
    parser.add_argument("--outdir", default="bench/results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    generator = TraceGenerator(args.seed)
    traces = build_traces(generator, args)

    sweep = {}
    if not args.skip_sweep:
        sweep = await replica_sweep(args, generator, traces["shared_prefix"])
        await reset_stack()

    results = await run_ab(generator, traces, args)

    for shape in SHAPES:
        print()
        print(render_table(results, shape))

    save_results(outdir, results, sweep)
    render_charts(outdir, results, sweep)


if __name__ == "__main__":
    asyncio.run(main())
