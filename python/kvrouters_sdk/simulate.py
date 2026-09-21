"""In-process routing simulation for tuning weights without any backends."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

from kvrouters import RouteDecision, Router


def simulate(
    prompts: Sequence[str],
    workers: Sequence[str] | None = None,
    policy: str = "cache_aware",
    repeats: int = 1,
    cache_weight: float = 1.0,
    load_weight: float = 0.5,
    block_size: int = 512,
    seed: int | None = None,
) -> dict[str, Any]:
    """Replay a prompt trace through the in-process Rust `Router`.

    The Rust router records every dispatched prompt in its belief index, so
    the predicted hit rate here is exactly what the server-side router would
    predict for the same trace — no backends, no network. Useful for tuning
    `cache_weight`/`load_weight` (and block size) before deploying.

    Returns a dict with at least:
      - ``predicted_hit_rate``: matched blocks / prompt blocks over the trace
      - ``requests``, ``matched_blocks``, ``prompt_blocks``
      - ``per_worker``: Counter of decisions per worker URL
      - ``decisions``: list of `RouteDecision` objects
    """
    trace: list[str] = list(prompts) * repeats
    if seed is not None:
        shuffled = list(trace)
        random.Random(seed).shuffle(shuffled)
        trace = shuffled

    router = Router(
        workers=list(workers or ["worker-0", "worker-1", "worker-2"]),
        cache_weight=cache_weight,
        load_weight=load_weight,
        block_size=block_size,
    )

    decisions: list[RouteDecision] = [router.route(prompt, policy) for prompt in trace]
    matched_blocks = sum(decision.matched_blocks for decision in decisions)
    prompt_blocks = sum(decision.prompt_blocks for decision in decisions)
    return {
        "policy": policy,
        "predicted_hit_rate": matched_blocks / prompt_blocks if prompt_blocks else 0.0,
        "requests": len(trace),
        "matched_blocks": matched_blocks,
        "prompt_blocks": prompt_blocks,
        "per_worker": Counter(decision.worker for decision in decisions),
        "decisions": decisions,
    }
