"""Mock vLLM backend for local development without GPUs.

Implements just enough of an OpenAI-compatible surface for kvRouteRS to proxy:

    GET  /health
    GET  /v1/models
    POST /v1/chat/completions   (streaming, non-streaming, plus prefill-only
                                 and decode-only phases for disaggregated mode)

Timing model
------------
* TTFT is proportional to the prompt token count (simulating prefill), plus a
  fixed dispatch delay for decode-only phases.
* Decode tokens arrive at a fixed inter-token delay.
* An in-process LRU of prompt BLOCK-prefix hashes stands in for the KV cache.
  Blocks are 512 characters, mirroring router-core's chain-hash layout. The
  fraction of cached blocks cuts the prefill cost proportionally, up to 80% on
  a full hit. This makes cache-aware routing measurable with zero GPUs.

Usage: python mock_vllm.py [port] [prefix_cache_capacity_blocks]
Environment: MOCK_PORT, MOCK_PREFIX_CAP
"""

import asyncio
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import uuid
from collections import OrderedDict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


PORT = int(sys.argv[1]) if len(sys.argv) > 1 else _env_int("MOCK_PORT", 8001)
PREFIX_CACHE_CAP = (
    int(sys.argv[2]) if len(sys.argv) > 2 else _env_int("MOCK_PREFIX_CAP", 2048)
)

BASE_TTFT_SECONDS = 0.05            # fixed per-request overhead
PREFILL_SECONDS_PER_TOKEN = 0.005   # simulated prefill cost per prompt token
INTER_TOKEN_SECONDS = 0.03          # simulated decode step
DECODE_DISPATCH_SECONDS = 0.005     # fixed delay before the first decode token
PREFIX_HIT_SPEEDUP = 0.8            # fraction of prefill cost avoided on a full hit
DEFAULT_MAX_TOKENS = 16
MAX_TOKENS_CAP = 256
# Mirrors router-core's default prompt_block_chars.
PREFIX_BLOCK_CHARS = 512
MODEL_ID = "mock-vllm"

VOCAB = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "router", "cache", "prefix", "token", "latency", "worker", "model",
    "serving", "stream", "decode", "prefill", "memory", "inference",
    "fast", "efficient", "system", "response", "generation", "batch",
]

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

app = FastAPI(title="mock-vllm")

# LRU of prompt BLOCK-prefix hashes this process has "cached" (our KV-cache
# stand-in). Entries are 512-character blocks, matching the router's chain hash
# layout; the capacity is in blocks so residency behaves like an engine with a
# block-based KV cache.
prefix_cache = OrderedDict()


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def canonical_prompt(messages):
    """Mirror of router-core's canonical_prompt: one `role: content` line per
    message, joined by newlines. The block layout below depends on this."""
    return "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}"
        for m in messages
        if isinstance(m.get("content"), str)
    )


def block_prefix_hashes(text):
    """SHA-256 chain over every block of the prompt (last partial block
    included, matching router-core's chain_hash block count)."""
    hasher = hashlib.sha256()
    hashes = []
    for i in range(0, len(text), PREFIX_BLOCK_CHARS):
        hasher.update(text[i:i + PREFIX_BLOCK_CHARS].encode("utf-8"))
        hashes.append(hasher.digest())
    return hashes


def record_and_match(hashes):
    """Insert all block hashes into the LRU; return the longest matching
    prefix length in blocks."""
    matched = 0
    for i in range(len(hashes) - 1, -1, -1):
        if hashes[i] in prefix_cache:
            matched = i + 1
            break
    for h in hashes:
        prefix_cache.pop(h, None)
        prefix_cache[h] = True
    while len(prefix_cache) > PREFIX_CACHE_CAP:
        prefix_cache.popitem(last=False)
    return matched


def prompt_tokens(messages):
    tokens = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            tokens.extend(tokenize(content))
    return tokens


def generate_words(seed, count):
    rng = random.Random(seed)
    return [rng.choice(VOCAB) for _ in range(count)]


def render(words):
    if not words:
        return ""
    return " ".join(words) + "."


def completion_chunk(chat_id, model, created, delta, finish_reason):
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "mock"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    phase = request.headers.get("x-router-phase", "full")
    body = await request.json()
    model = body.get("model", MODEL_ID)
    messages = body.get("messages") or []
    max_tokens = min(int(body.get("max_tokens") or DEFAULT_MAX_TOKENS), MAX_TOKENS_CAP)
    stream = bool(body.get("stream", False))

    prompt_text = canonical_prompt(messages)
    block_hashes = block_prefix_hashes(prompt_text)
    total_blocks = len(block_hashes) or 1
    matched_blocks = record_and_match(block_hashes)
    hit_fraction = matched_blocks / total_blocks

    tokens = prompt_tokens(messages)
    prompt_count = len(tokens)
    prefill_seconds = BASE_TTFT_SECONDS + PREFILL_SECONDS_PER_TOKEN * prompt_count
    ttft_seconds = BASE_TTFT_SECONDS + PREFILL_SECONDS_PER_TOKEN * prompt_count * (
        1 - PREFIX_HIT_SPEEDUP * hit_fraction
    )

    if phase == "prefill":
        # Prefill-only phase (disaggregated mode): sleep through the simulated
        # prefill, then return KV metadata without generating tokens. The
        # router uses this to "materialize" the KV cache before transferring
        # it to a decode worker.
        await asyncio.sleep(ttft_seconds)
        return JSONResponse(
            {
                "object": "prefill.result",
                "prompt_blocks": total_blocks,
                "matched_blocks": matched_blocks,
                "usage": {"prompt_tokens": prompt_count, "cached_blocks": matched_blocks},
            },
            headers={
                "x-kv-prefix-hit": str(matched_blocks),
                "x-ttft-ms": f"{ttft_seconds * 1000:.2f}",
                "x-completion-tokens": "0",
            },
        )

    if phase == "decode":
        # Decode-only phase: the KV cache is assumed present (the router just
        # transferred it), so the prefill cost is skipped and only a fixed
        # dispatch delay precedes the first token.
        ttft_seconds = DECODE_DISPATCH_SECONDS

    seed = int(hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(), 16)
    words = generate_words(seed, max_tokens)
    content = render(words)

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    headers = {
        "x-kv-prefix-hit": str(matched_blocks),
        "x-ttft-ms": f"{ttft_seconds * 1000:.2f}",
        "x-completion-tokens": str(len(words)),
    }

    if stream:
        async def event_stream():
            await asyncio.sleep(ttft_seconds)
            yield f"data: {json.dumps(completion_chunk(chat_id, model, created, {'role': 'assistant'}, None))}\n\n"
            pieces = [w + " " for w in words[:-1]] + ([words[-1] + "."] if words else [])
            for piece in pieces:
                await asyncio.sleep(INTER_TOKEN_SECONDS)
                yield f"data: {json.dumps(completion_chunk(chat_id, model, created, {'content': piece}, None))}\n\n"
            yield f"data: {json.dumps(completion_chunk(chat_id, model, created, {}, 'stop'))}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(), media_type="text/event-stream", headers=headers
        )

    await asyncio.sleep(ttft_seconds + INTER_TOKEN_SECONDS * len(words))
    return JSONResponse(
        {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_count,
                "completion_tokens": len(words),
                "total_tokens": prompt_count + len(words),
                "prompt_tokens_details": {"cached_blocks": matched_blocks},
            },
        },
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
