"""Mock vLLM backend for local development without GPUs.

Implements just enough of an OpenAI-compatible surface for kvRouteRS to proxy:

    GET  /health
    GET  /v1/models
    POST /v1/chat/completions   (streaming and non-streaming)

Timing model
------------
* Time-to-first-token is proportional to the prompt token count, simulating
  prefill: ``TTFT = base + prefill_cost * prompt_tokens``.
* Decode tokens then arrive at a fixed inter-token delay.
* An in-process LRU of prompt-prefix hashes stands in for the KV cache: when a
  prefix of the incoming prompt has been seen before, TTFT is cut by 80%.
  This makes cache-aware routing measurable with zero GPUs.

Usage: python mock_vllm.py [port]
"""

import asyncio
import hashlib
import json
import random
import re
import sys
import time
import uuid
from collections import OrderedDict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8001

BASE_TTFT_SECONDS = 0.05            # fixed per-request overhead
PREFILL_SECONDS_PER_TOKEN = 0.005   # simulated prefill cost per prompt token
INTER_TOKEN_SECONDS = 0.03          # simulated decode step
PREFIX_HIT_SPEEDUP = 0.2            # TTFT multiplier on a prefix hit (-80%)
DEFAULT_MAX_TOKENS = 16
MAX_TOKENS_CAP = 256
PREFIX_CACHE_CAP = 2048
MODEL_ID = "mock-vllm"

VOCAB = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "router", "cache", "prefix", "token", "latency", "worker", "model",
    "serving", "stream", "decode", "prefill", "memory", "inference",
    "fast", "efficient", "system", "response", "generation", "batch",
]

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

app = FastAPI(title="mock-vllm")

# LRU of prompt-prefix hashes this process has "cached" (our KV-cache stand-in).
prefix_cache = OrderedDict()


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def prefix_hashes(tokens):
    """SHA-256 digest of every non-empty prefix of the token list."""
    hasher = hashlib.sha256()
    hashes = []
    for token in tokens:
        hasher.update(token.encode("utf-8"))
        hasher.update(b"\x00")
        hashes.append(hasher.digest())
    return hashes


def record_and_match(hashes):
    """Insert all prefix hashes into the LRU; return the longest matched prefix."""
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
    body = await request.json()
    model = body.get("model", MODEL_ID)
    messages = body.get("messages") or []
    max_tokens = min(int(body.get("max_tokens") or DEFAULT_MAX_TOKENS), MAX_TOKENS_CAP)
    stream = bool(body.get("stream", False))

    tokens = prompt_tokens(messages)
    prompt_count = len(tokens)
    matched = record_and_match(prefix_hashes(tokens))

    prefill_seconds = BASE_TTFT_SECONDS + PREFILL_SECONDS_PER_TOKEN * prompt_count
    ttft_seconds = prefill_seconds * PREFIX_HIT_SPEEDUP if matched > 0 else prefill_seconds

    seed = int(hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(), 16)
    words = generate_words(seed, max_tokens)
    content = render(words)

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    headers = {"x-kv-prefix-hit": str(matched), "x-ttft-ms": f"{ttft_seconds * 1000:.2f}"}

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
                "prompt_tokens_details": {"cached_tokens": matched},
            },
        },
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
