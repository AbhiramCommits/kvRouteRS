"""Async OpenAI-compatible HTTP client for the kvRouteRS router server."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx


def iter_chunks(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Parse an SSE line stream into OpenAI chunk dicts (stops at `[DONE]`)."""
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return
        if payload:
            yield json.loads(payload)


class KvRouterClient:
    """Async client for a kvRouteRS router's OpenAI-compatible API.

    Thin by design: no retries, no caching, just the router endpoints.
    """

    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        timeout: float = 180.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "KvRouterClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _body(
        self,
        messages: list[dict[str, str]],
        stream: bool,
        model: str | None,
        max_tokens: int | None,
        temperature: float | None,
        **extra: Any,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model or self.model or "default",
            "messages": messages,
            "stream": stream,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        body.update(extra)
        return body

    async def chat(
        self,
        messages: list[dict[str, str]],
        stream: bool = True,
        model: str | None = None,
        max_tokens: int | None = 32,
        temperature: float | None = None,
        **extra: Any,
    ):
        """POST /v1/chat/completions.

        With ``stream=True`` (default) returns an async iterator that yields
        OpenAI-style chunk dicts; with ``stream=False`` returns the full
        completion JSON.
        """
        body = self._body(messages, stream, model, max_tokens, temperature, **extra)
        if stream:
            return self._stream_chunks(body)
        response = await self._client.post(
            f"{self.base_url}/v1/chat/completions", json=body
        )
        response.raise_for_status()
        return response.json()

    async def _stream_chunks(self, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        async with self._client.stream(
            "POST", f"{self.base_url}/v1/chat/completions", json=body
        ) as response:
            response.raise_for_status()
            for chunk in iter_chunks(response.aiter_lines()):
                yield chunk

    async def chat_text(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = 32,
        **extra: Any,
    ) -> str:
        """Convenience: full completion as a single string (non-streaming)."""
        result = await self.chat(messages, stream=False, model=model, max_tokens=max_tokens, **extra)
        return result["choices"][0]["message"]["content"]

    async def models(self) -> list[dict[str, Any]]:
        response = await self._client.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()["data"]

    async def health(self) -> bool:
        response = await self._client.get(f"{self.base_url}/health")
        return response.status_code == 200

    async def ready(self) -> bool:
        response = await self._client.get(f"{self.base_url}/ready")
        return response.status_code == 200
