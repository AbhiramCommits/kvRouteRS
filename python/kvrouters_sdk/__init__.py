"""kvrouters_sdk: the thin, idiomatic Python SDK over the kvrouters core.

Mirrors the Dynamo-style split: the Rust runtime (`kvrouters`) owns routing
and caching; this package is pure Python — an async HTTP client for the router
server and an in-process simulator for tuning routing weights without any
backends running.
"""

from .client import KvRouterClient, iter_chunks
from .simulate import simulate

__all__ = ["KvRouterClient", "iter_chunks", "simulate"]
