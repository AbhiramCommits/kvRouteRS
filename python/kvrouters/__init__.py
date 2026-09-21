"""kvrouters: KV-cache-aware inference routing, Rust core with a Python face.

The native extension lives in `kvrouters._native`; this module re-exports the
typed surface. The idiomatic SDK (async client + simulation) is in the sibling
package `kvrouters_sdk`.
"""

from kvrouters._native import (  # noqa: F401
    KvRouterError,
    NoHealthyWorkersError,
    PolicyError,
    PrefixIndex,
    RouteDecision,
    Router,
    __version__,
)

__all__ = [
    "KvRouterError",
    "NoHealthyWorkersError",
    "PolicyError",
    "PrefixIndex",
    "RouteDecision",
    "Router",
    "__version__",
]
