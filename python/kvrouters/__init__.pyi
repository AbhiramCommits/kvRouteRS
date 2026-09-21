"""Type stubs for the kvrouters native bindings."""

__version__: str


class KvRouterError(Exception):
    """Base class for all errors raised by the routing core."""

    def __init__(self, *args: object) -> None: ...


class NoHealthyWorkersError(KvRouterError):
    """No healthy worker (or none in the requested pool) was available."""


class PolicyError(KvRouterError):
    """The requested routing policy is unknown or unsupported here."""


class PrefixIndex:
    """Router-side belief about which workers hold which prompt blocks.

    Prompts are split into ``block_size``-character blocks and chain-hashed
    (mirroring vLLM's block-based prefix caching, tokenizer-free).
    """

    def __init__(
        self,
        block_size: int = 512,
        ttl_seconds: float = 300.0,
        max_entries: int = 100_000,
    ) -> None: ...

    def insert(self, worker_id: int, prompt: str) -> None:
        """Record that ``worker_id`` holds every block of ``prompt``."""

    def lookup(self, prompt: str) -> list[tuple[int, int]]:
        """For each known worker: ``(worker_id, matched_blocks)``, sorted by id."""

    def evict_stale(self) -> tuple[int, int]:
        """Apply TTL expiry and the LRU cap; returns ``(expired, lru_evicted)``."""

    def __len__(self) -> int: ...


class RouteDecision:
    """The outcome of one routing decision."""

    @property
    def worker(self) -> str: ...

    @property
    def matched_blocks(self) -> int: ...

    @property
    def prompt_blocks(self) -> int: ...

    @property
    def score(self) -> float: ...


class Router:
    """In-process router over a fixed worker set (all treated as healthy)."""

    def __init__(
        self,
        workers: list[str],
        cache_weight: float = 1.0,
        load_weight: float = 0.5,
        block_size: int = 512,
    ) -> None: ...

    def select_worker(self, prompt: str, policy: str) -> str:
        """Select a worker URL for ``prompt`` (``round_robin`` or ``cache_aware``)."""

    def route(self, prompt: str, policy: str) -> RouteDecision:
        """Full routing decision: worker, matched blocks, prompt blocks, score."""
