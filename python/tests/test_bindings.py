"""Tests for the kvrouters native bindings (prefix round-trip, router
selection, exception mapping)."""

import time

import pytest

import kvrouters
from kvrouters import (
    KvRouterError,
    NoHealthyWorkersError,
    PolicyError,
    PrefixIndex,
    Router,
)


def test_version_exposed():
    assert isinstance(kvrouters.__version__, str)
    assert kvrouters.__version__.count(".") >= 1


def test_prefix_round_trip():
    index = PrefixIndex()
    prompt = "the quick brown fox jumps over the lazy dog " * 50
    index.insert(7, prompt)

    rows = index.lookup(prompt)
    assert any(worker == 7 and matched > 0 for worker, matched in rows)
    assert all(matched == 0 for _, matched in index.lookup("completely different prompt"))
    assert len(index) > 0

    before = len(index)
    index.insert(3, "another prompt entirely with some length to it")
    assert len(index) > before


def test_lookup_is_sorted_and_per_worker():
    index = PrefixIndex(block_size=64)
    shared = "s" * 300
    index.insert(2, shared)
    index.insert(5, shared + " tail for worker five")
    rows = index.lookup(shared)
    assert [worker for worker, _ in rows] == sorted(worker for worker, _ in rows)
    assert any(worker == 2 and matched > 0 for worker, matched in rows)


def test_evict_stale():
    index = PrefixIndex(ttl_seconds=0.0, max_entries=100)
    index.insert(1, "temporary prompt that will expire")
    time.sleep(0.02)
    expired, lru_evicted = index.evict_stale()
    assert expired >= 1
    assert lru_evicted == 0
    assert len(index) == 0


def test_len_tracks_entries():
    index = PrefixIndex(block_size=16)
    assert len(index) == 0
    index.insert(0, "a" * 40)  # 3 blocks
    assert len(index) == 3


def test_router_cache_aware_is_sticky():
    router = Router(["http://a:8001", "http://b:8002"])
    prompt = "shared system context " * 60
    first = router.select_worker(prompt, "cache_aware")
    second = router.select_worker(prompt, "cache_aware")
    assert first == second

    decision = router.route(prompt, "cache_aware")
    assert decision.worker == first
    assert decision.matched_blocks > 0
    assert decision.prompt_blocks > decision.matched_blocks or decision.prompt_blocks >= 1
    assert decision.score >= 0.0


def test_router_round_robin_alternates():
    router = Router(["http://a:8001", "http://b:8002"])
    picks = [router.select_worker(f"prompt {i}", "round_robin") for i in range(4)]
    assert picks == [
        "http://a:8001",
        "http://b:8002",
        "http://a:8001",
        "http://b:8002",
    ]


def test_router_raises_policy_error():
    router = Router(["http://a:8001"])
    with pytest.raises(PolicyError):
        router.select_worker("hi", "bogus_policy")
    with pytest.raises(PolicyError):
        router.select_worker("hi", "disaggregated")


def test_router_empty_worker_set_raises():
    with pytest.raises((KvRouterError, NoHealthyWorkersError)):
        Router([])


def test_router_without_workers_raises_no_healthy():
    # A Router built with workers is fine; selecting with no healthy workers is
    # not reachable in-process (workers are marked healthy), but constructing
    # with zero workers fails construction.
    with pytest.raises(KvRouterError):
        Router([], cache_weight=1.0)


def test_prefix_index_bad_parameters():
    with pytest.raises(ValueError):
        PrefixIndex(block_size=0)
    with pytest.raises(ValueError):
        PrefixIndex(ttl_seconds=-1.0)


def test_router_bad_parameters():
    with pytest.raises(ValueError):
        Router(["http://a:1"], block_size=0)
