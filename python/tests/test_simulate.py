"""Simulation tests: replay traces through the in-process Rust router and
check the predicted hit-rate behaves as routing theory says it should."""

from kvrouters_sdk import simulate


def test_simulate_predicts_hits_for_shared_prefixes():
    prompts = ["long shared system prompt block " * 40 + f" query {i}" for i in range(20)]
    trace = prompts * 5  # each prefix revisited 5 times

    cache_aware = simulate(trace, workers=["w1", "w2", "w3"], policy="cache_aware")
    assert cache_aware["predicted_hit_rate"] > 0.5
    assert cache_aware["requests"] == len(trace)
    assert cache_aware["prompt_blocks"] > 0
    assert len(cache_aware["decisions"]) == len(trace)

    round_robin = simulate(trace, workers=["w1", "w2", "w3"], policy="round_robin")
    # Cache-aware routing must predict more hits than round-robin for this
    # workload (round-robin revisits rotate across the 3 workers).
    assert cache_aware["predicted_hit_rate"] > round_robin["predicted_hit_rate"]


def test_simulate_random_prompts_predict_no_win():
    prompts = [f"random prompt with no shared prefix {i} " + "y" * 100 for i in range(60)]

    cache_aware = simulate(prompts, workers=["w1", "w2", "w3"], policy="cache_aware")
    round_robin = simulate(prompts, workers=["w1", "w2", "w3"], policy="round_robin")
    assert round_robin["predicted_hit_rate"] == 0.0
    # Honest result: nothing to cache means no hit-rate win for cache_aware.
    assert cache_aware["predicted_hit_rate"] == 0.0


def test_simulate_is_seed_reproducible():
    prompts = ["prefix " * 50 + f" tail {i}" for i in range(30)]
    first = simulate(prompts, workers=["w1", "w2"], policy="cache_aware", seed=42)
    second = simulate(prompts, workers=["w1", "w2"], policy="cache_aware", seed=42)
    assert first["predicted_hit_rate"] == second["predicted_hit_rate"]
    assert [d.worker for d in first["decisions"]] == [d.worker for d in second["decisions"]]
