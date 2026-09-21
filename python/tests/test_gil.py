"""GIL-release test: 8 threads hammering the index must overlap, not
serialize. With the GIL held, 8 threads would take ~8x the single-thread
time; the bindings release the GIL around the shard-lock work."""

import threading
import time

from kvrouters import PrefixIndex


def test_gil_released_under_threads():
    index = PrefixIndex(block_size=256)
    prompts = [f"shared prefix payload {i} " + "x" * 800 for i in range(64)]
    for i, prompt in enumerate(prompts):
        index.insert(i % 8, prompt)

    iterations = 20_000

    def hammer():
        for i in range(iterations):
            index.lookup(prompts[i % len(prompts)])

    start = time.perf_counter()
    hammer()
    single = time.perf_counter() - start

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - start

    # 8 threads doing the same total work must finish well under 8x the
    # single-thread wall time; with the GIL released they overlap (typically
    # ~1-2x on a multi-core machine). Bound generously to stay robust on CI.
    assert wall < 4.0 * single, (
        f"GIL appears to be held during index work: "
        f"8-thread wall {wall:.3f}s vs single-thread {single:.3f}s"
    )
