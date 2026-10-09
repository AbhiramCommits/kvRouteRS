"""GIL-release test: while one thread is inside a long index call, other
Python threads must keep running. The bindings release the GIL around the
hashing and shard-lock work; if they held it, no Python bytecode could run
anywhere until the call returned."""

import threading
import time

from kvrouters import PrefixIndex


def test_gil_released_during_lookup():
    index = PrefixIndex(block_size=256)
    # Long enough that a single lookup spends tens of milliseconds hashing
    # blocks in Rust, even in a release build.
    prompt = "shared prefix " + "x" * 4_000_000
    index.insert(0, prompt)

    ticks = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.001)

    thread = threading.Thread(target=ticker)
    thread.start()
    try:
        time.sleep(0.05)  # let the ticker get going
        start = time.perf_counter()
        matches = index.lookup(prompt)
        end = time.perf_counter()
    finally:
        stop.set()
        thread.join()

    assert matches and matches[0][1] > 0
    # Counting ticks inside a single call (rather than comparing multi-thread
    # wall times) keeps the check independent of the runner's core count.
    during = sum(1 for t in ticks if start < t < end)
    assert during >= 5, (
        f"GIL appears to be held during index work: the ticker thread ran "
        f"{during} times during a {1000 * (end - start):.0f} ms lookup"
    )
