# Contributing to kvrouters

Thanks for your interest. This is a small project with an explicit quality
bar; this file tells you how to meet it.

## Development environment

You need Rust (stable), Python 3.11+, and Docker (for the integration paths).

```bash
cargo build --workspace           # compiles everything
make lint                         # cargo clippy --workspace --all-targets -- -D warnings
make test                         # cargo test --workspace
make python-test                  # maturin develop + pytest
```

The fast iteration loop for the router is Docker-free:

```bash
python3 -m venv mocks/.venv && mocks/.venv/bin/pip install -r mocks/requirements.txt
make run-mocks                    # two mock backends on 8001/8002
make run-router                   # router on 0.0.0.0:8080
```

## Conventions

- **No `unwrap()`/`expect()` outside tests.** Request paths return typed
  errors (`RouterError`, `ApiError`); panics are for invariants that are
  programmer bugs, and even those are avoided in server code.
- **Errors map to HTTP codes deliberately**: 400 malformed body, 502
  upstream failure, 503 no healthy worker, 501 unimplemented policy.
- **Public API is documented.** `router-core` denies missing docs; run
  `cargo doc --no-deps` and keep it warning-free.
- **Logs are structured** (JSON via `tracing`). Every proxied request logs
  `request_id`, worker, matched prefix blocks, policy, TTFT, and total
  latency.
- **Metrics** go through the `metrics` crate; new metrics get a
  `describe_*!` entry and a mention in `docs/ARCHITECTURE.md` if they change
  behavior.

## Tests

- Unit tests live next to the code (`#[cfg(test)] mod tests`). The
  concurrency and proptest tests are deliberately part of the default run.
- The vLLM e2e test is `#[ignore]`d and gated by `KVROUTERS_E2E=1`; it
  needs a GPU host.
- CI runs everything plus a kind-cluster smoke test and the A/B benchmark.
  The benchmark **fails if cache-aware routing regresses below the committed
  threshold** on the shared-prefix trace — if you change routing behavior,
  expect to update `bench/ab.py` defaults, the threshold in
  `.github/workflows/bench.yml`, or both, with justification in the PR.

## Pull requests

1. Open an issue or comment on an existing one first if the change is
   non-trivial.
2. Branch, implement, and include tests that would have caught the bug or
   prove the feature.
3. Run `make lint`, `make test`, and `cargo fmt --all -- --check` before
   pushing. If you touched Python: `make python-test`. If you touched
   routing behavior: run `make bench` locally.
4. The PR template asks for the design rationale and test evidence; fill it
   in.

## Commit style

Short imperative subject, body explaining *why* over *what*. Look at the git
history for the tone; each commit should leave the tree building and the
tests green.
