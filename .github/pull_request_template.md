## Summary

What and why, in a paragraph.

## Design

- Which seam does this touch (routing core, server, bindings, discovery,
  metrics)? Why here?
- For routing-behavior changes: the expected effect on hit rate / TTFT and
  why it is not a regression.

## Test evidence

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] Python changes: `make python-test`
- [ ] Routing-behavior changes: `make bench` (paste the relevant table rows
      below, or CI will)
- [ ] Docs updated where behavior changed (`docs/ARCHITECTURE.md`,
      `README.md`)

## Notes for reviewers

Anything subtle: lock ordering, GIL discipline, staleness assumptions,
benchmark sensitivity.
