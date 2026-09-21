VENV := mocks/.venv
MOCK_PY := $(VENV)/bin/python

.PHONY: fmt fmt-check lint test install-mocks run-mocks run-router

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

lint:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace

install-mocks:
	python3 -m venv $(VENV)
	$(MOCK_PY) -m pip install --quiet --upgrade pip
	$(MOCK_PY) -m pip install --quiet -r mocks/requirements.txt

run-mocks: install-mocks
	$(MOCK_PY) mocks/mock_vllm.py 8001 & $(MOCK_PY) mocks/mock_vllm.py 8002 & wait

run-router:
	cargo run -p router-server
