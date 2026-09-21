VENV := mocks/.venv
MOCK_PY := $(VENV)/bin/python
PY3 := $(shell command -v python3.11 || command -v python3.13 || command -v python3.12 || command -v python3)
BENCH_PY := bench/.venv/bin/python
PYENV_PY := python/.venv/bin/python

.PHONY: fmt fmt-check lint test install-mocks run-mocks run-router bench-install bench up down python-install python-test

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

bench-install:
	$(PY3) -m venv bench/.venv
	$(BENCH_PY) -m pip install --quiet -r bench/requirements.txt

bench: bench-install
	$(BENCH_PY) bench/ab.py --reset

up:
	docker compose up -d --build

down:
	docker compose down

python-install:
	$(PY3) -m venv python/.venv
	$(PYENV_PY) -m pip install --quiet --upgrade pip
	$(PYENV_PY) -m pip install --quiet maturin pytest

python-test: python-install
	$(PYENV_PY) -m maturin develop --manifest-path crates/router-py/Cargo.toml
	$(PYENV_PY) -m pytest python/tests -q
