SHELL := /usr/bin/env bash

ROOT := $(PWD)
VENV := $(ROOT)/.venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip

export PYTHONPATH := $(ROOT)/src/main

.PHONY: venv data features backtest report test all clean

venv:
	@if [ ! -x "$(PY)" ]; then \
		python3 -m venv "$(VENV)"; \
	fi
	@"$(PIP)" install -r requirements.txt
	@echo "Using Python: $$($(PY) --version)"
	@echo "PYTHONPATH=$(PYTHONPATH)"

data: venv
	@"$(PY)" -m data.ingest_daily_local

features: venv
	@"$(PY)" -m data.build_features

backtest: venv
	@"$(PY)" -m research.backtest

report: venv
	@"$(PY)" -m research.report

test: venv
	@"$(PY)" -m pytest -q

all: data features backtest report
	@echo "Pipeline complete. Results in src/main/artifacts/results"

clean:
	@rm -rf "$(VENV)"

# --- Additive targets for meta workflow and tooling ---
.PHONY: meta bt_meta lint format

meta: venv
	@"$(PY)" -m training.labels_make && "$(PY)" -m training.meta_train

bt_meta: venv
	@"$(PY)" -m research.backtest_meta

lint:
	@flake8 .
	@mypy --ignore-missing-imports .

format:
	@black .
	@isort .
