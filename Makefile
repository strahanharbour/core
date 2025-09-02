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
