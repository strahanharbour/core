# Core Research Scaffold

End-to-end research scaffold with ingestion, features, simple backtester, reporting, OMS demo, and utilities.

## Quickstart

macOS/Linux

```bash
bash scripts/dev_env.sh && make all
```

Windows (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dev_env.ps1
scripts\make.ps1
```

This sets up `.venv`, installs dependencies, runs data ingest → features → backtest → report, and writes outputs to `src/main/artifacts/`.

## What gets created

- Data: `src/main/artifacts/local_data/{SYMBOL}.parquet`
- Features: `src/main/artifacts/features/{SYMBOL}_features.parquet`
- Backtest trades: `src/main/artifacts/results/trades.parquet`
- Equity chart: `src/main/artifacts/results/equity.html`

## Makefile targets (macOS/Linux)

- `make venv` — create `.venv`, install `requirements.txt`
- `make data` — run `python -m data.ingest_daily_local`
- `make features` — run `python -m data.build_features`
- `make backtest` — run `python -m research.backtest`
- `make report` — run `python -m research.report`
- `make test` — run pytest
- `make all` — data → features → backtest → report

Make uses `PYTHONPATH=$(PWD)/src/main` and `.venv/bin/python`.

## Windows runner (PowerShell)

```powershell
# Bootstraps venv if needed, sets PYTHONPATH, and runs phases
scripts\make.ps1 -Target venv
scripts\make.ps1 -Target data
scripts\make.ps1 -Target features
scripts\make.ps1 -Target backtest
scripts\make.ps1 -Target report
scripts\make.ps1 -Target test
scripts\make.ps1 -Target all  # runs full pipeline
```

## Environment (.env)

Copy `.env.example` → `.env` and fill values as needed.

- `NEWS_API_KEY` — optional; enables NewsAPI in sentiment selector. If unset, selector falls back to a safe default.
- `DB_URL` — optional; SQLite path (e.g., `sqlite:///portfolio.db`).

Other optional envs:

- `DETERMINISTIC_EXEC=1` — make trade execution deterministic in tests/demos.
- `TEST_SEED` — seed for tests (default 42).

## Running modules directly

Make sure your shell has `PYTHONPATH=./src/main` (Linux/macOS) or run via the provided scripts.

```bash
python -m data.ingest_daily_local
python -m data.build_features
python -m research.backtest
python -m research.report
```

Streamlit UI:

```bash
streamlit run src/main/ui/app.py
```

## Notes

- Universe, paths, and parameters live in `src/main/config.yaml`.
- Loader/ETL pipelines target SQLite via `DB_URL` by default.
- KPIs/validation/regime/labels utilities are provided for research; the MVP backtester is intentionally simple (daily, long-only).

## Final Smoke Checks (Acceptance)

Run the end-to-end flow locally and verify expected outputs.

macOS/Linux

```bash
bash scripts/dev_env.sh && make all

# Or run step-by-step (ensure PYTHONPATH is set by Make or in your shell):
export PYTHONPATH="$PWD/src/main"
.venv/bin/python -m data.ingest_daily_local     # writes Parquets to src/main/artifacts/local_data
.venv/bin/python -m data.build_features         # writes *_features.parquet to src/main/artifacts/features
.venv/bin/python -m research.backtest           # writes trades.parquet to src/main/artifacts/results
.venv/bin/python -m research.report             # writes equity.html to src/main/artifacts/results
.venv/bin/python -m pytest -q                   # tests green, deterministic
```

Windows (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dev_env.ps1

# Recommended runner (sets PYTHONPATH automatically):
scripts\make.ps1 -Target all

# Or run step-by-step in the same session:
$env:PYTHONPATH = "$PWD\src\main"
\.venv\Scripts\python.exe -m data.ingest_daily_local   # Parquets in src\main\artifacts\local_data
\.venv\Scripts\python.exe -m data.build_features       # Features in src\main\artifacts\features
\.venv\Scripts\python.exe -m research.backtest         # trades.parquet in src\main\artifacts\results
\.venv\Scripts\python.exe -m research.report           # equity.html in src\main\artifacts\results
\.venv\Scripts\python.exe -m pytest -q                 # tests green
```

Optional

```bash
# SQLite ETL + OMS demo
python src/main/data/pipelines/etl_pipeline.py
```

Expected outputs

- `src/main/artifacts/local_data/{SYMBOL}.parquet`
- `src/main/artifacts/features/{SYMBOL}_features.parquet`
- `src/main/artifacts/results/trades.parquet`
- `src/main/artifacts/results/equity.html`

### Rank strategies
```powershell
$env:PYTHONPATH = (Resolve-Path .\src\main).Path
\.venv\Scripts\python.exe -m research.strategy_sweeps
\.venv\Scripts\python.exe -m research.rank_strategies


Outputs:

src/main/artifacts/results/param_sweep.csv

src/main/artifacts/results/ranked_strategies.csv

src/main/artifacts/results/rank_report.md
```
