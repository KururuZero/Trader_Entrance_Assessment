# Trader Entrance Assessment

Cross-exchange lead-lag analysis using centred VWAP windows, VWAP-to-VWAP returns, and non-overlapping regression subsamples.

Repository: https://github.com/KururuZero/Trader_Entrance_Assessment/

---

## Overview

This repository contains a reproducible research pipeline for studying lead-lag effects between crypto venues, primarily Binance and Massive (with Massive optionally split into constituent exchanges such as Exchange1, Exchange2, Exchange6 and Exchange23).

The pipeline:

1. Selects a common coin universe across Binance and Massive.
2. Downloads tick-level trade data.
3. Cleans and splits data into US-market-open, US-market-close and full-day datasets.
4. Computes centred VWAP windows, taker-side classification and taker notional imbalance.
5. Computes VWAP-to-VWAP forward returns for multiple horizons.
6. Builds cross-exchange lead-lag index mappings.
7. Runs lead-lag OLS regressions.
8. Produces analysis tables A1-A8 and a headline `summary.md`.

The scripts are designed to be restartable, cache-aware and memory-safe for large tick files.

---

## Repository layout

| File | Purpose |
|---|---|
| `P01_coin_selection.py` | Builds the common Binance/Massive coin universe and writes `coins.csv`, `ols_result.csv`, `volume_0830_0905.csv`. |
| `P02_trade_download.py` | Downloads raw trade data from Binance bulk mirror and Massive REST API. |
| `P03_clean_and_split.py` | Cleans raw trades, sorts by timestamp, deduplicates, and creates `clean_data/`, `clean_open_data/`, `clean_close_data/`. Also splits Massive data by exchange ID. |
| `P04_check_time.py` | Writes `checking.csv` with first/last US Eastern timestamps per cleaned file. |
| `P05_summary.py` | Computes inter-trade wait statistics (`waiting.csv`) and notional volume table (`nv.csv`). |
| `P06_linear_VWAP.py` | Builds `linear_vwap_construction/` with centred VWAP windows, side flags, next-row indices and taker imbalance. |
| `P07_linear_return.py` | Builds `linear_return/` with VWAP-to-VWAP returns, log returns, per-dt returns and excess-dt columns. |
| `P08_regression_assign.py` | Builds `regression_index/` by mapping lead trades to lag trades within `(a, b, c)` windows. |
| `P09_regression_run.py` | Runs full-sample lead-lag OLS regressions and writes `regression_result/{data_root}/regressions.csv`. |
| `P10_vwap_pipeline.py` | Runs non-overlapping-subsample regressions, tradability statistics, imbalance regressions and A1-A8 analysis tables. |
| `trader_api.py` | Local-only credential file. Not committed. Must be created by the user. |

Primary output directories:

```text
clean_data/
clean_open_data/
clean_close_data/
linear_vwap_construction/
linear_return/
regression_index/
regression_result/
regression_result_vwap/
regression_result_vwap/analysis/
```

---

## Requirements

Recommended environment:

- Python 3.10+
- Linux/macOS shell
- Sufficient disk space for tick data
- A Massive API key for `P01` and `P02`

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy pandas scipy statsmodels requests massive
```

If you prefer a `requirements.txt`:

```text
numpy
pandas
scipy
statsmodels
requests
massive
```

`scipy` and `statsmodels` are used for statistics and OLS summaries. If `scipy` is unavailable, some p-value paths fall back to approximations, but installing it is recommended.

---

## Secure API credential handling

Do **not** commit API keys, secrets or `trader_api.py`.

Create a local `trader_api.py` in the repository root:

```python
# trader_api.py
import os

API = os.environ["MASSIVE_API_KEY"]
```

Then set the environment variable in your shell:

```bash
export MASSIVE_API_KEY="your_massive_api_key_here"
```

Add the following to `.gitignore`:

```gitignore
trader_api.py
.env
*.key
*.pem
```

The public Binance bulk mirror used in `P02` does not require an API key. The Massive REST API does.

---

## Configuration

Before running the full pipeline, review the date ranges and coin lists in:

- `P01_coin_selection.py`
- `P02_trade_download.py`
- `P03_clean_and_split.py`

Key variables:

```python
START = date(2026, 8, 30)
END   = date(2026, 9, 5)
COINS = [
    ("X:BTCUSD", "BTCUSDT"),
    ...
]
```

`P06`-`P10` use a shared project root. Set it once:

```bash
export LEADLAG_ROOT="$(pwd)"
```

Alternatively, pass `--root` to each script that supports it.

---

## Running the main analysis

Run all commands from the repository root.

### 1. Build the coin universe and reference tables

```bash
python P01_coin_selection.py
```

Outputs:

- `volume_0830_0905.csv`
- `ols_result.csv`
- `coins.csv`

### 2. Download raw trades

```bash
python P02_trade_download.py
```

Downloads to:

```text
data/trades/binance/{symbol}/{date}.csv
data/trades/massive/{ticker}/{date}.csv
```

Progress is tracked in `progress.csv`.

### 3. Clean and split trades

```bash
python P03_clean_and_split.py
```

Outputs:

- `clean_data/`
- `clean_open_data/`
- `clean_close_data/`

Also splits Massive files by exchange ID into `Exchange{id}/` folders.

### 4. Check timestamps

```bash
python P04_check_time.py
```

Outputs:

- `checking.csv`

### 5. Compute wait and notional-volume summaries

```bash
python P05_summary.py
```

Outputs:

- `waiting.csv`
- `nv.csv`
- `nv_regressions.csv`

### 6. Build centred VWAP construction

```bash
python P06_linear_VWAP.py --root "$LEADLAG_ROOT"
```

Outputs:

- `linear_vwap_construction/`

### 7. Build VWAP returns

```bash
python P07_linear_return.py --root "$LEADLAG_ROOT"
```

Outputs:

- `linear_return/`

### 8. Build lead-lag index mappings

```bash
python P08_regression_assign.py --root "$LEADLAG_ROOT" --pairs all
```

Outputs:

- `regression_index/`

For only the primary Binance/Massive directions:

```bash
python P08_regression_assign.py --root "$LEADLAG_ROOT" --pairs primary
```

### 9. Run full-sample regressions

```bash
python P09_regression_run.py --root "$LEADLAG_ROOT" --pairs all
```

Outputs:

- `regression_result/{data_root}/regressions.csv`
- `regression_result/{data_root}/summary.csv`

### 10. Run non-overlapping regressions and analysis tables

```bash
python P10_vwap_pipeline.py --root "$LEADLAG_ROOT" --stage all --pairs all
```

Optional: compute notional volume first.

```bash
python P10_vwap_pipeline.py --root "$LEADLAG_ROOT" --stage volume
```

Main outputs:

- `regression_result_vwap/{data_root}/regressions.csv`
- `regression_result_vwap/analysis/A1_spec_summary.csv`
- `regression_result_vwap/analysis/A2_tier_by_spec.csv`
- `regression_result_vwap/analysis/A3_tier_rank_tests.csv`
- `regression_result_vwap/analysis/A4_open_vs_close.csv`
- `regression_result_vwap/analysis/A5_persistence_over_b.csv`
- `regression_result_vwap/analysis/A6_direction_by_coin.csv`
- `regression_result_vwap/analysis/A7_tradability.csv`
- `regression_result_vwap/analysis/A8_coverage.csv`
- `regression_result_vwap/analysis/summary.md`

---

## Research, modelling and backtesting code

The modelling and backtesting logic lives mainly in:

- `P08_regression_assign.py` — constructs the lead-lag alignment.
- `P09_regression_run.py` — full-sample OLS regressions.
- `P10_vwap_pipeline.py` — non-overlapping subsample regressions, tradability, cost scenarios and A1-A8 tables.

The core regression is:

```text
lag_ret_{a}[lag_idx] ~ alpha + beta * lead_ret_{a}[lead_idx]
```

where `ret_{a}` is P07's VWAP-to-VWAP return.

P10 also computes:

- correlation
- hit rate
- signed mean return in bps
- gross P&L per signal
- net P&L under round-trip cost scenarios in `COST_BPS`
- taker imbalance regression
- coverage statistics

The non-overlapping thinning in P10 keeps the first valid lead row per `THIN_MULT * a` bucket, so kept windows are at least `a` apart. This reduces the overstated significance of overlapping windows.

---

## Code used to generate reported outputs

The reported outputs are generated by running the full pipeline in order:

```bash
export LEADLAG_ROOT="$(pwd)"

python P01_coin_selection.py
python P02_trade_download.py
python P03_clean_and_split.py
python P04_check_time.py
python P05_summary.py

python P06_linear_VWAP.py --root "$LEADLAG_ROOT"
python P07_linear_return.py --root "$LEADLAG_ROOT"
python P08_regression_assign.py --root "$LEADLAG_ROOT" --pairs all
python P09_regression_run.py --root "$LEADLAG_ROOT" --pairs all
python P10_vwap_pipeline.py --root "$LEADLAG_ROOT" --stage all --pairs all
```

The headline tables are in:

```text
regression_result_vwap/analysis/
```

The narrative summary is:

```text
regression_result_vwap/analysis/summary.md
```

---

## Data-quality checks and safeguards

The pipeline includes several runtime checks and safeguards:

### Presence and cleaning

- `P03_clean_and_split.py` checks that all expected raw files exist before cleaning.
- Drops rows with missing critical fields.
- Casts timestamps, prices and sizes to numeric types.
- Sorts by timestamp using stable mergesort.
- Deduplicates by trade ID.

### Timestamp verification

- `P04_check_time.py` writes `checking.csv` with first and last US Eastern timestamp per split file.
- Use this to verify that open/close splits fall inside their intended US-session windows.

### VWAP construction

- `P06_linear_VWAP.py` checks whether source files are time-sorted.
- Falls back to a full in-memory sort for unsorted files.
- Uses atomic writes: temporary file + rename.
- Writes sidecar `.meta.json` cache files with source mtime, size, row count and interval list.
- Validates row counts between the light pass and the write pass.

### Return construction

- `P07_linear_return.py` uses cache versioning via `.meta.json`.
- Applies the `5_trades` gate when `REQUIRE_5_TRADES = True`.
- Forces zero returns to zero in log/per-dt columns.
- Checks row counts between passes and refuses to write misaligned output.

### Lead-lag alignment

- `P08_regression_assign.py` validates per-combo row counts.
- Cache metadata records mtime and size of every constituent file on both sides.
- Per-combo outputs are written atomically.

### Regression results

- `P09_regression_run.py` quarantines stale `regressions.csv` files whose `pipeline_version` does not match.
- Uses closed-form OLS for simple regression to avoid design-matrix memory overhead.
- Appends new rows instead of rewriting the whole results file.
- Builds summary tables with a single chunked pass.

### Analysis

- `P10_vwap_pipeline.py` reports coverage in A8.
- Uses non-overlapping subsample statistics for main `corr`, `tstat`, `p`, `r2`.
- Reports full-sample `corr_all` and `tstat_all` separately.
- A7 reports gross and net bps under multiple cost assumptions.

---

## Caching and restartability

Most stages are restartable.

- `P02` tracks progress in `progress.csv`.
- `P06` and `P07` use `.meta.json` sidecars with cache versions.
- `P08` uses per-pair `.meta.json` files.
- `P09` and `P10` keep `done_keys` and append new results.
- Stale result files are renamed, not deleted.

Cache versions are deliberately bumped when the return definition, statistics or column schema changes. If a result looks stale, check the `pipeline_version` column and the `.meta.json` files.

---

## Notes

- Raw data is not committed to the repository. Run `P02_trade_download.py` to download it.
- API credentials must be supplied through environment variables or a local uncommitted `trader_api.py`.
- Date ranges and coin lists are configured in `P01`, `P02` and `P03`.
- For large files, the pipeline uses chunked reads and writes to avoid OOM kills.
- Run all commands from the repository root unless you explicitly pass `--root` to `P06`-`P10`.

---

## Repository

https://github.com/KururuZero/Trader_Entrance_Assessment/
