# Trader Entrance Assessment

Cross-exchange lead-lag analysis using centred VWAP windows, VWAP-to-VWAP returns, and non-overlapping regression subsamples.

Repository: https://github.com/KururuZero/Trader_Entrance_Assessment/

---

## Overview

This repository contains a reproducible research pipeline for studying lead-lag effects between crypto venues, primarily Binance and Massive (with Massive optionally split into constituent exchanges, which are: Coinbase, Bitfinex, Bitstamp and Kraken.

The pipeline:

1. Selects a common coin universe across Binance and Massive.
2. Downloads tick-level trade data.
3. Cleans and splits data into US-market-open, US-market-close and full-day datasets.
4. Computes centred VWAP windows, taker-side classification and taker notional imbalance.
5. Computes VWAP-to-VWAP forward returns for multiple horizons.
6. Builds cross-exchange lead-lag index mappings.
7. Runs lead-lag OLS regressions.
8. Produces analysis tables A1-A8 and a headline `summary.md`.
9. (Optional) Runs a focused follow-up investigation on the one Exchange23 lag result that survives, producing tables F0-F8 and `focus_summary.md`.

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
| `P11_focus_exchange23.py` | Focused follow-up on the Exchange23 lag effect: free-`b` sweep, per-coin, hour-of-day, placebos, staleness, continuation, cost curve and OOS. Writes F0-F8 and `focus_summary.md`. |
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
regression_result_vwap/focus/
```

`regression_result_vwap/focus/` contains the P11 focus investigation outputs (F0-F8 tables and `focus_summary.md`).

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

`P06`-`P11` use a shared project root. Set it once:

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

### 11. Focus investigation on the Exchange23 lag effect

```bash
python P11_focus_exchange23.py --root "$LEADLAG_ROOT"
```

This is optional and only meaningful after P06/P07 have been run (it reads the
VWAP return arrays directly, bypassing P08's pre-built index files, so it can
trace `b` on a free grid). It drills into the single cluster at the top of
P10's A1 table — `<lead venue> -> Exchange23` at `a=1000, b=5000` — and tries
to falsify it.

Stage the run:

```bash
python P11_focus_exchange23.py --root "$LEADLAG_ROOT" --stage sweep
python P11_focus_exchange23.py --root "$LEADLAG_ROOT" --stage hours,placebo,cost
python P11_focus_exchange23.py --root "$LEADLAG_ROOT" --stage all --boot 2000
```

Stages: `sweep`, `hours`, `placebo`, `stale`, `horizon`, `cost`, `oos`
(default `all`). `--boot` sets the block-bootstrap resample count (default
1000).

Main outputs:

- `regression_result_vwap/focus/F0_daily.csv`
- `regression_result_vwap/focus/F1_spec.csv`
- `regression_result_vwap/focus/F2_by_coin.csv`
- `regression_result_vwap/focus/F3_hours.csv`
- `regression_result_vwap/focus/F4_placebo.csv`
- `regression_result_vwap/focus/F5_staleness.csv`
- `regression_result_vwap/focus/F6_horizon.csv`
- `regression_result_vwap/focus/F7_cost_curve.csv`
- `regression_result_vwap/focus/F8_oos.csv`
- `regression_result_vwap/focus/focus_summary.md`

---

## Research, modelling and backtesting code

The modelling and backtesting logic lives mainly in:

- `P08_regression_assign.py` — constructs the lead-lag alignment.
- `P09_regression_run.py` — full-sample OLS regressions.
- `P10_vwap_pipeline.py` — non-overlapping subsample regressions, tradability, cost scenarios and A1-A8 tables.
- `P11_focus_exchange23.py` — focused falsification battery around the Exchange23 lag cluster, producing F0-F8 and `focus_summary.md`.

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

### Focus investigation (P11)

P10's A1 table ranks 1,440 specs by BH q-value, and the top of that table is
not the proposal's primary hypothesis. It is a cluster of specs that all share
the same lag venue (`Exchange23`) at `a=1000, b=5000`, with the reverse
directions close to zero. P11 is built to try to kill that finding. It reuses
P06/P07's VWAP arrays but rebuilds the lead->lag matching itself (same window
rule as P08) so `b` is a free parameter rather than the four values P08 wrote
index files for.

What P11 does that P10 did not:

1. **Free `b`.** Traces the decay curve at `b = 1500 … 20000` and locates the half-life. `a` is still restricted to P06's grid.
2. **Per-coin results** for the Exchange23 pairs (A6 only covered binance/massive).
3. **Hour-of-day buckets** within `clean_data`, not just the US open/close day-split of A4.
4. **Four placebos**: reverse direction, negative `b`, wrong-day lead (same clock time), wrong-coin lead. All four should be flat if the effect is real.
5. **Lag-side staleness diagnostics.** If Exchange23 prints late, "prediction" is just a stale quote catching up. This is the most likely way the result dies, and it is tested explicitly (F5, `corr_dense` vs `corr_all`).
6. **Continuation vs reversal.** The same signal evaluated at `2b` and `4b`. Information transfer continues; transient impact reverses.
7. **Threshold × cost surface**, date-block bootstrap CI on gross bps, and a first-half / second-half OOS split.

Stage outputs (under `regression_result_vwap/focus/`):

| File | Contents |
|---|---|
| `F0_daily.csv` | one row per (coin, date, lead, lag, a, b, variant) |
| `F1_spec.csv` | per-spec aggregate + BH q over the (small) focus grid |
| `F2_by_coin.csv` | per-coin aggregate at the headline spec |
| `F3_hours.csv` | per hour-of-day bucket at the headline spec |
| `F4_placebo.csv` | reverse / negative-b / shuffled-day / shuffled-coin |
| `F5_staleness.csv` | lag-side dt ratios, lag idle time, conditional corr |
| `F6_horizon.csv` | signal evaluated at `b`, `2b`, `4b` |
| `F7_cost_curve.csv` | `|x|` threshold × cost scenario -> net bps and capacity |
| `F8_oos.csv` | first-half -> second-half out-of-sample |
| `focus_summary.md` | the numbers in the order the report needs them |

Headline spec: `a = 1000`, `b = 5000`, `c = b/10`. Focus pairs and controls
are declared in `FOCUS_PAIRS`; focus coins are `X_BTCUSD`, `X_ETHUSD`,
`X_XRPUSD` (the only coins for which Exchange23 has data). Coins are matched
across venues by the same `from_massive_name` rule used elsewhere in the
pipeline.

**Note on q-values.** `bh_q_focus` in `F1_spec.csv` is computed over the
~200 specs actually tested in P11, not P10's 1,440. A1's `bh_q` remains the
honest number for "did the original search find anything"; `bh_q_focus` only
ranks within the already-selected focus set and is not directly comparable.

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
python P11_focus_exchange23.py --root "$LEADLAG_ROOT" --stage all
```

The headline tables are in:

```text
regression_result_vwap/analysis/
```

The narrative summary is:

```text
regression_result_vwap/analysis/summary.md
```

The focus investigation outputs are in:

```text
regression_result_vwap/focus/
```

with the narrative in:

```text
regression_result_vwap/focus/focus_summary.md
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

### Focus investigation

- `P11_focus_exchange23.py` reuses the same non-overlap thinning (`THIN_MULT * a`) and `MIN_OBS` gate as P10.
- Rebuilds the lead->lag matching inline so that the negative-`b` placebo differs from the positive test *only* in the sign of `b`.
- Cross-source placebos align by time-of-day, not absolute timestamp.
- Staleness is reported as `corr_dense` vs `corr_all` so a stale-quote artefact is visible rather than averaged away.
- Bootstrap CIs resample whole dates, not individual windows, because within-day windows are not independent.
- Negative-`b` and wrong-day placebo window rules scale `c` with `|b|`, not the headline `b`, so the placebo is not accidentally a weaker real test.

---

## Caching and restartability

Most stages are restartable.

- `P02` tracks progress in `progress.csv`.
- `P06` and `P07` use `.meta.json` sidecars with cache versions.
- `P08` uses per-pair `.meta.json` files.
- `P09` and `P10` keep `done_keys` and append new results.
- `P11` reads the arrays produced by P06/P07 directly and can be run per stage
  (`--stage sweep`, `--stage hours,placebo,cost`, etc.). Re-running a single
  stage regenerates only that stage's table.
- Stale result files are renamed, not deleted.

Cache versions are deliberately bumped when the return definition, statistics or column schema changes. If a result looks stale, check the `pipeline_version` column and the `.meta.json` files.

---

## Notes

- Raw data is not committed to the repository. Run `P02_trade_download.py` to download it.
- API credentials must be supplied through environment variables or a local uncommitted `trader_api.py`.
- Date ranges and coin lists are configured in `P01`, `P02` and `P03`.
- For large files, the pipeline uses chunked reads and writes to avoid OOM kills.
- Run all commands from the repository root unless you explicitly pass `--root` to `P06`-`P11`.
- `P11` currently uses `clean_data` only (all hours) and re-derives the US open/close contrast internally in F3, rather than reading `clean_open_data` / `clean_close_data`. If finer `a` values are needed for the focus grid, a P06 rerun with a denser interval list is required.

---

## Repository

https://github.com/KururuZero/Trader_Entrance_Assessment/

## Assistance and source disclosure

### Academic papers consulted
- None
  
### External datasets used
- **Binance public bulk trade data** (`data.binance.vision`) – daily aggTrades files for Binance spot symbols.

### Existing repositories or code consulted
- None

### Tutorials or articles used
- None

### AI tools used
- **ChatGPT (GPT-5.6) and Claude Sonnet** were used extensively for:
  - Refinement of the hypothesis proposed by me and the methodology used to prove or disprove the hypothesis.
  - Follow my instructions to generate the python code that implement methodology.
  - Consulting about setup of AWS EC2 instance.
  - Refactoring and modularising the pipeline (P06–P11), including shared IO/config layers and cache metadata.
  - Writing and improving docstrings, comments, and the README.
  - Designing memory-efficient chunked processing to avoid OOM kills on large tick files.
  - Implementing statistical routines: closed-form OLS, Benjamini–Hochberg q-values, one-sample t-tests, date-block bootstrap confidence intervals, and signed-log transforms.
  - Designing the P11 falsification battery (free-`b` sweep, negative-`b` / shuffled-day / shuffled-coin placebos, lag-side staleness diagnostics, continuation-vs-reversal horizon test, threshold × cost surface, first-half / second-half OOS).
  - Debugging and correcting return calculations, window validity gates, and non-overlapping subsample logic.
  - Generating the assistance and source disclosure section you are reading now.

### Assistance received from another person
- None.
