# Trader_Entrance_Assessment

This repository contains the code required to reproduce the cross-venue lead-lag VWAP analysis. It covers:

- Coin universe selection
- Raw trade download from Binance and Massive
- Data cleaning and US-session splitting
- Data-quality checks
- VWAP construction over centred windows
- Return construction
- Cross-exchange lead-lag index assignment
- Regression modelling and backtesting statistics
- Generation of the reported A1–A8 study tables

The pipeline is modular. Scripts `P01`–`P05` prepare and validate the data. Scripts `P06`–`P10` build the VWAP features, returns, alignment indices, regressions and final analysis outputs.

---

## 1. Setup

### 1.1 Clone the repository

```bash
git clone [https://github.com/KururuZero/Trader_Entrance_Assessment]
cd Trader_Entrance_Assessment
```

### 1.2 Python environment

Python 3.10+ is recommended because the code uses `zoneinfo` and modern type annotations.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
pandas
numpy
scipy
statsmodels
requests
massive
python-dotenv
pytest
```

`statsmodels` is used by the earlier data-preparation scripts `P01` and `P05`.  
`scipy` is optional but recommended for p-values and Wilcoxon tests in `P09` and `P10`.

### 1.3 Project root

The analysis scripts `P06`–`P10` use a configurable project root. Set it to the repository root:

```bash
export LEADLAG_ROOT="$PWD"
```

You can also pass `--root /path/to/project` to `P06`–`P10`.

Scripts `P01`–`P05` use relative paths such as `data/trades`, `clean_data`, etc. Run them from the repository root so those paths resolve correctly.

### 1.4 Secure API credential handling

Do **not** hardcode API keys in the repository.  
Create a `.env` file in the project root, or export environment variables directly.

Example `.env.example`:

```text
MASSIVE_API_KEY=your_massive_api_key_here
LEADLAG_ROOT=/absolute/path/to/leadlag-vwap
```

Example `trader_api.py` that reads from the environment:

```python
import os

API = os.environ["MASSIVE_API_KEY"]
```

Make sure `.gitignore` excludes secrets and large data artefacts:

```text
.env
data/
clean_data/
clean_open_data/
clean_close_data/
linear_vwap_construction/
linear_return/
regression_index/
regression_result/
regression_result_vwap/
*.csv
*.meta.json
```

If you use a different secret manager, adapt `trader_api.py` to read from it. The only requirement is that `from trader_api import API` returns a valid Massive API key at runtime.

---

## 2. Repository layout

Typical repository structure:

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── trader_api.py                  # local, not committed with real keys
├── P01_coin_selection.py
├── P02_trade_download.py
├── P03_clean_and_split.py
├── P04_check_time.py
├── P05_summary.py
├── P06_linear_VWAP.py
├── P07_linear_return.py
├── P08_regression_assign.py
├── P09_regression_run.py
├── P10_vwap_pipeline.py
├── data/
│   └── trades/
│       ├── binance/
│       └── massive/
├── clean_data/
├── clean_open_data/
├── clean_close_data/
├── linear_vwap_construction/
├── linear_return/
├── regression_index/
├── regression_result/
├── regression_result_vwap/
└── coins.csv, ols_result.csv, waiting.csv, nv.csv, checking.csv
```

Raw and derived data are not committed to the public repository because of size and exchange terms. Run the download and processing scripts to recreate them.

---

## 3. Pipeline overview

| Script | Purpose |
|---|---|
| `P01_coin_selection.py` | Fetch 7-day volume from Massive, intersect with active Binance spot symbols, run OLS on notional volume, and write `coins.csv` and `ols_result.csv`. |
| `P02_trade_download.py` | Download daily trade files from Binance bulk mirror and Massive REST API into `data/trades/{exchange}/{coin}/{date}.csv`. Progress tracked in `progress.csv`. |
| `P03_clean_and_split.py` | Verify raw files exist, clean trades, split into `clean_data`, `clean_open_data` and `clean_close_data`. Splits Massive files by exchange ID into `Exchange{id}/`. |
| `P04_check_time.py` | Walk cleaned open/close files and write `checking.csv` with first/last US Eastern timestamps for data-quality verification. |
| `P05_summary.py` | Compute inter-trade wait statistics into `waiting.csv`, build notional-volume table `nv.csv`, and regress notional volume on mean wait time. |
| `P06_linear_VWAP.py` | Build centred VWAP features over intervals `[200, 500, 1000, 5000, 10000, 30000]` ms. Adds `possible`, `5_trades`, `side`, `next`, `vwap`, `imb` columns. Writes `linear_vwap_construction/`. |
| `P07_linear_return.py` | Build VWAP-to-VWAP returns, log returns, per-dt returns, excess dt and dt columns. Writes `linear_return/`. |
| `P08_regression_assign.py` | For each ordered exchange pair, find nearest lag trade in `[t + max(2a, b-c), t + b + c]`. Writes index mapping CSVs under `regression_index/`. |
| `P09_regression_run.py` | Run cross-venue OLS regressions of lag return on lead return. Writes `regression_result/{data_root}/regressions.csv` and `summary.csv`. |
| `P10_vwap_pipeline.py` | Final analysis layer: non-overlapping subsample regressions, tradability statistics, imbalance regression, and A1–A8 study tables under `regression_result_vwap/analysis/`. |

---

## 4. Running the main analysis

Run all commands from the repository root.

### 4.1 Prepare credentials and root

```bash
export LEADLAG_ROOT="$PWD"
# Ensure MASSIVE_API_KEY is set, or that trader_api.py can read it.
```

### 4.2 Coin selection and data download

```bash
python P01_coin_selection.py
python P02_trade_download.py
```

`P02_trade_download.py` has two flags at the top:

```python
DOWNLOAD_BINANCE = True
DOWNLOAD_MASSIVE = True
```

Set either to `False` to skip that venue.

### 4.3 Clean, split and validate

```bash
python P03_clean_and_split.py
python P04_check_time.py
python P05_summary.py
```

`P03` will abort if any expected raw file is missing.  
`P04` writes `checking.csv`.  
`P05` writes `waiting.csv` and `nv.csv`.

### 4.4 VWAP construction and returns

```bash
python P06_linear_VWAP.py --root "$LEADLAG_ROOT"
python P07_linear_return.py --root "$LEADLAG_ROOT"
```

These steps are chunked and cache-aware. They can be re-run safely.  
To rebuild everything from scratch, delete the relevant output directory and rerun.

### 4.5 Index assignment and regressions

```bash
python P08_regression_assign.py --root "$LEADLAG_ROOT" --pairs all
python P09_regression_run.py --root "$LEADLAG_ROOT" --pairs all
```

For a faster run restricted to the primary Binance/Massive directions:

```bash
python P08_regression_assign.py --root "$LEADLAG_ROOT" --pairs primary
python P09_regression_run.py --root "$LEADLAG_ROOT" --pairs primary
```

### 4.6 Final analysis and reported tables

```bash
python P10_vwap_pipeline.py --root "$LEADLAG_ROOT" --stage all
```

Optional: compute a notional-volume proxy from raw trades for A3:

```bash
python P10_vwap_pipeline.py --root "$LEADLAG_ROOT" --stage volume
```

Then run the analysis stage:

```bash
python P10_vwap_pipeline.py --root "$LEADLAG_ROOT" --stage analyze
```

`P10` writes the final A1–A8 tables and `summary.md` under:

```text
regression_result_vwap/analysis/
```

---

## 5. Reported outputs

The main reported outputs are:

```text
regression_result/{data_root}/regressions.csv
regression_result/{data_root}/summary.csv

regression_result_vwap/{data_root}/regressions.csv

regression_result_vwap/analysis/A1_spec_summary.csv
regression_result_vwap/analysis/A2_tier_by_spec.csv
regression_result_vwap/analysis/A2_tier_pooled.csv
regression_result_vwap/analysis/A3_tier_rank_tests.csv
regression_result_vwap/analysis/A4_open_vs_close.csv
regression_result_vwap/analysis/A4_open_vs_close_by_tier.csv
regression_result_vwap/analysis/A5_persistence_over_b.csv
regression_result_vwap/analysis/A6_direction_by_coin.csv
regression_result_vwap/analysis/A6_direction_tests.csv
regression_result_vwap/analysis/A7_tradability.csv
regression_result_vwap/analysis/A7_tradability_by_tier.csv
regression_result_vwap/analysis/A8_coverage.csv
regression_result_vwap/analysis/summary.md
```

The headline results are summarised in `summary.md`.

---

## 6. Data-quality checks and tests

The pipeline includes several embedded checks:

- **Presence check** — `P03_clean_and_split.py` verifies all expected raw trade files exist before cleaning.
- **Timestamp verification** — `P04_check_time.py` produces `checking.csv`, showing first and last US Eastern timestamps per cleaned split file.
- **Wait-time statistics** — `P05_summary.py` writes `waiting.csv` with trade-gap statistics per coin, exchange and state.
- **Sortedness and row-count guards** — `P06_linear_VWAP.py` checks that source timestamps are sorted and that light-pass and write-pass row counts match.
- **Atomic writes** — all major writers use temp-file-plus-rename, so a killed process cannot leave a partial destination that is later mistaken for a valid cache.
- **Cache stamps** — `P06`, `P07`, `P08` and `P09` write sidecar `.meta.json` files recording source mtime, size, row count and pipeline version. Stale caches are rebuilt automatically.
- **Stale-result quarantine** — `P09` and `P10` rename a `regressions.csv` written under an older `pipeline_version` rather than silently topping it up.
- **Minimum observation and trade gates** — `P07`, `P09` and `P10` enforce `MIN_OBS` and the `5_trades` window-validity gate.
- **Non-overlap thinning** — `P10` reports both full-sample and thinned non-overlapping regressions, because overlapping VWAP windows overstate significance.

To run the optional test suite:

```bash
pytest
```

If no `tests/` directory is present, the checks above are the primary validation path.

---

## 7. Reproducibility notes

- Set `LEADLAG_ROOT` to the repository root before running `P06`–`P10`.
- Run `P01`–`P05` from the repository root so relative paths match.
- The pipeline is cache-aware. Re-running a script skips work that is already valid.
- Cache versions are bumped when formulas change. For example, `P07` changed the return reference from the raw trade print to the window VWAP, and `P09`/`P10` quarantine old results accordingly.
- Raw data are not included. Use `P02_trade_download.py` to recreate them.
- Exchange mapping and Massive substitution rules are centralised in `P08_regression_assign.py` and imported by `P09` and `P10`.
- All outputs are CSV unless otherwise noted.

---

## 8. Security

- Never commit API keys, `.env` files or secrets.
- Use environment variables or a local secret manager.
- `trader_api.py` is intentionally not committed with a real key.
- Large data directories are ignored by Git to keep the repository small and compliant with exchange data terms.

---

## 9. License and contact

Add your license and contact details here before submission.

Example:

```text
MIT License
Maintainer: your-name
Contact: your-email@example.com
```
