"""
summary05.py

1. Compute inter-trade wait statistics per (coin, exchange, state) -> waiting.csv
2. Build notional-volume table per coin-venue-period with caching -> nv.csv
3. Regress notional volume on mean wait time, per venue and state
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


# ---------- config ----------

CLEAN_OPEN_DIR  = Path("clean_open_data")
CLEAN_CLOSE_DIR = Path("clean_close_data")
CLEAN_DIR       = Path("clean_data")

WAITING_CSV     = "waiting.csv"
COINS_CSV       = "coins.csv"
NV_CSV          = "nv.csv"
OLS_RESULT_CSV  = "ols_result.csv"
NV_OLS_OUT      = "nv_regressions.csv"

# period used when computing binance notional from clean_data
NV_START = date(2026, 8, 30)
NV_END   = date(2026, 9, 5)

pd.set_option("display.max_rows",    None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width",       None)


# ---------- helpers ----------

def ts_col_for(exchange: str) -> str:
    return "transact_time_ms" if exchange == "binance" else "participant_ts_ms"


def size_col_for(exchange: str) -> str:
    return "quantity" if exchange == "binance" else "size"


def massive_to_binance(m_tick: str) -> str:
    """X:BTCUSD -> BTCUSDT. Assumes the universe only has USD-paired Massive tickers."""
    return m_tick.replace("X:", "").replace("USD", "USDT")


# ---------- waiting time ----------

def process_dir(source_dir: Path, state: str, results: list):
    if not source_dir.exists():
        print(f"[skip] {source_dir} does not exist")
        return

    for venue_dir in sorted(source_dir.iterdir()):
        if not venue_dir.is_dir():
            continue
        exchange = venue_dir.name
        ts_col   = ts_col_for(exchange)

        for coin_dir in sorted(venue_dir.iterdir()):
            if not coin_dir.is_dir():
                continue
            coin = coin_dir.name

            total_gap_ms = 0
            n_gaps       = 0
            n_files      = 0

            for f in sorted(coin_dir.glob("*.csv")):
                df = pd.read_csv(f)
                if ts_col not in df.columns:
                    print(f"[skip] {f}: missing {ts_col}")
                    continue
                ts = df[ts_col].to_numpy(dtype="int64")
                if len(ts) < 2:
                    continue
                ts = np.sort(ts)
                gaps = np.diff(ts)
                total_gap_ms += int(gaps.sum())
                n_gaps       += len(gaps)
                n_files      += 1

            if n_gaps > 0:
                results.append({
                    "coin":         coin,
                    "exchange":     exchange,
                    "state":        state,
                    "n_files":      n_files,
                    "n_trades":     n_gaps + n_files,
                    "n_gaps":       n_gaps,
                    "total_gap_ms": total_gap_ms,
                    "mean_wait_ms": total_gap_ms / n_gaps,
                })
            else:
                results.append({
                    "coin":         coin,
                    "exchange":     exchange,
                    "state":        state,
                    "n_files":      n_files,
                    "n_trades":     0,
                    "n_gaps":       0,
                    "total_gap_ms": 0,
                    "mean_wait_ms": np.nan,
                })


def compute_waits() -> pd.DataFrame:
    results = []
    process_dir(CLEAN_OPEN_DIR,  "open",  results)
    process_dir(CLEAN_CLOSE_DIR, "close", results)

    df = (pd.DataFrame(results)
            .sort_values(["exchange", "coin", "state"])
            .reset_index(drop=True))
    df.to_csv(WAITING_CSV, index=False)
    print(f"Saved -> {WAITING_CSV} ({len(df)} rows)")
    return df


# ---------- notional volume table ----------

NV_COLS = ["Tick Name", "Notional Volume", "Start Date", "End Date", "Venue", "Computed"]


def load_existing_nv() -> pd.DataFrame:
    if Path(NV_CSV).exists():
        return pd.read_csv(NV_CSV)
    return pd.DataFrame(columns=NV_COLS)


def has_cached(nv: pd.DataFrame, tick: str, start: date, end: date, venue: str) -> bool:
    if nv.empty:
        return False
    return (
        (nv["Tick Name"]   == tick) &
        (nv["Start Date"]  == str(start)) &
        (nv["End Date"]    == str(end)) &
        (nv["Venue"]       == venue)
    ).any()


def massive_notional(tick: str, start: date, end: date):
    """Read notional volume for one Massive ticker from ols_result.csv."""
    if not Path(OLS_RESULT_CSV).exists():
        return None
    ols = pd.read_csv(OLS_RESULT_CSV)
    row = ols[ols["Tick Name"] == tick]
    if row.empty:
        return None
    return float(row["Notional Volume"].iloc[0])


def binance_notional(tick: str, start: date, end: date) -> float:
    """Sum quantity*price across clean_data/binance/{tick}/{date}.csv in [start, end]."""
    total = 0.0
    d = start
    while d <= end:
        f = CLEAN_DIR / "binance" / tick / f"{d}.csv"
        if f.exists():
            df = pd.read_csv(f, usecols=["price", "quantity"])
            total += float((df["price"] * df["quantity"]).sum())
        d += timedelta(days=1)
    return total


def build_nv(coins_csv: str = COINS_CSV,
             start: date = NV_START,
             end:   date = NV_END) -> pd.DataFrame:
    """Build/extend nv.csv for the chosen coins. Skips (tick, start, end, venue)
    tuples that already exist."""
    coins = pd.read_csv(coins_csv)

    tick_col = next((c for c in ("Tick Name", "M Tick Name", "tick")
                     if c in coins.columns), coins.columns[0])

    nv = load_existing_nv()
    added = 0

    for m_tick in coins[tick_col].astype(str):
        b_tick = massive_to_binance(m_tick)

        # --- Massive: pull from ols_result.csv ---
        if not has_cached(nv, m_tick, start, end, "massive"):
            v = massive_notional(m_tick, start, end)
            if v is not None:
                nv.loc[len(nv)] = {
                    "Tick Name":       m_tick,
                    "Notional Volume": v,
                    "Start Date":      str(start),
                    "End Date":        str(end),
                    "Venue":           "massive",
                    "Computed":        False,
                }
                added += 1

        # --- Binance: compute from clean_data ---
        if not has_cached(nv, b_tick, start, end, "binance"):
            v = binance_notional(b_tick, start, end)
            nv.loc[len(nv)] = {
                "Tick Name":       b_tick,
                "Notional Volume": v,
                "Start Date":      str(start),
                "End Date":        str(end),
                "Venue":           "binance",
                "Computed":        True,
            }
            added += 1

    nv.to_csv(NV_CSV, index=False)
    print(f"Saved -> {NV_CSV} ({len(nv)} rows, {added} added)")
    return nv


# ---------- OLS: notional volume ~ mean wait ----------

def fit_ols(x: np.ndarray, y: np.ndarray):
    X = sm.add_constant(x)
    model = OLS(y, X).fit()
    return {
        "intercept": float(model.params[0]),
        "slope":     float(model.params[1]),
        "r2":        float(model.rsquared),
        "p_value":   float(model.pvalues[1]),
    }


def run_notional_wait_regressions(nv, waits):
    rows = []
    for venue in ("binance", "massive"):
        v_nv = nv[nv["Venue"] == venue][["Tick Name", "Notional Volume"]].copy()
        if venue == "massive":
            v_nv["Tick Name"] = v_nv["Tick Name"].str.replace(":", "_", regex=False)

        for state in ("open", "close"):
            v_wt = waits[(waits["exchange"] == venue) &
                         (waits["state"]    == state)][["coin", "mean_wait_ms"]]
            merged = v_nv.merge(v_wt, left_on="Tick Name",
                                       right_on="coin", how="inner")
            ...
            merged = merged.dropna(subset=["Notional Volume", "mean_wait_ms"])

            if len(merged) < 3:
                rows.append({
                    "venue": venue, "state": state, "n": len(merged),
                    "intercept": np.nan, "slope": np.nan,
                    "r2": np.nan, "p_value": np.nan,
                })
                continue

            stats = fit_ols(
                merged["mean_wait_ms"].to_numpy(dtype=float),
                merged["Notional Volume"].to_numpy(dtype=float),
            )
            rows.append({
                "venue": venue, "state": state, "n": len(merged),
                **stats,
            })

    out = pd.DataFrame(rows)
    out.to_csv(NV_OLS_OUT, index=False)
    print(f"\nSaved -> {NV_OLS_OUT}")
    return out


# ---------- main ----------

def main():
    # 1) Waiting time table
    waits = compute_waits()
    print("\n=== waiting.csv ===")
    print(waits)

    # 2) Notional volume table (cached)
    nv = build_nv()
    print("\n=== nv.csv ===")
    print(nv)

    # 3) Regressions
    print("\n=== Regressions: Notional Volume ~ mean_wait_ms ===")
    reg = run_notional_wait_regressions(nv, waits)
    print(reg)


if __name__ == "__main__":
    main()