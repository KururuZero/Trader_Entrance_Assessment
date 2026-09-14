"""
check_times.py

Two things:
  1. ms_to_us(ms) — convert a Unix-ms timestamp to US Eastern datetime
  2. Walk every cleaned file and produce checking.csv with:
        file_path, first_row_us, last_row_us

Useful for verifying that each split file actually falls inside its intended
US-session window.
"""

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ET = ZoneInfo("America/New_York")

OPEN_DIR   = Path("clean_open_data")
CLOSE_DIR  = Path("clean_close_data")
OUT_CSV    = "checking.csv"


# ---------- timestamp helper ----------

def ms_to_us(ms: int) -> datetime:
    """Unix milliseconds -> US Eastern datetime (DST-aware)."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(ET)


def ms_to_us_str(ms: int) -> str:
    """Same as ms_to_us but returns a clean string."""
    return ms_to_us(ms).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]   # trim to ms


# ---------- file walker ----------

def ts_col_for(venue: str) -> str:
    return "transact_time_ms" if venue == "binance" else "participant_ts_ms"


def first_last_us(path: Path, ts_col: str):
    """Return (first_us_str, last_us_str) for a cleaned CSV, or (None, None)."""
    try:
        df = pd.read_csv(path, usecols=[ts_col])
    except Exception as e:
        return None, None
    if df.empty:
        return None, None
    ts = df[ts_col].dropna()
    if ts.empty:
        return None, None
    return ms_to_us_str(int(ts.iloc[0])), ms_to_us_str(int(ts.iloc[-1]))


def walk(source_dir: Path, source_label: str, rows: list):
    if not source_dir.exists():
        print(f"[skip] {source_dir} missing")
        return
    for venue_dir in sorted(source_dir.iterdir()):
        if not venue_dir.is_dir():
            continue
        venue = venue_dir.name
        ts_col = ts_col_for(venue)

        for coin_dir in sorted(venue_dir.iterdir()):
            if not coin_dir.is_dir():
                continue
            for f in sorted(coin_dir.glob("*.csv")):
                first_us, last_us = first_last_us(f, ts_col)
                rows.append({
                    "file":         str(f),
                    "source":       source_label,
                    "exchange":     venue,
                    "coin":         coin_dir.name,
                    "first_row_us": first_us,
                    "last_row_us":  last_us,
                })


# ---------- main ----------

def main():
    rows = []
    walk(OPEN_DIR,  "open",  rows)
    walk(CLOSE_DIR, "close", rows)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}  ({len(df)} files)")

    # show all rows
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print(df[["source", "exchange", "coin", "first_row_us", "last_row_us"]])


if __name__ == "__main__":
    main()