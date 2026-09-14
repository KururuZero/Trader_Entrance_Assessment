"""
clean_and_split.py

Pipeline:
  1. Verify all 210 raw files exist.
  2. Clean each: drop NaN, cast, sort by ts, dedupe by trade id.
       -> clean_data/{venue}/{coin}/{date}.csv
  3. US-market-open periods (13:30-20:00 UTC on US trading days).
       -> clean_open_data/{venue}/{coin}/{date}.csv
  4. US-market-closed periods, concatenated into CONTINUOUS blocks:
        - 20:01 of trading day D -> 13:29 of next trading day
        - Friday 20:01 -> Monday 13:29 (weekend)
        - leading partial: START 00:00 -> first trading day 13:29
        - trailing partial: last trading day 20:01 -> END 23:59
     Each block is labeled by the day whose MORNING it ends on.
       -> clean_close_data/{venue}/{coin}/{label_date}.csv
"""

import shutil
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd


# ---------- config ----------

START = date(2026, 9, 6)
END   = date(2026, 9, 12) 

COINS = [
    ("X:BTCUSD",     "BTCUSDT"),
    ("X:ETHUSD",     "ETHUSDT"),
    ("X:XRPUSD",     "XRPUSDT"),
    ("X:SYRUPUSD",   "SYRUPUSDT"),
    ("X:SEIUSD",     "SEIUSDT"),
    ("X:RENDERUSD",  "RENDERUSDT"),
    ("X:SUPERUSD",   "SUPERUSDT"),
    ("X:AUCTIONUSD", "AUCTIONUSDT"),
    ("X:XTZUSD",     "XTZUSDT"),
    ("X:GLMUSD",     "GLMUSDT"),
    ("X:BLURUSD",    "BLURUSDT"),
    ("X:BIGTIMEUSD", "BIGTIMEUSDT"),
    ("X:ADXUSD",     "ADXUSDT"),
    ("X:LSKUSD",     "LSKUSDT"),
    ("X:CHRUSD",     "CHRUSDT"),
]

RAW_DIR         = Path("data/trades")
CLEAN_DIR       = Path("clean_data")
CLEAN_OPEN_DIR  = Path("clean_open_data")
CLEAN_CLOSE_DIR = Path("clean_close_data")

# US regular session boundaries in UTC (EDT, Mar–Nov)
US_OPEN_UTC  = time(13, 30)
US_CLOSE_UTC = time(20, 0)

US_HOLIDAYS_2026 = {
    date(2026, 1, 1),  date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3),  date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3),  date(2026, 9, 7),  date(2026, 11, 26),
    date(2026, 12, 25),
}

BINANCE_COLS = ["agg_trade_id", "price", "quantity",
                "first_trade_id", "transact_time_ms",
                "is_buyer_maker", "is_best_match"]

MASSIVE_COLS = ["id", "participant_ts_ns", "received_ts_ns",
                "participant_ts_ms", "received_ts_ms",
                "price", "size", "exchange", "conditions"]


# ---------- helpers ----------

def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def safe_tick(ticker):
    return ticker.replace(":", "_").replace("/", "_")


def day_ms(d: date, hour=0, minute=0, second=0, ms=0) -> int:
    return int(datetime(d.year, d.month, d.day,
                        tzinfo=timezone.utc).timestamp() * 1000) \
           + ((hour * 3600 + minute * 60 + second) * 1000 + ms)


def is_us_market_open_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    if d in US_HOLIDAYS_2026:
        return False
    return True


def ts_col_for(venue: str) -> str:
    return "transact_time_ms" if venue == "binance" else "participant_ts_ms"


def expected_raw_files():
    for m_tick, b_tick in COINS:
        for d in daterange(START, END):
            yield "binance", b_tick,            d
            yield "massive", safe_tick(m_tick), d


# ---------- step 1: presence ----------

def check_presence() -> bool:
    missing = []
    total = 0
    for venue, coin_dir, d in expected_raw_files():
        total += 1
        p = RAW_DIR / venue / coin_dir / f"{d}.csv"
        if not p.exists() or p.stat().st_size == 0:
            missing.append(p)
    print(f"[presence] expected={total}  missing={len(missing)}")
    for p in missing[:20]:
        print(f"  missing: {p}")
    return not missing


# ---------- step 2: clean ----------

def clean_binance(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[c for c in BINANCE_COLS if c in df.columns]].copy()
    df = df.dropna(subset=["agg_trade_id", "price", "quantity", "transact_time_ms"])
    df["agg_trade_id"]     = df["agg_trade_id"].astype("int64")
    df["transact_time_ms"] = df["transact_time_ms"].astype("int64")
    df["price"]            = df["price"].astype(float)
    df["quantity"]         = df["quantity"].astype(float)
    df = df.sort_values("transact_time_ms", kind="mergesort")
    df = df.drop_duplicates(subset=["agg_trade_id"], keep="first")
    return df.reset_index(drop=True)


def clean_massive(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[c for c in MASSIVE_COLS if c in df.columns]].copy()
    df = df.dropna(subset=["id", "participant_ts_ms", "price", "size"])
    df["id"]                = df["id"].astype(str)          # <-- was astype("int64")
    df["participant_ts_ms"] = df["participant_ts_ms"].astype("int64")
    df["price"]             = df["price"].astype(float)
    df["size"]              = df["size"].astype(float)
    df = df.sort_values("participant_ts_ms", kind="mergesort")
    df = df.drop_duplicates(subset=["id"], keep="first")
    return df.reset_index(drop=True)


def clean_one(venue: str, coin_dir: str, d: date):
    src = RAW_DIR / venue / coin_dir / f"{d}.csv"
    if not src.exists():
        return 0, 0

    raw = pd.read_csv(src)
    cleaned = clean_binance(raw) if venue == "binance" else clean_massive(raw)

    out = CLEAN_DIR / venue / coin_dir / f"{d}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out, index=False)

    return len(raw), len(cleaned)


# ---------- step 3: build open and close periods ----------

def build_periods():
    """Return (open_periods, close_periods), each a list of
    (label_date, start_ms, end_ms)."""
    open_days = [d for d in daterange(START, END) if is_us_market_open_day(d)]

    open_periods = []
    for d in open_days:
        open_periods.append((d,
                             day_ms(d, 13, 30, 0, 0),
                             day_ms(d, 20, 0, 0, 0)))

    close_periods = []
    if open_days:
        # leading partial: START 00:00 -> first open day 13:29:59.999
        close_periods.append((open_days[0],
                              day_ms(START, 0, 0, 0, 0),
                              day_ms(open_days[0], 13, 29, 59, 999)))

        # between consecutive open days: prev 20:01 -> cur 13:29:59.999
        for i in range(1, len(open_days)):
            close_periods.append((open_days[i],
                                  day_ms(open_days[i-1], 20, 1, 0, 0),
                                  day_ms(open_days[i],   13, 29, 59, 999)))

        # trailing partial: last open day 20:01 -> END 23:59:59.999
        close_periods.append((END,
                              day_ms(open_days[-1], 20, 1, 0, 0),
                              day_ms(END, 23, 59, 59, 999)))

    return open_periods, close_periods


# ---------- step 4: slice by period ----------

def gather_rows(venue: str, coin_dir: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Concatenate cleaned per-day files that overlap [start_ms, end_ms)."""
    start_date = date.fromtimestamp(start_ms / 1000)
    end_date   = date.fromtimestamp(end_ms   / 1000)

    frames = []
    ts_col = ts_col_for(venue)
    for d in daterange(start_date, end_date):
        p = CLEAN_DIR / venue / coin_dir / f"{d}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        mask = (df[ts_col] >= start_ms) & (df[ts_col] < end_ms)
        if mask.any():
            frames.append(df[mask])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_period_file(venue, coin_dir, label_date, start_ms, end_ms, out_dir) -> int:
    df = gather_rows(venue, coin_dir, start_ms, end_ms)
    if df.empty:
        return 0
    out = out_dir / venue / coin_dir / f"{label_date}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return len(df)

def split_massive_by_exchange(source_dir: Path):
    """Split Massive files by exchange ID into Exchange{id}/ subdirectories.

    Reads from:  source_dir/massive/X_BTCUSD/date.csv
    Writes to:   source_dir/Exchange{id}/{id}_BTCUSD/date.csv

    Returns (n_files_written, {exchange_id: n_files}).
    """
    massive_root = source_dir / "massive"
    if not massive_root.exists():
        return 0, {}

    counts = {}
    n_written = 0

    for coin_dir in sorted(massive_root.iterdir()):
        if not coin_dir.is_dir():
            continue

        coin_name = coin_dir.name                  # e.g. "X_BTCUSD"
        base      = coin_name.replace("X_", "")    # "BTCUSD"

        for f in sorted(coin_dir.glob("*.csv")):
            df = pd.read_csv(f)
            if "exchange" not in df.columns:
                print(f"[skip] {f}: no 'exchange' column")
                continue

            df["exchange"] = df["exchange"].astype("int64")

            for exch_id, sub in df.groupby("exchange"):
                out_dir = source_dir / f"Exchange{exch_id}" / f"{exch_id}_{base}"
                out_dir.mkdir(parents=True, exist_ok=True)
                sub.reset_index(drop=True).to_csv(out_dir / f.name, index=False)
                n_written += 1
                counts[int(exch_id)] = counts.get(int(exch_id), 0) + 1

    return n_written, counts
# ---------- main ----------

def main():
    if not check_presence():
        print("Presence check failed — aborting.")
        return

    for d_ in (CLEAN_DIR, CLEAN_OPEN_DIR, CLEAN_CLOSE_DIR):
        if d_.exists():
            shutil.rmtree(d_)

    # ---- Step A: clean ----
    print("\n=== Cleaning raw files ===")
    n_clean = 0
    for i, (venue, coin_dir, d) in enumerate(expected_raw_files(), 1):
        raw_n, clean_n = clean_one(venue, coin_dir, d)
        if raw_n:
            n_clean += 1
        if i % 20 == 0 or raw_n == 0:
            print(f"  [{i:>3}/210] {venue:8s} {coin_dir:14s} {d}  "
                  f"{raw_n:>9,} -> {clean_n:>9,}")
    print(f"cleaned files: {n_clean}")

    open_periods, close_periods = build_periods()

    print("\n=== Open periods ===")
    for lab, s, e in open_periods:
        print(f"  {lab}  {s} -> {e}")

    print("\n=== Close periods ===")
    for lab, s, e in close_periods:
        print(f"  {lab}  {s} -> {e}  ({(e-s)/3.6e6:.2f}h)")

    # ---- Step B: open files ----
    print("\n=== Writing open files ===")
    for m_tick, b_tick in COINS:
        for lab, s, e in open_periods:
            write_period_file("binance", b_tick,            lab, s, e, CLEAN_OPEN_DIR)
            write_period_file("massive", safe_tick(m_tick), lab, s, e, CLEAN_OPEN_DIR)
    print("done")

    # ---- Step C: close files ----
    print("\n=== Writing close files ===")
    for m_tick, b_tick in COINS:
        for lab, s, e in close_periods:
            write_period_file("binance", b_tick,            lab, s, e, CLEAN_CLOSE_DIR)
            write_period_file("massive", safe_tick(m_tick), lab, s, e, CLEAN_CLOSE_DIR)
    print("done")

    # ---- Step D: split Massive by exchange ID ----
    print("\n=== Splitting Massive by exchange ID ===")
    for label, d_ in [("clean_data",      CLEAN_DIR),
                      ("clean_open_data", CLEAN_OPEN_DIR),
                      ("clean_close_data",CLEAN_CLOSE_DIR)]:
        n, counts = split_massive_by_exchange(d_)
        print(f"  {label}: {n} files written across {len(counts)} exchanges")
        for eid in sorted(counts):
            print(f"    exchange {eid:>3}: {counts[eid]} files")
    # ---- summary ----
    n_open  = sum(1 for _ in CLEAN_OPEN_DIR.rglob("*.csv"))
    n_close = sum(1 for _ in CLEAN_CLOSE_DIR.rglob("*.csv"))
    print(f"\nOpen files:  {n_open}  (expect {len(open_periods)  * len(COINS) * 2})")
    print(f"Close files: {n_close}  (expect {len(close_periods) * len(COINS) * 2})")


if __name__ == "__main__":
    main()