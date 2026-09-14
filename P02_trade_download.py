"""
download_trades.py

Download tick-level trade history for the coin universe from Binance and Massive.
One CSV per (exchange, coin, day). Progress tracked in progress.csv so the script
can be interrupted, resumed, or rerun safely.

Binance trades come from the public bulk mirror (data.binance.vision) — no rate
limit, much faster than the API. Massive trades come from the REST API.

Edit DOWNLOAD_BINANCE / DOWNLOAD_MASSIVE below, then run:
    python download_trades.py
"""

import io
import os
import time
import zipfile
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from massive import RESTClient
from trader_api import API


# ---------- config ----------

DOWNLOAD_BINANCE = True     # set False to skip Binance
DOWNLOAD_MASSIVE = True     # set False to skip Massive

START = date(2026, 9, 6)
END   = date(2026, 9, 12)    # inclusive

# Massive ticker -> Binance spot symbol
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

DATA_DIR      = "data/trades"
PROGRESS_CSV  = "progress.csv"

BINANCE_BULK_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
MASSIVE_PAGE     = 50000

PROGRESS_COLS = ["m_tick", "b_tick", "date", "exchange",
                 "status", "rows", "path", "note"]


# ---------- helpers ----------

def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def safe_tick(ticker):
    return ticker.replace(":", "_").replace("/", "_")


def ns_to_ms(ns):
    """Convert a nanosecond timestamp to milliseconds, or None."""
    if ns is None:
        return None
    return int(ns) // 1_000_000


# ---------- progress ----------

def load_progress():
    if not os.path.exists(PROGRESS_CSV):
        return pd.DataFrame({
            "m_tick":   pd.Series(dtype="object"),
            "b_tick":   pd.Series(dtype="object"),
            "date":     pd.Series(dtype="object"),
            "exchange": pd.Series(dtype="object"),
            "status":   pd.Series(dtype="object"),
            "rows":     pd.Series(dtype="float64"),
            "path":     pd.Series(dtype="object"),
            "note":     pd.Series(dtype="object"),
        })

    df = pd.read_csv(PROGRESS_CSV)

    # force text columns to object so we can write "" into them later
    for c in ("m_tick", "b_tick", "date", "exchange", "status", "path", "note"):
        df[c] = df[c].astype("object")
    df["rows"] = pd.to_numeric(df["rows"], errors="coerce")

    return df


def save_progress(df):
    df.to_csv(PROGRESS_CSV, index=False)


def upsert_progress(df, m_tick, b_tick, d, exchange,
                    status, rows=None, path=None, note=""):
    mask = (
        (df["m_tick"] == m_tick)
        & (df["b_tick"] == b_tick)
        & (df["date"] == str(d))
        & (df["exchange"] == exchange)
    )
    if mask.any():
        df.loc[mask, "status"] = status
        if rows is not None:
            df.loc[mask, "rows"] = rows
        if path is not None:
            df.loc[mask, "path"] = path
        df.loc[mask, "note"] = note
    else:
        df.loc[len(df)] = {
            "m_tick": m_tick, "b_tick": b_tick, "date": str(d),
            "exchange": exchange, "status": status, "rows": rows,
            "path": path, "note": note,
        }
    return df


def already_done(df, m_tick, b_tick, d, exchange):
    return (
        (df["m_tick"] == m_tick)
        & (df["b_tick"] == b_tick)
        & (df["date"] == str(d))
        & (df["exchange"] == exchange)
        & (df["status"] == "downloaded")
    ).any()


# ---------- Binance (bulk mirror) ----------

BINANCE_BULK_COLS = [
    "agg_trade_id", "price", "quantity",
    "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
]


# actual column order in the bulk file
BULK_COLS = ["price", "quantity", "agg_trade_id", "first_trade_id",
             "transact_time", "is_buyer_maker", "is_best_match"]


def fetch_binance_day_bulk(symbol, d):
    """Download one day of aggTrades from Binance's bulk mirror.
    Handles files with or without a header row, and renames columns
    to the canonical schema."""
    url = (f"{BINANCE_BULK_URL}/{symbol}/"
           f"{symbol}-aggTrades-{d.isoformat()}.zip")

    for attempt in range(5):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return pd.DataFrame()
            r.raise_for_status()
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        raw = z.read(name)

    # decide whether the first line is a header or data
    first_line = raw.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    first_cell = first_line.split(",", 1)[0].strip().strip('"')

    try:
        float(first_cell)
        has_header = False
    except ValueError:
        has_header = True

    if has_header:
        df = pd.read_csv(io.BytesIO(raw), skiprows=1, names=BULK_COLS)
    else:
        df = pd.read_csv(io.BytesIO(raw), header=None, names=BULK_COLS)

    return df
def download_binance_day(symbol, d, out_path):
    df = fetch_binance_day_bulk(symbol, d)

    if not df.empty:
        df["price"]    = df["price"].astype(float)
        df["quantity"] = df["quantity"].astype(float)

        # bulk file writes transact_time in MICROSECONDS; convert to ms
        df["transact_time_ms"] = (df["transact_time"] // 1000).astype("int64")

        df = df[["agg_trade_id", "price", "quantity",
                 "first_trade_id", "transact_time_ms",
                 "is_buyer_maker", "is_best_match"]]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return len(df)


# ---------- Massive ----------

def fetch_massive_day(client, ticker, d):
    """Fetch every trade for one ticker on one UTC calendar day via the REST API."""
    rows = []
    for t in client.list_trades(
        ticker=ticker,
        timestamp=d.isoformat(),
        limit=MASSIVE_PAGE,
        order="asc",
        sort="timestamp",
    ):
        participant_ns = getattr(t, "participant_timestamp", None)
        received_ns    = getattr(t, "received_timestamp",    None)

        rows.append({
            "id":                getattr(t, "id",       None),
            "participant_ts_ns": participant_ns,
            "received_ts_ns":    received_ns,
            "participant_ts_ms": ns_to_ms(participant_ns),
            "received_ts_ms":    ns_to_ms(received_ns),
            "price":             getattr(t, "price",    None),
            "size":              getattr(t, "size",     None),
            "exchange":          getattr(t, "exchange", None),
            "conditions":        str(getattr(t, "conditions", None)),
        })
        if len(rows) % 100000 == 0:
            print(f"    rows so far {len(rows):,}")
    return rows


def download_massive_day(client, ticker, d, out_path):
    rows = fetch_massive_day(client, ticker, d)
    df = pd.DataFrame(rows)

    if not df.empty:
        df["price"] = df["price"].astype(float)
        df["size"]  = df["size"].astype(float)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return len(df)


# ---------- driver ----------

def download_trades(use_binance: bool, use_massive: bool):
    """Download trade history.

    use_binance=True  -> pull from Binance bulk mirror
    use_massive=True  -> pull from Massive REST API
    Both False        -> no-op
    """
    exchanges = []
    if use_binance:
        exchanges.append("binance")
    if use_massive:
        exchanges.append("massive")

    if not exchanges:
        print("Both flags False — nothing to do.")
        return

    print(f"Downloading from: {', '.join(exchanges)}")
    print(f"Universe: {len(COINS)} coins × {START} → {END}")

    progress = load_progress()
    client   = RESTClient(API) if use_massive else None

    total = len(COINS) * len(list(daterange(START, END))) * len(exchanges)
    done  = 0

    for m_tick, b_tick in COINS:
        for d in daterange(START, END):
            for ex in exchanges:
                done += 1
                tag = f"[{done:>4}/{total}] {ex:8s} {m_tick:14s} {d}"

                if already_done(progress, m_tick, b_tick, d, ex):
                    print(f"{tag}  SKIP")
                    continue

                print(f"{tag}  downloading...")

                if ex == "binance":
                    out = f"{DATA_DIR}/binance/{b_tick}/{d}.csv"
                else:
                    out = f"{DATA_DIR}/massive/{safe_tick(m_tick)}/{d}.csv"

                progress = upsert_progress(progress, m_tick, b_tick, d, ex,
                                           "downloading", path=out)
                save_progress(progress)

                try:
                    if ex == "binance":
                        n = download_binance_day(b_tick, d, out)
                    else:
                        n = download_massive_day(client, m_tick, d, out)

                    progress = upsert_progress(progress, m_tick, b_tick, d, ex,
                                               "downloaded", rows=n, path=out)
                    save_progress(progress)
                    print(f"{tag}  OK ({n:,} rows)")

                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                    progress = upsert_progress(progress, m_tick, b_tick, d, ex,
                                               "failed", path=out, note=err)
                    save_progress(progress)
                    print(f"{tag}  FAILED: {err}")

    print("\nDone. Status counts:")
    print(progress["status"].value_counts().to_string())


if __name__ == "__main__":
    download_trades(DOWNLOAD_BINANCE, DOWNLOAD_MASSIVE)