#!/usr/bin/env python3
"""
P06_linear_VWAP.py   (construction step)

Walk every file under {ROOT}/clean_data, {ROOT}/clean_open_data and
{ROOT}/clean_close_data and produce an augmented copy under
{ROOT}/linear_vwap_construction/.

Files are discovered dynamically -- this script never hardcodes a coin list
or an exchange list.  It walks each source dir's exchange folders
(binance/, massive/, Exchange1/, ...), then each coin subfolder, then every
*.csv inside.

Output layout is namespaced by source directory:

    linear_vwap_construction/clean_data/...
    linear_vwap_construction/clean_open_data/...
    linear_vwap_construction/clean_close_data/...

For each a (ms) in VWAP_INTERVALS, adds six columns:

    a_{a}_possible   bool   window [t-a/2, t+a/2] fits inside the file
    a_{a}_5_trades   bool   possible AND >= MIN_TRADES_IN_WINDOW trades in it
    a_{a}_side       float  1=buyers only, 2=sellers only, 3=mixed, NaN=empty
    a_{a}_next       float  row index of the first possible row >= a ms later
    a_{a}_vwap       float  size-weighted price over the centred a ms window
    a_{a}_imb        float  taker-buy notional imbalance of the window, [-1, 1]

`a_{a}_imb` is new in this version.  It used to be computed independently by
P10; it belongs here, next to the VWAP it shares all its prefix sums with, so
that P10 no longer has to re-read and re-reduce the trade files just to get it.


This module is also the shared IO / config layer
------------------------------------------------
P07-P10 import `get_root`, `set_root`, `read_columns`, `count_csv_rows`,
`write_columns_chunked`, `read_meta`, `write_meta` and the exchange column
helpers from here rather than re-defining their own copies.


Caching
-------
Each destination file gets a sidecar `{name}.csv.meta.json` recording the
cache version, the source file's mtime and size, the row count and the list
of intervals written.  A destination is reused only when all of those match.

The previous scheme instead sniffed the destination's own header for
`a_{a}_possible` columns and spot-checked two rows of `a_{a}_next` for
NaN-ness.  That had a concrete failure mode: a destination written before
`a_{a}_vwap` existed still had `a_{a}_possible` and a valid `a_{a}_next`, so
every interval passed validation, the file was skipped, and `a_{a}_vwap` was
never added -- which then made P07 silently emit no `ret_{a}` columns at all
for that file, forever.  A content-independent stamp cannot drift like that.

Because caching is now all-or-nothing per file, the partial per-interval
merge path is gone too: recomputing the interval math for all six intervals
is cheap relative to the CSV read and write that a partial update needs
anyway.


Memory design
-------------
Source rows carry object/string columns (a UUID-style trade id, exchange
condition codes) that are expensive per row -- far more so than the int64 /
float64 data the interval math actually needs.  Materialising the entire row
set at once is what causes OOM kills on unusually large files.  So:

  1. A light pass reads ONLY timestamp, price, size/quantity and the column
     used to derive is_buyer, in CHUNK_ROWS-row chunks, and computes every
     a_* column as plain numpy arrays.

  2. A chunked write pass re-reads the source in CHUNK_ROWS-row chunks purely
     to carry the original columns through, slices the precomputed arrays to
     match, and appends the augmented chunk to the destination.

This assumes the source is sorted by timestamp (P03 sorts before writing).
That is checked explicitly; an unsorted source falls back to a full
in-memory sort-and-write for that one file rather than silently emitting
misaligned rows.

Writes are atomic (temp file + rename), so a process killed mid-write cannot
leave a partial destination that a later run mistakes for a cached one.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================ config

_ROOT = Path(os.environ.get("LEADLAG_ROOT", Path.home()))


def get_root() -> Path:
    """Project root.  Every path in P06-P10 is built from this."""
    return _ROOT


def set_root(p) -> Path:
    """Point the whole pipeline at a different project root."""
    global _ROOT
    _ROOT = Path(p).expanduser()
    return _ROOT


SOURCE_DIRS = ["clean_data", "clean_open_data", "clean_close_data"]

CONSTRUCT_DIR = "linear_vwap_construction"

VWAP_INTERVALS = [200, 500, 1000, 5000, 10000, 30000]     # ms

MIN_TRADES_IN_WINDOW = 5

# Rows per chunk for every chunked read and write in the pipeline.
CHUNK_ROWS = 200_000

FLOAT_FMT = "%.10g"

INTERVAL_SUFFIXES = ("possible", "5_trades", "side", "next", "vwap", "imb")

# Bump to invalidate every linear_vwap_construction/ file.
#   1  original (no a_{a}_vwap)
#   2  a_{a}_vwap added
#   3  a_{a}_imb added, meta-json caching, vectorised a_{a}_next
CACHE_VERSION = 3

# Backwards-compatible aliases for callers that used the old names.
VWAP_interval = VWAP_INTERVALS
WRITE_CHUNK_SIZE = CHUNK_ROWS


# ============================================================ shared light IO

def count_csv_rows(path: Path) -> int:
    """Data-row count (newline count minus the header), read in binary."""
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            n += block.count(b"\n")
    return max(n - 1, 0)


def csv_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def read_columns(path: Path, spec: dict[str, str]) -> dict[str, np.ndarray]:
    """Read only `spec` columns ({name: dtype}) in CHUNK_ROWS chunks and
    return them as 1-D numpy arrays.  Columns absent from the file are
    silently skipped, so callers must check what they got back."""
    header = csv_header(path)
    use = [c for c in spec if c in header]
    if not use:
        return {}
    pieces: dict[str, list[np.ndarray]] = {c: [] for c in use}
    for chunk in pd.read_csv(path, usecols=use, dtype={c: spec[c] for c in use},
                             chunksize=CHUNK_ROWS):
        for c in use:
            pieces[c].append(chunk[c].to_numpy(copy=True))
        del chunk
    out: dict[str, np.ndarray] = {}
    for c in use:
        out[c] = pieces[c][0] if len(pieces[c]) == 1 else np.concatenate(pieces[c])
        pieces[c] = []
    return out


def write_columns_chunked(dst: Path, columns: dict[str, np.ndarray]) -> None:
    """Write a dict of equal-length arrays to a CSV, CHUNK_ROWS at a time,
    atomically."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    n = len(next(iter(columns.values())))
    with open(tmp, "w", newline="") as f:
        first = True
        for s in range(0, max(n, 1), CHUNK_ROWS):
            part = pd.DataFrame({k: v[s:s + CHUNK_ROWS] for k, v in columns.items()})
            part.to_csv(f, index=False, header=first, float_format=FLOAT_FMT)
            first = False
            if n == 0:
                break
    tmp.replace(dst)


# ============================================================ shared cache meta

def meta_path(dst: Path) -> Path:
    return dst.with_name(dst.name + ".meta.json")


def read_meta(dst: Path) -> dict | None:
    p = meta_path(dst)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def write_meta(dst: Path, meta: dict) -> None:
    p = meta_path(dst)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, sort_keys=True))
    tmp.replace(p)


def src_stamp(src: Path) -> dict:
    """Identity of an input file, for cache validation."""
    st = src.stat()
    return {"src_mtime": int(st.st_mtime), "src_size": int(st.st_size)}


def stamp_matches(meta: dict | None, src: Path, version: int) -> bool:
    if not meta or meta.get("version") != version:
        return False
    want = src_stamp(src)
    return all(meta.get(k) == v for k, v in want.items())


# ============================================================ exchange columns

def source_columns(exchange: str) -> tuple[str, str]:
    """(timestamp column, column used to derive is_buyer) for an exchange."""
    if exchange == "binance":
        return "transact_time_ms", "is_buyer_maker"
    return "participant_ts_ms", "conditions"


def time_col_for(exchange: str) -> str:
    return source_columns(exchange)[0]


def volume_column(exchange: str) -> str:
    return "quantity" if exchange == "binance" else "size"


def derive_is_buyer(exchange: str, side) -> np.ndarray:
    """Taker-buy flag.

    binance: `is_buyer_maker` True means the BUYER was the maker, so the
    aggressor was a seller -- hence the negation.
    massive-schema: condition code contains '2' (buy) and not '1' (sell).
    """
    s = pd.Series(np.asarray(side))
    if exchange == "binance":
        if s.dtype == bool:
            flag = s.to_numpy()
        else:
            flag = s.astype(str).str.strip().str.lower().isin(
                ["true", "1", "t", "yes"]).to_numpy()
        return ~flag
    cond = s.astype(str)
    has_buy = cond.str.contains("2", regex=False)
    has_sell = cond.str.contains("1", regex=False)
    return (has_buy & ~has_sell).to_numpy()


# aliases kept so older callers importing the private names keep working
_source_columns = source_columns
_volume_column = volume_column
_derive_is_buyer = derive_is_buyer


# ============================================================ light loader

def load_light(path: Path, exchange: str):
    """Read only the four columns the interval math needs.

    Returns (ts_ms, price, volume, is_buyer, sorted_ok).
    """
    ts_col, side_col = source_columns(exchange)
    vol_col = volume_column(exchange)

    header = csv_header(path)
    for col in (ts_col, side_col, "price", vol_col):
        if col not in header:
            raise ValueError(f"{path}: missing {col} column")

    cols = read_columns(path, {ts_col: "int64", "price": "float64",
                               vol_col: "float64", side_col: "object"})
    ts_ms = cols[ts_col]
    price = cols["price"]
    volume = cols[vol_col]
    is_buyer = derive_is_buyer(exchange, cols[side_col])

    sorted_ok = bool(np.all(ts_ms[1:] >= ts_ms[:-1])) if len(ts_ms) > 1 else True
    return ts_ms, price, volume, is_buyer, sorted_ok


# ============================================================ core interval math

def compute_interval_arrays(ts_ms: np.ndarray,
                            price: np.ndarray,
                            volume: np.ndarray,
                            is_buyer: np.ndarray,
                            a_values) -> dict[str, np.ndarray]:
    """Compute every a_{a}_{suffix} column as plain numpy arrays.

    Window for row i is W1 = [t_i - a//2, t_i + a//2], both ends inclusive.
    `a_{a}_next` is the first *possible* row j with t_j >= t_i + a, so W2 and
    W1 touch at most at a single instant and never overlap.
    """
    n = len(ts_ms)
    out: dict[str, np.ndarray] = {}
    if not a_values:
        return out
    if n == 0:
        # Keep the column schema identical for empty files, so a zero-row day
        # still has every a_* column and downstream discovery doesn't see it
        # as a file with no intervals.
        dt = {"possible": bool, "5_trades": bool}
        for a in sorted(a_values):
            for s in INTERVAL_SUFFIXES:
                out[f"a_{a}_{s}"] = np.empty(0, dtype=dt.get(s, float))
        return out

    ts_ms = np.asarray(ts_ms)
    price = np.asarray(price, dtype=float)
    volume = np.asarray(volume, dtype=float)
    is_buyer = np.asarray(is_buyer, dtype=bool)

    notional = price * volume
    buy_pref = np.concatenate([[0], np.cumsum(is_buyer.astype("int64"))])
    sell_pref = np.concatenate([[0], np.cumsum((~is_buyer).astype("int64"))])
    volume_pref = np.concatenate([[0.0], np.cumsum(volume)])
    dollar_pref = np.concatenate([[0.0], np.cumsum(notional)])
    buy_dollar_pref = np.concatenate(
        [[0.0], np.cumsum(np.where(is_buyer, notional, 0.0))])
    del notional

    first_ts = ts_ms[0]
    last_ts = ts_ms[-1]

    for a in sorted(a_values):
        half = a // 2

        possible = (ts_ms - first_ts >= half) & (last_ts - ts_ms >= half)
        out[f"a_{a}_possible"] = possible

        left_idx = np.searchsorted(ts_ms, ts_ms - half, side="left")
        right_idx = np.searchsorted(ts_ms, ts_ms + half, side="right")
        counts = right_idx - left_idx

        out[f"a_{a}_5_trades"] = possible & (counts >= MIN_TRADES_IN_WINDOW)

        buy_in = buy_pref[right_idx] - buy_pref[left_idx]
        sell_in = sell_pref[right_idx] - sell_pref[left_idx]
        volume_in = volume_pref[right_idx] - volume_pref[left_idx]
        dollar_in = dollar_pref[right_idx] - dollar_pref[left_idx]
        buy_dollar_in = buy_dollar_pref[right_idx] - buy_dollar_pref[left_idx]

        side = np.full(n, np.nan)
        valid = counts > 0
        side[valid & (buy_in > 0) & (sell_in == 0)] = 1
        side[valid & (sell_in > 0) & (buy_in == 0)] = 2
        side[valid & (buy_in > 0) & (sell_in > 0)] = 3
        out[f"a_{a}_side"] = side

        positive_volume = volume_in > 0
        vwap = np.full(n, np.nan)
        vwap[positive_volume] = (dollar_in[positive_volume]
                                 / volume_in[positive_volume])
        out[f"a_{a}_vwap"] = vwap

        # taker-buy notional imbalance in [-1, 1]
        imb = np.full(n, np.nan)
        positive_dollar = dollar_in > 0
        imb[positive_dollar] = (
            (2.0 * buy_dollar_in[positive_dollar] - dollar_in[positive_dollar])
            / dollar_in[positive_dollar])
        out[f"a_{a}_imb"] = imb

        # ---- next possible row at least a ms later (vectorised) ----
        # The old implementation was an O(n) Python two-pointer loop; this is
        # the same result -- first possible row with t_j >= t_i + a -- in one
        # searchsorted.  (Since a > 0, the result is always strictly after i.)
        next_a = np.full(n, np.nan)
        pos = np.flatnonzero(possible)
        if len(pos):
            k = np.searchsorted(ts_ms[pos], ts_ms[pos] + a, side="left")
            ok = k < len(pos)
            next_a[pos[ok]] = pos[k[ok]]
        out[f"a_{a}_next"] = next_a

    return out


def interval_columns(a_values) -> list[str]:
    return [f"a_{a}_{s}" for a in sorted(a_values) for s in INTERVAL_SUFFIXES]


# ============================================================ writers

def write_chunked(src: Path, dst: Path, exchange: str,
                  is_buyer: np.ndarray, arrays: dict[str, np.ndarray],
                  n_rows: int) -> None:
    """Stream src in CHUNK_ROWS-row chunks, attach the precomputed is_buyer
    flag and a_* arrays by row offset, and write dst atomically."""
    ts_col, _ = source_columns(exchange)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")

    offset = 0
    first_chunk = True
    try:
        with open(tmp_path, "w", newline="") as f:
            for chunk in pd.read_csv(src, chunksize=CHUNK_ROWS):
                m = len(chunk)
                chunk = chunk.rename(columns={ts_col: "ts_ms"})
                chunk["is_buyer"] = is_buyer[offset:offset + m]
                for col, arr in arrays.items():
                    chunk[col] = arr[offset:offset + m]
                chunk.to_csv(f, index=False, header=first_chunk,
                             float_format=FLOAT_FMT)
                first_chunk = False
                offset += m
                del chunk
            if first_chunk:
                # A zero-row source yields no chunks at all.  Without this the
                # destination would be a zero-byte file with no header, which
                # every downstream reader raises EmptyDataError on.
                head = [("ts_ms" if c == ts_col else c) for c in csv_header(src)]
                head += ["is_buyer"] + list(arrays)
                pd.DataFrame(columns=head).to_csv(f, index=False)
        if offset != n_rows:
            raise ValueError(
                f"{src}: row count changed between passes "
                f"({offset} on write pass vs {n_rows} on light pass) -- "
                "refusing to write a possibly-misaligned output")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.replace(dst)


def write_full_sorted(src: Path, dst: Path, exchange: str, a_values) -> int:
    """Fallback for a source that is not time-sorted: read it whole, sort it,
    recompute against the sorted order, write it out.  Only reachable when
    the sortedness check fails, which should not happen for P03 output."""
    ts_col, side_col = source_columns(exchange)
    vol_col = volume_column(exchange)
    df = pd.read_csv(src)
    for col in (ts_col, side_col, "price", vol_col):
        if col not in df.columns:
            raise ValueError(f"{src}: missing {col} column")
    df = df.rename(columns={ts_col: "ts_ms"})
    df = df.sort_values("ts_ms", kind="mergesort").reset_index(drop=True)

    is_buyer = derive_is_buyer(exchange, df[side_col])
    df["is_buyer"] = is_buyer
    arrays = compute_interval_arrays(df["ts_ms"].to_numpy(dtype="int64"),
                                     df["price"].to_numpy(dtype=float),
                                     df[vol_col].to_numpy(dtype=float),
                                     is_buyer, a_values)
    for col, arr in arrays.items():
        df[col] = arr

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    df.to_csv(tmp_path, index=False, float_format=FLOAT_FMT)
    tmp_path.replace(dst)
    return len(df)


# ============================================================ driver

def _fmt(lst) -> str:
    return "[" + ",".join(str(x) for x in sorted(lst)) + "]"


def is_cached(dst: Path, src: Path, a_values) -> bool:
    if not (dst.exists() and dst.stat().st_size > 0):
        return False
    meta = read_meta(dst)
    if not stamp_matches(meta, src, CACHE_VERSION):
        return False
    have = set(meta.get("intervals", []))
    return set(a_values).issubset(have)


def process_one_file(src: Path, rel: Path) -> int:
    """Returns row count, or -1 if skipped / failed."""
    exchange = rel.parts[1]          # {source_dir}/{exchange}/{coin}/{date}.csv
    kind = "binance" if exchange == "binance" else "massive"
    dst = get_root() / CONSTRUCT_DIR / rel

    if is_cached(dst, src, VWAP_INTERVALS):
        print(f"      A={_fmt(VWAP_INTERVALS)}  ACTION=skip")
        return -1

    action = "recompute" if dst.exists() else "fresh"
    print(f"      A={_fmt(VWAP_INTERVALS)}  ACTION={action}")

    try:
        ts_ms, price, volume, is_buyer, sorted_ok = load_light(src, kind)
    except Exception as e:
        print(f"      [skip] cannot load source ({e})")
        return -1

    n_rows = len(ts_ms)

    if not sorted_ok:
        print("      [warn] source not time-sorted; using full in-memory fallback")
        try:
            n_rows = write_full_sorted(src, dst, kind, VWAP_INTERVALS)
        except Exception as e:
            print(f"      [skip] fallback failed ({e})")
            return -1
    else:
        arrays = compute_interval_arrays(ts_ms, price, volume, is_buyer,
                                         VWAP_INTERVALS)
        try:
            write_chunked(src, dst, kind, is_buyer, arrays, n_rows)
        except Exception as e:
            print(f"      [skip] write failed ({e})")
            return -1

    write_meta(dst, {"version": CACHE_VERSION,
                     "n_rows": n_rows,
                     "intervals": sorted(VWAP_INTERVALS),
                     "suffixes": list(INTERVAL_SUFFIXES),
                     "exchange": kind,
                     **src_stamp(src)})
    return n_rows


def walk_source(source_name: str, counters: dict) -> None:
    """Discover every exchange folder, coin folder and csv under a source dir."""
    source_dir = get_root() / source_name
    if not source_dir.exists():
        print(f"[skip] {source_dir} missing")
        return

    for exchange_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        name = exchange_dir.name
        if name not in ("binance", "massive") and not name.startswith("Exchange"):
            print(f"  [skip dir] {exchange_dir}: unrecognized exchange folder name")
            continue

        for coin_dir in sorted(p for p in exchange_dir.iterdir() if p.is_dir()):
            for f in sorted(coin_dir.glob("*.csv")):
                rel = Path(source_name) / f.relative_to(source_dir)
                print(f"  [{counters['total'] + 1:>4}] {rel}")

                n = process_one_file(f, rel)
                counters["total"] += 1
                if n == -1:
                    counters["skipped"] += 1
                    print("      -> SKIP")
                else:
                    counters["computed"] += 1
                    print(f"      -> OK ({n:,} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="project root (default: $LEADLAG_ROOT or ~)")
    ap.add_argument("--roots", default=",".join(SOURCE_DIRS),
                    help="comma-separated data roots to process")
    args = ap.parse_args()
    if args.root:
        set_root(args.root)

    out_dir = get_root() / CONSTRUCT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    counters = {"total": 0, "skipped": 0, "computed": 0}
    for source_name in [r for r in args.roots.split(",") if r]:
        print(f"\n=== {source_name} ===")
        walk_source(source_name, counters)

    print("\nDone.")
    print(f"  total:    {counters['total']}")
    print(f"  computed: {counters['computed']}")
    print(f"  skipped:  {counters['skipped']}")
    print(f"  output:   {out_dir}/")


if __name__ == "__main__":
    main()
