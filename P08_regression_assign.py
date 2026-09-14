#!/usr/bin/env python3
"""
P08_regression_assign.py

Generalized cross-exchange lead-lag index assignment.

For every ordered pair of exchanges (lead, lag) with lead != lag, and every
(coin, day) that exists on both, find for each lead trade at time t the lag
trade nearest to (t + b) that also falls within

    [t + max(2a, b - c),  t + b + c]

Both endpoints inclusive, all offsets in milliseconds.  Only (a, b, c)
triples with b > 2a are produced, so the two VWAP windows cannot overlap.


This module owns the naming machinery
-------------------------------------
`effective_exchanges`, `side_label`, `label_to_effective`, `to_massive_name`,
`from_massive_name`, `valid_combos`, `ordered_pairs` and `valid_label_pairs`
live here.  P09 and P10 used to each carry their own copy of all of them --
three definitions of the same mapping that had to be kept in step by hand.
They now import these from here.  Light CSV IO and cache-meta helpers come
from P06.


Exchange set
------------
    binance, massive, Exchange1, Exchange2, Exchange6, Exchange23

Massive substitution
--------------------
`massive` is an aggregate of Exchange1/2/6/23.  Whenever one side of a pair is
`massive` and the other is one of those constituents, the `massive` side is
replaced by the remaining three: their CSVs are read, concatenated and sorted
by timestamp, and the side is labelled `massive_minus_<counterpart>`.

Coin-dir naming
---------------
    binance      BTCUSDT
    massive      X_BTCUSD
    ExchangeN    N_BTCUSD

All mapped through the canonical Massive-format name (X_BTCUSD).

Outputs, one CSV per (data_root, lead, lag, coin, day, a, b, c):

    {ROOT}/regression_index/{data_root}/{lead_label}/{coin}/{date}/
        {lead_label}_{lag_label}_{a}_{b}_{c}.csv

Columns:
    lead_{lead_label}  int    row index into the lead (combined) file
    lag_{lag_label}    int    row index into the lag (combined) file, -1 if none
    no_duplications    bool   True for the first lead row that claims a lag row
    found_counterpart  bool   True if a lag trade was found in the window


Paths
-----
Every path is built from P06.get_root() (`$LEADLAG_ROOT`, default `~`), not
from the process's working directory.  Previously this script used bare
relative paths while P10 used a configurable root, so running P10 with
`--root` from a different directory made the two disagree about where
`regression_index/` was -- silently, by producing an empty second copy.


Caching
-------
Per (data_root, lead, lag, coin, day) a sidecar
`{lead}_{lag}.meta.json` in the date directory records the cache version, the
mtime and size of every constituent file on both sides, the lead row count
and the combos written.  Combos are also individually row-count checked, so a
half-finished run still resumes correctly.


Memory design
-------------
`nearest_in_range` -- the only thing that consumes lead/lag data -- needs
nothing but the two timestamp arrays.  `load_side_ts` reads ONLY the
timestamp column, in chunks, for each constituent file, concatenates, and
stable-sorts only if the result is not already monotonic.  Ties break in
concatenation order, matching what P09/P10 reproduce when they rebuild the
same combined side from `linear_return/` to look these indices back up.

Per-combo outputs are written atomically (temp file + rename).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import P06_linear_VWAP as P06
from P06_linear_VWAP import (
    CHUNK_ROWS, count_csv_rows, csv_header, get_root, read_columns, read_meta,
    set_root, time_col_for, write_meta,
)


# ============================================================ config

DATA_ROOTS = ["clean_data", "clean_open_data", "clean_close_data"]

EXCHANGES = ["binance", "massive",
             "Exchange1", "Exchange2", "Exchange6", "Exchange23"]

MASSIVE_CONSTITUENTS = ["Exchange1", "Exchange2", "Exchange6", "Exchange23"]

INDEX_DIR = "regression_index"

# intervals a, in milliseconds -- kept in step with P06
A_VALUES = list(P06.VWAP_INTERVALS)

# (b, c) pairs: b = lag in ms, c = tolerance in ms
PARAMS = [(5000, 500), (10000, 1000), (30000, 3000), (60000, 6000)]

PRIMARY_PAIRS = [("binance", "massive"), ("massive", "binance")]

# optional filter: list of (lead, lag) tuples, or None for every ordered pair
PAIR_FILTER = None

CACHE_VERSION = 2


# ============================================================ combos and pairs

def valid_combos():
    """Yield (a, b, c) triples with b > 2a."""
    for a in A_VALUES:
        for b, c in PARAMS:
            if b > 2 * a:
                yield a, b, c


def ordered_pairs():
    """Yield (lead, lag) exchange tuples for every lead != lag."""
    if PAIR_FILTER is not None:
        for lead, lag in PAIR_FILTER:
            if lead in EXCHANGES and lag in EXCHANGES and lead != lag:
                yield lead, lag
        return
    for lead in EXCHANGES:
        for lag in EXCHANGES:
            if lead != lag:
                yield lead, lag


# ============================================================ massive substitution

def effective_exchanges(exchange: str, counterpart: str) -> list[str]:
    """Which exchange(s) actually supply the data for one side of a pair."""
    if exchange == "massive" and counterpart in MASSIVE_CONSTITUENTS:
        return [e for e in MASSIVE_CONSTITUENTS if e != counterpart]
    return [exchange]


def side_label(original: str, effective: list[str], counterpart: str) -> str:
    """Label used in output paths and column names for one side."""
    if len(effective) == 1 and effective[0] == original:
        return original
    return f"{original}_minus_{counterpart}"


def label_to_effective(label: str) -> list[str] | None:
    """Inverse of side_label: label -> constituent exchanges, or None."""
    if label in EXCHANGES:
        return [label]
    if label.startswith("massive_minus_"):
        counter = label[len("massive_minus_"):]
        if counter in MASSIVE_CONSTITUENTS:
            return [e for e in MASSIVE_CONSTITUENTS if e != counter]
    return None


def valid_label_pairs(mode: str = "all") -> list[tuple[str, str]]:
    """Deduplicated (lead_label, lag_label) pairs.  mode='primary' restricts
    to the binance/massive directions; otherwise those come first so a
    partial run still produces the headline results."""
    base = (PRIMARY_PAIRS if mode == "primary"
            else PRIMARY_PAIRS + [p for p in ordered_pairs()
                                  if p not in PRIMARY_PAIRS])
    out, seen = [], set()
    for lo, go in base:
        le, ge = effective_exchanges(lo, go), effective_exchanges(go, lo)
        ll, gl = side_label(lo, le, go), side_label(go, ge, lo)
        if (ll, gl) not in seen:
            seen.add((ll, gl))
            out.append((ll, gl))
    return out


# ============================================================ coin-name mapping

def to_massive_name(coin_dir: str, exchange: str) -> str | None:
    """Canonical Massive-format name X_BTCUSD from any exchange's coin dir."""
    if exchange == "binance":
        return f"X_{coin_dir[:-4]}USD" if coin_dir.endswith("USDT") else None
    if exchange == "massive" or exchange.startswith("massive_minus_"):
        return (coin_dir if coin_dir.startswith("X_") and coin_dir.endswith("USD")
                else None)
    if exchange.startswith("Exchange"):
        parts = coin_dir.split("_", 1)
        return f"X_{parts[1]}" if len(parts) == 2 else None
    return None


def from_massive_name(massive_name: str, exchange: str) -> str | None:
    """Coin directory name for the given exchange, from X_BTCUSD."""
    if not (massive_name.startswith("X_") and massive_name.endswith("USD")):
        return None
    base = massive_name[2:-3]
    if exchange == "binance":
        return f"{base}USDT"
    if exchange == "massive":
        return massive_name
    if exchange.startswith("Exchange"):
        return f"{exchange.replace('Exchange', '')}_{base}USD"
    return None


def coins_for_exchange(data_root: str, exchange: str) -> dict | None:
    """{massive_name: coin_dir_path}, or None if the exchange dir is missing."""
    root = get_root() / data_root / exchange
    if not root.exists():
        return None
    out: dict[str, Path] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        mn = to_massive_name(d.name, exchange)
        if mn is not None:
            out[mn] = d
    return out


# ============================================================ core alignment

def nearest_in_range(lead_ts: np.ndarray, lag_ts: np.ndarray,
                     b: int, c: int, a: int):
    """Return (lag_idx, found) for every lead row.

    Window: [t + max(2a, b - c), t + b + c], endpoints inclusive.
    Nearest to (t + b) wins; ties favour the earlier index.
    """
    n, m = len(lead_ts), len(lag_ts)
    if n == 0 or m == 0:
        return np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=bool)

    lower_off = max(2 * a, b - c)
    upper_off = b + c

    lo = np.searchsorted(lag_ts, lead_ts + lower_off, side="left")
    hi = np.searchsorted(lag_ts, lead_ts + upper_off, side="right")
    target = lead_ts + b
    p = np.searchsorted(lag_ts, target, side="left")

    c1, c2 = p - 1, p
    BIG = np.iinfo(np.int64).max

    v1 = (c1 >= lo) & (c1 < hi) & (c1 >= 0) & (c1 < m)
    v2 = (c2 >= lo) & (c2 < hi) & (c2 >= 0) & (c2 < m)

    d1 = np.full(n, BIG, dtype=np.int64)
    d2 = np.full(n, BIG, dtype=np.int64)

    i1 = np.flatnonzero(v1)
    if len(i1):
        d1[i1] = np.abs(lag_ts[c1[i1]] - target[i1])
    i2 = np.flatnonzero(v2)
    if len(i2):
        d2[i2] = np.abs(lag_ts[c2[i2]] - target[i2])

    use1 = d1 <= d2
    chosen = np.where(use1, c1, c2)
    dist = np.where(use1, d1, d2)

    found = dist < BIG
    lag_idx = np.full(n, -1, dtype=np.int64)
    lag_idx[found] = chosen[found]
    return lag_idx, found


def compute_no_dup(lag_idx: np.ndarray) -> np.ndarray:
    """True for the first lead row that claims a given lag row."""
    n = len(lag_idx)
    no_dup = np.zeros(n, dtype=bool)
    positions = np.flatnonzero(lag_idx >= 0)
    if not len(positions):
        return no_dup
    _, first_idx = np.unique(lag_idx[positions], return_index=True)
    no_dup[positions[first_idx]] = True
    return no_dup


# ============================================================ side loading

def side_paths(data_root: str, exchanges: list[str],
               massive_name: str, date: str) -> list[Path] | None:
    """CSV path for each exchange in a side, or None if any is missing."""
    paths = []
    for exch in exchanges:
        coin_dir = from_massive_name(massive_name, exch)
        if coin_dir is None:
            return None
        path = get_root() / data_root / exch / coin_dir / f"{date}.csv"
        if not path.exists():
            return None
        paths.append(path)
    return paths


def load_side_ts(data_root: str, exchanges: list[str],
                 massive_name: str, date: str) -> np.ndarray | None:
    """Read ONLY the timestamp column for one side, in chunks.

    Concatenates in `exchanges` order, then stable-sorts if not already
    monotonic -- identical ordering to what P09/P10 reproduce when they
    rebuild the same side from linear_return/.
    """
    paths = side_paths(data_root, exchanges, massive_name, date)
    if paths is None:
        return None

    arrays = []
    for path, exch in zip(paths, exchanges):
        tc = time_col_for(exch)
        cols = read_columns(path, {tc: "int64"})
        if tc not in cols:
            raise ValueError(f"missing time column {tc} in {path}")
        arrays.append(cols[tc])

    ts = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
    if len(ts) > 1 and not np.all(ts[1:] >= ts[:-1]):
        ts = np.sort(ts, kind="mergesort")
    return ts


def count_side_rows(data_root: str, exchanges: list[str],
                    massive_name: str, date: str) -> int:
    paths = side_paths(data_root, exchanges, massive_name, date)
    return 0 if paths is None else sum(count_csv_rows(p) for p in paths)


def side_stamps(data_root: str, exchanges: list[str],
                massive_name: str, date: str) -> list[dict]:
    paths = side_paths(data_root, exchanges, massive_name, date) or []
    return [{"path": str(p.relative_to(get_root())),
             "mtime": int(p.stat().st_mtime),
             "size": int(p.stat().st_size)} for p in paths]


# ============================================================ caching

def out_columns(lead_label: str, lag_label: str) -> list[str]:
    return [f"lead_{lead_label}", f"lag_{lag_label}",
            "no_duplications", "found_counterpart"]


def combo_is_cached(path: Path, expected_rows: int,
                    lead_label: str, lag_label: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        header = csv_header(path)
    except Exception:
        return False
    if not all(c in header for c in out_columns(lead_label, lag_label)):
        return False
    return count_csv_rows(path) == expected_rows


def pair_meta_path(date_dir: Path, lead_label: str, lag_label: str) -> Path:
    return date_dir / f"{lead_label}_{lag_label}.meta.json"


# ============================================================ per-pair driver

def process_one_pair(data_root: str,
                     lead_eff: list[str], lag_eff: list[str],
                     lead_label: str, lag_label: str,
                     massive_name: str, date: str, rel_dir: Path) -> dict:
    """Return {(a, b, c): 'skip'|'computed'} for one (lead, lag, coin, day)."""
    date_dir = get_root() / INDEX_DIR / data_root / rel_dir
    expected_rows = count_side_rows(data_root, lead_eff, massive_name, date)

    stamps = {"lead": side_stamps(data_root, lead_eff, massive_name, date),
              "lag": side_stamps(data_root, lag_eff, massive_name, date)}
    meta = read_meta(pair_meta_path(date_dir, lead_label, lag_label))
    meta_ok = (meta is not None
               and meta.get("version") == CACHE_VERSION
               and meta.get("stamps") == stamps
               and meta.get("expected_rows") == expected_rows)

    statuses, need_any = {}, False
    for a, b, c in valid_combos():
        out_path = date_dir / f"{lead_label}_{lag_label}_{a}_{b}_{c}.csv"
        if meta_ok and combo_is_cached(out_path, expected_rows,
                                       lead_label, lag_label):
            statuses[(a, b, c)] = ("skip", out_path)
        else:
            statuses[(a, b, c)] = ("compute", out_path)
            need_any = True

    if not need_any:
        return {k: "skip" for k in statuses}

    lead_ts = load_side_ts(data_root, lead_eff, massive_name, date)
    lag_ts = load_side_ts(data_root, lag_eff, massive_name, date)
    if lead_ts is None or lag_ts is None:
        # a file disappeared between the presence check and this read
        return {k: "skip" for k in statuses}

    lead_idx_arr = np.arange(len(lead_ts), dtype=np.int64)
    lead_col, lag_col = f"lead_{lead_label}", f"lag_{lag_label}"

    results = {}
    for a, b, c in valid_combos():
        status, out_path = statuses[(a, b, c)]
        if status == "skip":
            results[(a, b, c)] = "skip"
            continue

        lag_idx, found = nearest_in_range(lead_ts, lag_ts, b, c, a)
        out = pd.DataFrame({lead_col: lead_idx_arr,
                            lag_col: lag_idx,
                            "no_duplications": compute_no_dup(lag_idx),
                            "found_counterpart": found})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        out.to_csv(tmp_path, index=False)
        tmp_path.replace(out_path)
        results[(a, b, c)] = "computed"

    write_meta(pair_meta_path(date_dir, lead_label, lag_label),
               {"version": CACHE_VERSION, "stamps": stamps,
                "expected_rows": expected_rows,
                "combos": [list(k) for k in sorted(statuses)]})
    return results


# ============================================================ driver

def _dates_across(dirs: list[Path]) -> set:
    if not dirs:
        return set()
    return set.intersection(*[set(f.stem for f in d.glob("*.csv")) for d in dirs])


def run_exchange_pair(data_root: str, lead_orig: str, lag_orig: str,
                      counters: dict) -> None:
    lead_eff = effective_exchanges(lead_orig, lag_orig)
    lag_eff = effective_exchanges(lag_orig, lead_orig)
    lead_label = side_label(lead_orig, lead_eff, lag_orig)
    lag_label = side_label(lag_orig, lag_eff, lead_orig)

    lead_maps, lag_maps = {}, {}
    for maps, effs in ((lead_maps, lead_eff), (lag_maps, lag_eff)):
        for exch in effs:
            m = coins_for_exchange(data_root, exch)
            if m is None:
                print(f"  [skip] {data_root}/{exch} missing")
                return
            maps[exch] = m

    common_coins = set.intersection(*(set(m) for m in lead_maps.values()))
    common_coins &= set.intersection(*(set(m) for m in lag_maps.values()))

    print(f"\n=== {data_root} | {lead_label} leads {lag_label} ===")
    combos = list(valid_combos())

    for massive_name in sorted(common_coins):
        lead_dirs = [lead_maps[e][massive_name] for e in lead_eff]
        lag_dirs = [lag_maps[e][massive_name] for e in lag_eff]
        common_dates = _dates_across(lead_dirs) & _dates_across(lag_dirs)

        coin_label = lead_dirs[0].name if len(lead_eff) == 1 else massive_name

        for date in sorted(common_dates):
            rel_dir = Path(lead_label) / coin_label / date
            try:
                results = process_one_pair(data_root, lead_eff, lag_eff,
                                           lead_label, lag_label,
                                           massive_name, date, rel_dir)
            except Exception as e:
                print(f"    [error] {coin_label}/{date}: {type(e).__name__}: {e}")
                continue

            n_ok = sum(1 for v in results.values() if v == "computed")
            n_skp = sum(1 for v in results.values() if v == "skip")
            counters["total"] += len(combos)
            counters["computed"] += n_ok
            counters["skipped"] += n_skp
            print(f"    {coin_label}/{date}  computed={n_ok:>2}  skipped={n_skp:>2}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="project root (default: $LEADLAG_ROOT or ~)")
    ap.add_argument("--roots", default=",".join(DATA_ROOTS),
                    help="comma-separated data roots to process")
    ap.add_argument("--pairs", default="all", choices=["all", "primary"])
    args = ap.parse_args()
    if args.root:
        set_root(args.root)

    combos = list(valid_combos())
    print(f"ROOT={get_root()}")
    print(f"Combos (a, b, c) with b > 2a: {len(combos)}")

    pairs = (PRIMARY_PAIRS if args.pairs == "primary" else list(ordered_pairs()))
    print(f"Ordered (lead, lag) pairs: {len(pairs)}")

    counters = {"total": 0, "computed": 0, "skipped": 0}
    for data_root in [r for r in args.roots.split(",") if r]:
        for lead_exchange, lag_exchange in pairs:
            run_exchange_pair(data_root, lead_exchange, lag_exchange, counters)

    print("\nDone.")
    print(f"  total:    {counters['total']:,}")
    print(f"  computed: {counters['computed']:,}")
    print(f"  skipped:  {counters['skipped']:,}")
    print(f"  output:   {get_root() / INDEX_DIR}/")


if __name__ == "__main__":
    main()
