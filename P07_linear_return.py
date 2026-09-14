#!/usr/bin/env python3
"""
P07_linear_return.py

Read the augmented files from {ROOT}/linear_vwap_construction/ and produce
parallel copies under {ROOT}/linear_return/ with derived return columns for
every interval `a` present in the source file.

Intervals are discovered dynamically from the source header (any column named
`a_{N}_possible` defines an interval a = N).


The return definition
---------------------
For row i, let j = a_{a}_next[i] -- the first *possible* row at least a ms
later, whose window W2 therefore does not overlap row i's window W1.  Then

    ret_{a}[i] = VWAP(W2) / VWAP(W1) - 1
               = a_{a}_vwap[j] / a_{a}_vwap[i] - 1

Both legs are window VWAPs.  An earlier version of this script used the raw
trade print at the window centre as the reference price (`price[i]`) instead
of `a_{a}_vwap[i]`, which mixed a single noisy print with a smoothed target
and injected the full bid-ask bounce of that one print into every return.

That bug is fixed in the formula, but fixing the formula is not enough on its
own -- see "Invalidating the buggy outputs" below.


Window validity
---------------
A return is produced only where BOTH windows are usable:

    a_{a}_possible[i] and a_{a}_5_trades[i]
    a_{a}_possible[j] and a_{a}_5_trades[j]

The `5_trades` gate (>= P06.MIN_TRADES_IN_WINDOW prints in the window) is
applied here for the first time.  P06 has always computed the flag, and P10
has always enforced it, but this script did not -- so `ret_{a}` used to
include windows built from one or two prints, whose "VWAP" is just that print
and carries none of the averaging the VWAP construction is for.  Set
REQUIRE_5_TRADES = False to recover the old, ungated behaviour.


Columns added per discovered `a` (force-zero convention throughout)
-------------------------------------------------------------------
    ret_{a}              = vwap[j] / vwap[i] - 1
    logret_{a}           = log(vwap[j] / vwap[i]), forced to 0 when ret == 0
    ret_per_dt_{a}       = ret / dt
    log_ret_per_dt_{a}   = sign(r) * log(|r|) for r = ret/dt, 0 when r == 0
    logret_per_dt_{a}    = logret / dt, forced to 0 when ret == 0
    excess_dt_{a}        = dt - a
    dt_{a}               = ts[j] - ts[i]

(The module docstring used to describe `logret_{a}` as sign(ret)*log(|ret|).
It was never that; it is and always was the ordinary log return
log(1 + ret).  Only `log_ret_per_dt_{a}` uses the signed-log transform.)


Invalidating the buggy outputs
------------------------------
The old cache check asked only whether `ret_{a}` was non-NaN on two sample
rows.  A `linear_return/` file written by the price-referenced version passes
that check perfectly -- same column names, same NaN pattern, different
numbers -- so a rerun after the formula fix would report ACTION=skip and keep
the wrong values indefinitely.  The consequence of the bug outlives the bug.

So caching is now stamped, not sniffed: each destination gets a sidecar
`{name}.csv.meta.json` recording CACHE_VERSION, the source file's mtime and
size, the row count, the interval list and the REQUIRE_5_TRADES setting.  The
version was bumped past every value the buggy code could have written, so
every pre-existing `linear_return/` file is now treated as stale and rebuilt
on the next run.  Delete nothing by hand; just rerun.

Downstream, P09 and P10 carry the same stamp into their result files -- see
their headers -- so their caches cannot serve you regressions computed from
the old returns either.


Memory design
-------------
Source files here are P06's output: every original column (including
string-heavy ones) plus 36 a_* columns.  Loading one whole is what caused OOM
kills.  As in P06 the work is split into a light pass (ts_ms and the specific
a_* columns needed, read in chunks) and a chunked write pass that re-reads
the source only to carry the original columns through.

No sortedness assumption is needed: `a_{a}_next` values are row-positional
indices into this same file, and both passes read it front to back.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import P06_linear_VWAP as P06
from P06_linear_VWAP import (
    CHUNK_ROWS, FLOAT_FMT, csv_header, get_root, read_columns, read_meta,
    set_root, src_stamp, stamp_matches, write_meta,
)


# ============================================================ config

SRC_DIR = P06.CONSTRUCT_DIR
OUT_DIR = "linear_return"

RET_SUFFIXES = ("ret", "logret", "ret_per_dt",
                "log_ret_per_dt", "logret_per_dt",
                "excess_dt", "dt")

# Require >= P06.MIN_TRADES_IN_WINDOW trades in BOTH windows (matches P10).
REQUIRE_5_TRADES = True

# Bump to invalidate every linear_return/ file.
#   1  ret referenced to the raw trade print  (BUG)
#   2  ret referenced to a_{a}_vwap
#   3  5_trades gate + meta-json caching
CACHE_VERSION = 3

_A_POSSIBLE_RE = re.compile(r"^a_(\d+)_possible$")

WRITE_CHUNK_SIZE = CHUNK_ROWS


# ============================================================ discovery

def discover_source_intervals(columns) -> list[int]:
    """Sorted list of `a` from the source's a_{a}_possible columns."""
    found = set()
    for c in columns:
        m = _A_POSSIBLE_RE.match(c)
        if m:
            found.add(int(m.group(1)))
    return sorted(found)


def usable_intervals(header, a_list) -> tuple[list[int], list[int]]:
    """Split `a_list` into (usable, unusable) given what the source carries.

    An interval needs a_{a}_vwap and a_{a}_next; the possible / 5_trades
    flags are additionally required when REQUIRE_5_TRADES is on.
    """
    need = ["vwap", "next"] + (["possible", "5_trades"] if REQUIRE_5_TRADES else [])
    usable = [a for a in a_list
              if all(f"a_{a}_{s}" in header for s in need)]
    return usable, sorted(set(a_list) - set(usable))


# ============================================================ light loader

def load_light_for_intervals(src: Path, header, a_list):
    """Read ts_ms plus the per-interval flag columns for `a_list`.

    `a_{a}_next` is read as float64, not float32: float32 is only exact for
    row indices below 2**24 (~16.7M), and the oversized files this pipeline
    exists to survive can exceed that.

    Returns (ts_ms, per_interval_dict, skipped, n_rows).
    """
    usable, skipped = usable_intervals(header, a_list)

    spec: dict[str, str] = {"ts_ms": "int64"}
    for a in usable:
        spec[f"a_{a}_vwap"] = "float64"
        spec[f"a_{a}_next"] = "float64"
        if REQUIRE_5_TRADES:
            spec[f"a_{a}_possible"] = "bool"
            spec[f"a_{a}_5_trades"] = "bool"

    cols = read_columns(src, spec)
    if "ts_ms" not in cols:
        raise ValueError(f"{src}: missing ts_ms column")
    ts_ms = cols["ts_ms"]

    per_a: dict[int, dict[str, np.ndarray]] = {}
    for a in usable:
        entry = {"vwap": cols[f"a_{a}_vwap"], "next": cols[f"a_{a}_next"]}
        if REQUIRE_5_TRADES:
            entry["possible"] = cols[f"a_{a}_possible"]
            entry["five"] = cols[f"a_{a}_5_trades"]
        per_a[a] = entry

    return ts_ms, per_a, skipped, len(ts_ms)


# ============================================================ core computation

def signed_log(x: np.ndarray) -> np.ndarray:
    """sign(x) * log(|x|).  NaN at x == 0; the caller forces that to 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sign(x) * np.log(np.abs(x))


def compute_return_arrays(ts_ms: np.ndarray,
                          per_a: dict[int, dict[str, np.ndarray]]
                          ) -> dict[str, np.ndarray]:
    """Compute every {suffix}_{a} column as plain numpy arrays."""
    n = len(ts_ms)
    out: dict[str, np.ndarray] = {}
    if not per_a:
        return out
    if n == 0:
        # Keep the schema identical for empty files: a zero-row day still gets
        # every ret_* column, so P09/P10 see it as a valid (if empty) side
        # rather than a file with no returns at all.
        for a in sorted(per_a):
            for s in RET_SUFFIXES:
                out[f"{s}_{a}"] = np.empty(0, dtype=float)
        return out

    for a in sorted(per_a):
        entry = per_a[a]
        vwap = np.asarray(entry["vwap"], dtype=float)
        next_arr = np.asarray(entry["next"], dtype=float)

        has_next = ~np.isnan(next_arr)
        j = np.where(has_next, np.nan_to_num(next_arr, nan=0.0), 0).astype(np.int64)
        has_next &= (j >= 0) & (j < n)
        j = np.where(has_next, j, 0)

        # ---- window validity, both legs ----
        valid = has_next & ~np.isnan(vwap) & ~np.isnan(vwap[j])
        if REQUIRE_5_TRADES:
            possible = np.asarray(entry["possible"], dtype=bool)
            five = np.asarray(entry["five"], dtype=bool)
            valid &= possible & five & possible[j] & five[j]

        p1 = np.where(valid, vwap, np.nan)
        p2 = np.where(valid, vwap[j], np.nan)
        t1 = ts_ms.astype(float)
        dt = np.where(valid, ts_ms[j].astype(float) - t1, np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            good = valid & (p1 > 0) & (p2 > 0)
            ret = np.where(good, p2 / p1 - 1.0, np.nan)
            logret = np.where(good, np.log(np.where(good, p2 / p1, 1.0)), np.nan)

            ret_per_dt = np.where(dt > 0, ret / dt, np.nan)
            log_ret_per_dt = signed_log(ret_per_dt)
            logret_per_dt = np.where(dt > 0, logret / dt, np.nan)
            excess_dt = np.where(~np.isnan(dt), dt - a, np.nan)

        # ---- force 0 (never NaN) where the linear return is exactly 0 ----
        zero_ret = (ret == 0)
        zero_retdt = (ret_per_dt == 0)
        logret = np.where(zero_ret, 0.0, logret)
        logret_per_dt = np.where(zero_ret, 0.0, logret_per_dt)
        log_ret_per_dt = np.where(zero_retdt, 0.0, log_ret_per_dt)

        out[f"ret_{a}"] = ret
        out[f"logret_{a}"] = logret
        out[f"ret_per_dt_{a}"] = ret_per_dt
        out[f"log_ret_per_dt_{a}"] = log_ret_per_dt
        out[f"logret_per_dt_{a}"] = logret_per_dt
        out[f"excess_dt_{a}"] = excess_dt
        out[f"dt_{a}"] = dt

    return out


# ============================================================ chunked writer

def write_chunked(src: Path, dst: Path,
                  arrays: dict[str, np.ndarray], n_rows: int) -> None:
    """Stream src in CHUNK_ROWS-row chunks, attach the precomputed return
    arrays by row offset, and write dst atomically."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")

    offset = 0
    first_chunk = True
    try:
        with open(tmp_path, "w", newline="") as f:
            for chunk in pd.read_csv(src, chunksize=CHUNK_ROWS):
                m = len(chunk)
                for col, arr in arrays.items():
                    chunk[col] = arr[offset:offset + m]
                chunk.to_csv(f, index=False, header=first_chunk,
                             float_format=FLOAT_FMT)
                first_chunk = False
                offset += m
                del chunk
        if offset != n_rows:
            raise ValueError(
                f"{src}: row count changed between passes "
                f"({offset} on write pass vs {n_rows} on light pass) -- "
                "refusing to write a possibly-misaligned output")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.replace(dst)


# ============================================================ driver

def _fmt(lst) -> str:
    return "[" + ",".join(str(x) for x in sorted(lst)) + "]"


def is_cached(dst: Path, src: Path, a_values) -> bool:
    if not (dst.exists() and dst.stat().st_size > 0):
        return False
    meta = read_meta(dst)
    if not stamp_matches(meta, src, CACHE_VERSION):
        return False
    if meta.get("require_5_trades") != REQUIRE_5_TRADES:
        return False
    return set(a_values).issubset(set(meta.get("intervals", [])))


def process_one_file(src: Path, rel: Path) -> int:
    """Returns row count, or -1 if skipped / failed."""
    dst = get_root() / OUT_DIR / rel

    try:
        src_header = csv_header(src)
    except Exception as e:
        print(f"      [skip] cannot read source header ({e})")
        return -1

    src_intervals = discover_source_intervals(src_header)
    if not src_intervals:
        print("      [skip] no a_{a}_possible columns in source")
        return -1

    usable, unusable = usable_intervals(src_header, src_intervals)
    if unusable:
        print(f"      [warn] source lacks the required a_* columns for "
              f"{_fmt(unusable)}; those intervals get no ret_* columns. "
              f"Rerun P06 (its cache version was bumped) to add them.")
    if not usable:
        print("      [skip] no usable intervals in source")
        return -1

    if is_cached(dst, src, usable):
        print(f"      SRC={_fmt(src_intervals)}  USE={_fmt(usable)}  ACTION=skip")
        return -1

    action = "recompute" if dst.exists() else "fresh"
    print(f"      SRC={_fmt(src_intervals)}  USE={_fmt(usable)}  ACTION={action}")

    try:
        ts_ms, per_a, skipped, n_rows = load_light_for_intervals(
            src, src_header, usable)
    except Exception as e:
        print(f"      [skip] cannot read source columns ({e})")
        return -1

    arrays = compute_return_arrays(ts_ms, per_a)
    del per_a

    try:
        write_chunked(src, dst, arrays, n_rows)
    except Exception as e:
        print(f"      [skip] write failed ({e})")
        return -1

    write_meta(dst, {"version": CACHE_VERSION,
                     "n_rows": n_rows,
                     "intervals": sorted(usable),
                     "suffixes": list(RET_SUFFIXES),
                     "require_5_trades": REQUIRE_5_TRADES,
                     "ret_reference": "window_vwap",
                     **src_stamp(src)})
    return n_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="project root (default: $LEADLAG_ROOT or ~)")
    args = ap.parse_args()
    if args.root:
        set_root(args.root)

    src_root = get_root() / SRC_DIR
    if not src_root.exists():
        print(f"Source {src_root} missing -- nothing to do.")
        return

    out_root = get_root() / OUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    counters = {"total": 0, "skipped": 0, "computed": 0}
    for f in sorted(src_root.rglob("*.csv")):
        rel = f.relative_to(src_root)
        print(f"  [{counters['total'] + 1:>4}] {rel}")

        n = process_one_file(f, rel)
        counters["total"] += 1
        if n == -1:
            counters["skipped"] += 1
            print("      -> SKIP")
        else:
            counters["computed"] += 1
            print(f"      -> OK ({n:,} rows)")

    print("\nDone.")
    print(f"  total files:    {counters['total']}")
    print(f"  computed:       {counters['computed']}")
    print(f"  skipped:        {counters['skipped']}")
    print(f"  output:         {out_root}/")


if __name__ == "__main__":
    main()
