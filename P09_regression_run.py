#!/usr/bin/env python3
"""
P09_regression_run.py

Run cross-venue lead-lag regressions using the index mapping written by P08.

Model:
    lag_ret_{a}[lag_idx]  ~  alpha + beta * lead_ret_{a}[lead_idx]

where ret_{a} is P07's VWAP-to-VWAP return.  The (lead, lag) pair is
discovered from the directory layout P08 writes:

    {ROOT}/regression_index/{data_root}/{lead_label}/{coin}/{date}/
        {lead_label}_{lag_label}_{a}_{b}_{c}.csv


Why the existing results file gets quarantined
----------------------------------------------
An earlier P07 computed ret_{a} against the raw trade print at the window
centre rather than that window's VWAP.  Fixing the formula does not fix the
numbers already on disk: `done_keys` here is keyed on
(coin, date, lead, lag, a, b, c) and says nothing about *how* the returns
that produced a row were computed, so every row written from the buggy
returns would be treated as done forever and never recomputed.

So every result row now carries a `pipeline_version` column.  On startup, a
`regressions.csv` whose version does not match RESULT_VERSION is renamed to
`regressions.csv.v{old}.bak` and the run starts clean.  Nothing is deleted.

What that version covers: it is bumped whenever P07's return definition or
this script's statistics change.  Move an old file back into place only if
you know it was produced by the current definitions.


What changed besides the version stamp
--------------------------------------
* Paths come from P06.get_root(), not the working directory -- this script
  used bare relative paths while P10 used a configurable root.
* The clean_data fallback path was `Path("clean_data") / data_root`, which
  for data_root="clean_data" resolved to `clean_data/clean_data/...` and for
  the open/close roots pointed at the wrong root entirely.  It is now
  `{ROOT}/{data_root}`.  That path is a last resort (only reachable for a
  multi-exchange side whose return file carries no timestamp at all), which
  is why it went unnoticed.
* The naming machinery -- effective_exchanges, side_label, label_to_effective,
  to_massive_name, from_massive_name, valid_label_pairs -- was a verbatim
  second copy of P08's.  It is now imported from P08.
* OLS is computed in closed form from running sums rather than through
  statsmodels: identical slope/intercept/t/p/R^2 for a simple regression,
  no design matrix, and P10 reuses the same function instead of carrying its
  own.  statsmodels is no longer a dependency.
* Results carry both `coin` (canonical X_BTCUSD) and `coin_label` (the lead
  exchange's own folder name), so P09 and P10 key their rows identically.


Row selection (per combo)
-------------------------
    - found_counterpart == True
    - no_duplications == True   (P08 flags, per lag row, the first lead row
      that claims it; without this, densely-clustered lead rows repeat the
      same lag y-value across observations, shrinking standard errors)
    - both ret_{a} values finite
    - indices in bounds

Note that consecutive lead rows are milliseconds apart, so their VWAP
windows still overlap almost completely and the t-statistics here remain
optimistic even after the no_duplications filter.  P10 reports the same
regressions on a thinned, non-overlapping subsample alongside these
full-sample numbers; treat `tstat` here as an upper bound on significance.


Memory design
-------------
Only a compact `done_keys` set is held (the 7 identifying columns, not the
stats), and new rows are appended to the output CSV rather than rewriting it.
The summary is built with a single chunked pass over the regressions file
(running sums per group) instead of an in-memory groupby.

Outputs, one per data_root:
    {ROOT}/regression_result/{data_root}/regressions.csv
    {ROOT}/regression_result/{data_root}/summary.csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
except Exception:
    sps = None

import P06_linear_VWAP as P06
import P07_linear_return as P07
import P08_regression_assign as P08
from P06_linear_VWAP import (
    CHUNK_ROWS, count_csv_rows, csv_header, get_root, read_columns, set_root,
    time_col_for,
)
from P08_regression_assign import (  # re-exported: P10 imports these from here
    DATA_ROOTS, EXCHANGES, MASSIVE_CONSTITUENTS, effective_exchanges,
    from_massive_name, label_to_effective, side_label, to_massive_name,
    valid_label_pairs,
)


# ============================================================ config

RESULT_DIR = "regression_result"
RETURN_DIR = P07.OUT_DIR

MIN_OBS = 30

# Bumped whenever P07's return definition or these statistics change.
#   1  ret referenced to the raw trade print  (BUG)
#   2  ret referenced to window VWAP, 5_trades gate, closed-form OLS
RESULT_VERSION = 2

RESULT_COLS = [
    "coin", "coin_label", "date", "lead", "lag", "a", "b", "c",
    "n_lead", "n_found", "n_used",
    "slope", "intercept", "tstat", "pvalue", "r2", "corr",
    "pipeline_version",
]

RESULT_KEY_COLS = ["coin", "date", "lead", "lag", "a", "b", "c"]

SUMMARY_CHUNK_SIZE = CHUNK_ROWS

_RET_RE = re.compile(r"^ret_(\d+)$")


# ============================================================ filename parsing

def parse_idx_filename(name: str, label_pairs: list[tuple[str, str]]):
    """Return (lead_label, lag_label, a, b, c) or None."""
    if not name.endswith(".csv"):
        return None
    stem = name[:-4]
    # longest labels first, since a label can itself contain '_'
    for lead_label, lag_label in sorted(
            label_pairs, key=lambda p: -(len(p[0]) + len(p[1]))):
        prefix = f"{lead_label}_{lag_label}_"
        if not stem.startswith(prefix):
            continue
        parts = stem[len(prefix):].split("_")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            continue
        a, b, c = (int(p) for p in parts)
        if b <= 2 * a:
            continue
        return lead_label, lag_label, a, b, c
    return None


# ============================================================ output paths

def out_paths(data_root: str):
    out_dir = get_root() / RESULT_DIR / data_root
    return out_dir, out_dir / "regressions.csv", out_dir / "summary.csv"


def quarantine_stale_results(out_file: Path, version: int = RESULT_VERSION,
                             siblings=("summary.csv",)) -> None:
    """Move aside a results CSV written under a different pipeline version.

    Replaces the old `migrate_legacy_schema`, which back-filled missing
    lead/lag columns in place -- right when the only change was a schema
    addition, exactly wrong when the *numbers* changed.  P10 reuses this.
    """
    if not out_file.exists() or out_file.stat().st_size == 0:
        return
    try:
        header = csv_header(out_file)
    except Exception:
        header = []

    old = None
    if "pipeline_version" in header:
        try:
            vals = pd.read_csv(out_file, usecols=["pipeline_version"])
            uniq = set(pd.unique(vals["pipeline_version"].dropna()))
            if uniq == {version}:
                return
            old = sorted(uniq)[0] if uniq else None
        except Exception:
            old = None

    tag = old if old is not None else "legacy"
    bak = out_file.with_suffix(out_file.suffix + f".v{tag}.bak")
    i = 1
    while bak.exists():
        bak = out_file.with_suffix(out_file.suffix + f".v{tag}.{i}.bak")
        i += 1
    out_file.replace(bak)
    for name in siblings:
        p = out_file.parent / name
        if p.exists():
            p.unlink()
    print(f"  [quarantine] {out_file.name} was written under pipeline_version="
          f"{tag}, current is {version}. Moved to {bak.name}; rebuilding "
          f"from scratch.")


def load_done_keys(out_file: Path) -> set[tuple]:
    """Only the 7 identifying columns of existing rows, as a set of tuples."""
    if not out_file.exists() or out_file.stat().st_size == 0:
        return set()
    df = pd.read_csv(out_file, usecols=RESULT_KEY_COLS, dtype={"date": str})
    return {(coin, date, lead, lag, int(a), int(b), int(c))
            for coin, date, lead, lag, a, b, c
            in df.itertuples(index=False, name=None)}


def append_results(out_file: Path, new_rows: list[dict],
                   file_has_header: bool, columns=None) -> bool:
    """Append rows; write the header only when the file doesn't have one."""
    if not new_rows:
        return file_has_header
    df = pd.DataFrame(new_rows, columns=columns or RESULT_COLS)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, mode="a", index=False, header=not file_has_header)
    return True


# ============================================================ return loading

def _default_col_filter(c: str) -> bool:
    return bool(_RET_RE.match(c))


def load_side_returns(data_root: str, exchanges: list[str],
                      massive_name: str, date: str,
                      col_filter: Callable[[str], bool] = _default_col_filter,
                      dtype: str = "float64") -> Optional[pd.DataFrame]:
    """Load the wanted columns for one side of a pair, in P08's row order.

    `col_filter` picks which columns of the linear_return file to keep; P10
    passes a wider filter (ret_/dt_/a_*_imb) so that both scripts share this
    one loader instead of each having its own.

    - Single exchange: file order is used as-is; no timestamp needed.
    - Multiple exchanges (massive_minus_*): a timestamp is needed to
      interleave the constituents the way P08 did.  Preference order:
      `ts_ms` in the return file (what P06/P07 always write) -> the
      exchange's native time column -> the timestamp read from the
      corresponding clean file for this data_root.
    """
    ret_root = get_root() / RETURN_DIR / data_root
    # NOTE: this used to be Path("clean_data") / data_root, i.e.
    # clean_data/clean_data/... -- see the module docstring.
    trades_root = get_root() / data_root

    need_ts = len(exchanges) > 1
    frames = []

    for exch in exchanges:
        coin_dir = from_massive_name(massive_name, exch)
        if coin_dir is None:
            return None
        rpath = ret_root / exch / coin_dir / f"{date}.csv"
        if not rpath.exists():
            return None

        try:
            header = csv_header(rpath)
        except Exception:
            return None
        want = [c for c in header if col_filter(c)]
        if not want:
            return None

        tc = time_col_for(exch)
        spec = {c: dtype for c in want}

        if "ts_ms" in header:
            spec["ts_ms"] = "int64"
            cols = read_columns(rpath, spec)
            cols["_ts"] = cols.pop("ts_ms")
        elif tc in header:
            spec[tc] = "int64"
            cols = read_columns(rpath, spec)
            cols["_ts"] = cols.pop(tc)
        elif need_ts:
            tpath = trades_root / exch / coin_dir / f"{date}.csv"
            if not tpath.exists():
                return None
            ts_cols = read_columns(tpath, {tc: "int64"})
            cols = read_columns(rpath, spec)
            if tc not in ts_cols or len(ts_cols[tc]) != len(next(iter(cols.values()))):
                return None
            cols["_ts"] = ts_cols[tc]
        else:
            cols = read_columns(rpath, spec)

        frames.append(pd.DataFrame(cols))

    if len(frames) == 1:
        return frames[0]

    out = pd.concat(frames, ignore_index=True)
    if "_ts" not in out.columns:
        return None
    if not out["_ts"].is_monotonic_increasing:
        out = out.sort_values("_ts", kind="mergesort").reset_index(drop=True)
    return out


# ============================================================ regression

def ols_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple OLS y ~ a + b x in closed form from running sums.

    Identical to statsmodels' OLS with a constant for this one-regressor
    case, with no design matrix allocated.  P10 imports this rather than
    defining a second copy.
    """
    n = len(x)
    nan = dict(slope=np.nan, intercept=np.nan, tstat=np.nan,
               pvalue=np.nan, r2=np.nan, corr=np.nan)
    if n < 3:
        return nan
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dx, dy = x - x.mean(), y - y.mean()
    sxx, syy, sxy = float(dx @ dx), float(dy @ dy), float(dx @ dy)
    if sxx <= 0 or syy <= 0:
        return nan
    slope = sxy / sxx
    rss = max(syy - slope * sxy, 0.0)
    se = math.sqrt(rss / (n - 2) / sxx) if n > 2 and sxx > 0 else np.nan
    t = slope / se if se and se > 0 and np.isfinite(se) else np.nan
    if not np.isfinite(t):
        p = np.nan
    elif sps is not None:
        p = float(2 * sps.t.sf(abs(t), n - 2))
    else:
        # normal approximation when scipy is unavailable
        p = float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))
    return dict(slope=slope, intercept=float(y.mean() - slope * x.mean()),
                tstat=t, pvalue=p, r2=1.0 - rss / syy,
                corr=sxy / math.sqrt(sxx * syy))


def select_pairs(lead_ret: np.ndarray, lag_ret: np.ndarray,
                 lead_idx: np.ndarray, lag_idx: np.ndarray,
                 found: np.ndarray, no_dup: np.ndarray):
    """Apply P08's flags and bounds/NaN filtering.

    Returns (x, y, li, gi, n_lead, n_found) with li/gi the surviving lead and
    lag row indices, so callers that need more per-observation data (P10's
    thinning and imbalance regression) can index back into their own arrays.
    """
    n_lead = len(lead_idx)
    keep = found & no_dup & (lag_idx >= 0)
    li, gi = lead_idx[keep], lag_idx[keep]
    n_found = int(keep.sum())

    inb = (li >= 0) & (li < len(lead_ret)) & (gi < len(lag_ret))
    li, gi = li[inb], gi[inb]

    x, y = lead_ret[li], lag_ret[gi]
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok], li[ok], gi[ok], n_lead, n_found


def run_one_regression(lead_ret, lag_ret, lead_idx, lag_idx, found, no_dup):
    x, y, _li, _gi, n_lead, n_found = select_pairs(
        lead_ret, lag_ret, lead_idx, lag_idx, found, no_dup)
    n_used = int(len(x))
    if n_found == 0:
        return None
    base = {"n_lead": n_lead, "n_found": n_found, "n_used": n_used}
    if n_used < MIN_OBS:
        return {**base, "slope": np.nan, "intercept": np.nan, "tstat": np.nan,
                "pvalue": np.nan, "r2": np.nan, "corr": np.nan}
    return {**base, **ols_stats(x, y)}


# ============================================================ per (lead, lag, coin, date)

def read_index(path: Path, lead_label: str, lag_label: str) -> Optional[dict]:
    cols = read_columns(path, {f"lead_{lead_label}": "int64",
                               f"lag_{lag_label}": "int64",
                               "no_duplications": "bool",
                               "found_counterpart": "bool"})
    if len(cols) < 4:
        return None
    return {"lead": cols[f"lead_{lead_label}"], "lag": cols[f"lag_{lag_label}"],
            "nodup": cols["no_duplications"], "found": cols["found_counterpart"]}


def process_one_lead_lag_date(lead_label, lag_label, lead_df, lag_df,
                              massive_name, coin_label, date, idx_dir,
                              done_keys, label_pairs) -> list[dict]:
    """Newly-computed result rows for one (lead, lag, coin, date)."""
    new_rows = []
    for idx_file in sorted(idx_dir.glob(f"{lead_label}_{lag_label}_*.csv")):
        parsed = parse_idx_filename(idx_file.name, label_pairs)
        if parsed is None:
            continue
        p_lead, p_lag, a, b, c = parsed
        if p_lead != lead_label or p_lag != lag_label:
            continue

        key = (massive_name, date, lead_label, lag_label, a, b, c)
        if key in done_keys:
            continue

        col = f"ret_{a}"
        if col not in lead_df.columns or col not in lag_df.columns:
            continue

        idx = read_index(idx_file, lead_label, lag_label)
        if idx is None:
            raise ValueError(
                f"{idx_file} is missing one of the four expected columns; "
                "regenerate it with the current P08 before rerunning P09.")

        stats = run_one_regression(lead_df[col].to_numpy(dtype=float),
                                   lag_df[col].to_numpy(dtype=float),
                                   idx["lead"], idx["lag"],
                                   idx["found"], idx["nodup"])
        if stats is None:
            continue

        new_rows.append({"coin": massive_name, "coin_label": coin_label,
                         "date": date, "lead": lead_label, "lag": lag_label,
                         "a": a, "b": b, "c": c,
                         "pipeline_version": RESULT_VERSION, **stats})
        done_keys.add(key)

    return new_rows


# ============================================================ per data_root

def run_one_data_root(data_root: str, label_pairs, done_keys,
                      out_file: Path, file_has_header: bool) -> bool:
    index_root = get_root() / P08.INDEX_DIR / data_root
    if not index_root.exists():
        print(f"[skip] {index_root} missing")
        return file_has_header

    for lead_dir in sorted(d for d in index_root.iterdir() if d.is_dir()):
        lead_label = lead_dir.name
        lead_eff = label_to_effective(lead_label)
        if lead_eff is None:
            continue

        lag_candidates = [(gl, label_to_effective(gl))
                          for ll, gl in label_pairs if ll == lead_label]
        lag_candidates = [(gl, eff) for gl, eff in lag_candidates if eff]

        print(f"\n=== {data_root} | lead={lead_label} "
              f"({len(lag_candidates)} lag labels) ===")

        for coin_dir in sorted(d for d in lead_dir.iterdir() if d.is_dir()):
            coin_label = coin_dir.name
            massive_name = (coin_label if lead_label.startswith("massive_minus_")
                            else to_massive_name(coin_label, lead_label))
            if massive_name is None or not (massive_name.startswith("X_")
                                            and massive_name.endswith("USD")):
                continue

            for date_dir in sorted(d for d in coin_dir.iterdir() if d.is_dir()):
                date = date_dir.name
                lead_df = load_side_returns(data_root, lead_eff, massive_name, date)
                if lead_df is None:
                    continue

                for lag_label, lag_eff in lag_candidates:
                    lag_df = load_side_returns(data_root, lag_eff,
                                               massive_name, date)
                    if lag_df is None:
                        continue

                    new_rows = process_one_lead_lag_date(
                        lead_label, lag_label, lead_df, lag_df,
                        massive_name, coin_label, date, date_dir,
                        done_keys, label_pairs)
                    del lag_df

                    if new_rows:
                        file_has_header = append_results(
                            out_file, new_rows, file_has_header)
                        print(f"  {coin_label}/{date}  "
                              f"{lead_label} -> {lag_label}  +{len(new_rows)}")
                del lead_df

    return file_has_header


# ============================================================ summary (streamed)

def build_summary_streaming(out_file: Path, summary_file: Path,
                            chunksize: int = SUMMARY_CHUNK_SIZE) -> pd.DataFrame:
    """Same grouped stats an in-memory groupby would give, via one chunked
    pass: peak memory is bounded by chunk size plus group count."""
    if not out_file.exists() or out_file.stat().st_size == 0:
        empty = pd.DataFrame()
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(summary_file, index=False)
        return empty

    usecols = ["coin", "lead", "lag", "a", "b", "c",
               "n_used", "slope", "tstat", "r2", "pvalue"]
    acc: dict[tuple, dict] = {}

    for chunk in pd.read_csv(out_file, usecols=usecols, chunksize=chunksize):
        chunk = chunk.dropna(subset=["slope"])
        if chunk.empty:
            continue
        for key, g in chunk.groupby(["coin", "lead", "lag", "a", "b", "c"],
                                    sort=False):
            entry = acc.setdefault(key, {
                "n_days": 0, "n_used_sum": 0, "slope_sum": 0.0,
                "slope_sumsq": 0.0, "tstat_sum": 0.0, "r2_sum": 0.0,
                "sig_sum": 0})
            entry["n_days"] += len(g)
            entry["n_used_sum"] += int(g["n_used"].sum())
            entry["slope_sum"] += float(g["slope"].sum())
            entry["slope_sumsq"] += float((g["slope"] ** 2).sum())
            entry["tstat_sum"] += float(g["tstat"].sum())
            entry["r2_sum"] += float(g["r2"].sum())
            entry["sig_sum"] += int((g["pvalue"] < 0.05).sum())

    rows = []
    for (coin, lead, lag, a, b, c), v in acc.items():
        n = v["n_days"]
        mean = v["slope_sum"] / n
        if n > 1:
            var = (v["slope_sumsq"] - n * mean * mean) / (n - 1)
            std = float(np.sqrt(var)) if var > 0 else 0.0
        else:
            std = np.nan
        rows.append({"coin": coin, "lead": lead, "lag": lag,
                     "a": a, "b": b, "c": c, "n_days": n,
                     "n_used_sum": v["n_used_sum"], "slope_mean": mean,
                     "slope_std": std, "tstat_mean": v["tstat_sum"] / n,
                     "r2_mean": v["r2_sum"] / n,
                     "frac_significant": v["sig_sum"] / n})

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["coin", "lead", "lag", "a", "b", "c"]).reset_index(drop=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_file, index=False)
    return summary


# ============================================================ main

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

    label_pairs = valid_label_pairs(args.pairs)
    print(f"ROOT={get_root()}")
    print(f"Known (lead, lag) label pairs: {len(label_pairs)}")

    for data_root in [r for r in args.roots.split(",") if r]:
        out_dir, out_file, summary_file = out_paths(data_root)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n########## data_root = {data_root} ##########")
        quarantine_stale_results(out_file)

        done_keys = load_done_keys(out_file)
        print(f"Loaded {len(done_keys):,} existing regression keys")

        file_has_header = out_file.exists() and out_file.stat().st_size > 0
        file_has_header = run_one_data_root(
            data_root, label_pairs, done_keys, out_file, file_has_header)

        n_rows = count_csv_rows(out_file) if file_has_header else 0
        print(f"\nSaved -> {out_file}   ({n_rows:,} rows)")

        summary = build_summary_streaming(out_file, summary_file)
        print(f"Saved -> {summary_file}   ({len(summary):,} rows)")

        if not summary.empty:
            print("\n=== Summary preview (top 20) ===")
            print(summary.head(20).to_string())


if __name__ == "__main__":
    main()
