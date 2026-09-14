from __future__ import annotations
"""
P09_regression_run.py

Run cross-venue lead-lag regressions using the index mapping written by
P08_regression_assign.py.

Model (generic):
    lag_return_{a}[lag_idx]  ~  alpha + beta * lead_return_{a}[lead_idx]

The (lead, lag) pair is discovered from the subdirectory layout written by
P08:

    regression_index/{data_root}/{lead_label}/{coin}/{date}/
        {lead_label}_{lag_label}_{a}_{b}_{c}.csv

P08 may substitute `massive` with the remaining three constituents of
{Exchange1, Exchange2, Exchange6, Exchange23} whenever the counterpart is
itself one of those constituents.  In that case the label reads
`massive_minus_<counterpart>` and P08 concatenated the remaining
constituents' CSVs, sorted by timestamp.  This script reproduces that
concatenation so the row indices P08 wrote line up with the return arrays.

Return-file handling
--------------------
P06/P07 rename the source timestamp column to `ts_ms` in EVERY file they
write, regardless of exchange (binance's `transact_time_ms` and the other
exchanges' `participant_ts_ms` both become `ts_ms`). So that's checked
first. Older/foreign return files that still carry the original column name
are supported via a `tc` fallback, and files with neither are supported via
a last-resort read of the corresponding clean_data CSV:

  * single-exchange side     -> no timestamp needed, rows are used in file
                                order (matches what P08 saw)
  * multi-exchange side      -> timestamps are needed to interleave the
                                constituents. Preference order: `ts_ms` in
                                the return file itself (the common case,
                                given current P06/P07) -> the exchange's own
                                time column name, if present -> the
                                timestamp column read from the corresponding
                                clean_data CSV (same row order, produced by
                                the same upstream pass).

Row selection (per combo):
    - keep only rows with found_counterpart == True
    - keep only rows with no_duplications == True (P08 flags, per lag row,
      the first lead row that claims it; this drops the pseudo-replicated
      extra lead rows that would otherwise repeat the same lag y-value
      across multiple regression observations and artificially shrink
      standard errors / inflate significance)
    - drop rows where either ret_{a} is NaN
    - drop rows whose index is out of bounds

Memory design
-------------
Earlier versions of this script accumulated every regression result for an
entire data_root run in one in-memory DataFrame, and rewrote the *entire*
output CSV from scratch every time a handful of new rows were added. Across
every (coin, date, lead, lag, a, b, c) combination that's potentially
millions of rows held in memory simultaneously, rewritten repeatedly -- the
same class of problem the light-pass fixes in P06/P07/P08 addressed for
wide input rows, just showing up here on the output side instead.

This version keeps only a compact `done_keys` set in memory (the 7
identifying columns, not the 8 stat columns) for O(1) already-done checks,
and appends new rows to the output CSV as they're computed rather than
rewriting the whole file. A pre-existing file with the old (pre-
substitution) schema -- missing `lead`/`lag` columns -- is migrated once,
up front, in a single bounded pass; every write after that is append-only.

The final summary is built with a single chunked pass over the regressions
file (running sum / sum-of-squares per group) instead of loading the whole
table for a `groupby`, so peak memory is bounded by chunk size and the
number of distinct groups, not the total row count.

Outputs (incremental, one file per data_root):
    regression_result/{data_root}/regressions.csv
    regression_result/{data_root}/summary.csv
"""


from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------- config ----------

DATA_ROOTS = ["clean_data", "clean_open_data", "clean_close_data"]

EXCHANGES = [
    "binance",
    "massive",
    "Exchange1",
    "Exchange2",
    "Exchange6",
    "Exchange23",
]

MASSIVE_CONSTITUENTS = ["Exchange1", "Exchange2", "Exchange6", "Exchange23"]

MIN_OBS = 30

RESULT_COLS = [
    "coin", "date", "lead", "lag", "a", "b", "c",
    "n_lead", "n_found", "n_used",
    "slope", "intercept", "tstat", "pvalue", "r2",
]

# Identifying columns only -- used for the light-read dedup key set, so we
# never have to pull the 8 stat columns into memory just to know what's
# already been computed.
RESULT_KEY_COLS = ["coin", "date", "lead", "lag", "a", "b", "c"]

# Rows per chunk when streaming the regressions file for the summary pass.
SUMMARY_CHUNK_SIZE = 200_000


# ---------- label machinery (mirrors P08) ----------

def effective_exchanges(exchange: str, counterpart: str) -> list[str]:
    if exchange == "massive" and counterpart in MASSIVE_CONSTITUENTS:
        return [e for e in MASSIVE_CONSTITUENTS if e != counterpart]
    return [exchange]


def side_label(original: str, effective: list[str], counterpart: str) -> str:
    if len(effective) == 1 and effective[0] == original:
        return original
    return f"{original}_minus_{counterpart}"


def label_to_effective(label: str) -> Optional[list[str]]:
    if label in EXCHANGES:
        return [label]
    if label.startswith("massive_minus_"):
        counter = label[len("massive_minus_"):]
        if counter in MASSIVE_CONSTITUENTS:
            return [e for e in MASSIVE_CONSTITUENTS if e != counter]
    return None


def valid_label_pairs() -> list[tuple[str, str]]:
    pairs, seen = [], set()
    for lead_orig in EXCHANGES:
        for lag_orig in EXCHANGES:
            if lead_orig == lag_orig:
                continue
            lead_eff = effective_exchanges(lead_orig, lag_orig)
            lag_eff = effective_exchanges(lag_orig, lead_orig)
            ll = side_label(lead_orig, lead_eff, lag_orig)
            gl = side_label(lag_orig, lag_eff, lead_orig)
            if (ll, gl) not in seen:
                seen.add((ll, gl))
                pairs.append((ll, gl))
    return pairs


# ---------- coin-name mapping (mirrors P08) ----------

def to_massive_name(coin_dir: str, exchange: str) -> Optional[str]:
    if exchange == "binance":
        if not coin_dir.endswith("USDT"):
            return None
        return f"X_{coin_dir[:-4]}USD"
    if exchange == "massive":
        if coin_dir.startswith("X_") and coin_dir.endswith("USD"):
            return coin_dir
        return None
    if exchange.startswith("Exchange"):
        parts = coin_dir.split("_", 1)
        if len(parts) != 2:
            return None
        return f"X_{parts[1]}"
    return None


def from_massive_name(massive_name: str, exchange: str) -> Optional[str]:
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


def time_col_for(exchange: str) -> str:
    return "transact_time_ms" if exchange == "binance" else "participant_ts_ms"


# ---------- filename parsing ----------

def parse_idx_filename(name: str,
                       label_pairs: list[tuple[str, str]]):
    """Return (lead_label, lag_label, a, b, c) or None."""
    if not name.endswith(".csv"):
        return None
    stem = name[:-4]
    # try longer prefixes first, in case a label contains '_'
    for lead_label, lag_label in sorted(
            label_pairs, key=lambda p: -(len(p[0]) + len(p[1]))):
        prefix = f"{lead_label}_{lag_label}_"
        if not stem.startswith(prefix):
            continue
        rest = stem[len(prefix):]
        parts = rest.split("_")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            continue
        a, b, c = (int(p) for p in parts)
        if b <= 2 * a:
            continue
        return lead_label, lag_label, a, b, c
    return None


# ---------- output paths ----------

def _out_paths(data_root: str):
    out_dir = Path("regression_result") / data_root
    return out_dir, out_dir / "regressions.csv", out_dir / "summary.csv"


def count_csv_rows(path: Path) -> int:
    """Fast byte-mode line count minus header."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def migrate_legacy_schema(out_file: Path) -> None:
    """One-time migration for regressions.csv files written before the
    massive-substitution feature existed, which have no `lead`/`lag`
    columns at all (every row implicitly meant lead=binance,
    lag=massive).

    This is a single bounded full read+rewrite, done once at startup --
    not a per-iteration cost. Everything after this point (the dedup key
    set and all new writes) relies on `lead`/`lag` being present, so this
    has to run before those.
    """
    if not out_file.exists() or out_file.stat().st_size == 0:
        return
    header = pd.read_csv(out_file, nrows=0).columns.tolist()
    if "lead" in header and "lag" in header:
        return

    print(f"  [migrate] {out_file} missing lead/lag columns -- backfilling "
          f"as binance -> massive (legacy pre-substitution schema)")
    df = pd.read_csv(out_file)
    df["date"] = df["date"].astype(str)
    for col, default in (("lead", "binance"), ("lag", "massive")):
        if col not in df.columns:
            df[col] = default
    df = df[RESULT_COLS]

    tmp = out_file.with_suffix(out_file.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_file)


def load_done_keys(out_file: Path) -> set[tuple]:
    """Load only the 7 identifying columns of existing regression rows, as
    a set of tuples, for an O(1) already-done check -- never the full
    15-column stats table, which is what made the old already_done() scan
    (and the DataFrame it scanned) grow without bound over a long run.
    """
    if not out_file.exists() or out_file.stat().st_size == 0:
        return set()
    df = pd.read_csv(out_file, usecols=RESULT_KEY_COLS, dtype={"date": str})
    return {
        (coin, date, lead, lag, int(a), int(b), int(c))
        for coin, date, lead, lag, a, b, c
        in df.itertuples(index=False, name=None)
    }


def append_results(out_file: Path, new_rows: list[dict],
                   file_has_header: bool) -> bool:
    """Append new rows to out_file. Writes a header only the first time
    (i.e. when the file doesn't already have one) -- every call after that
    just appends, so the file is never re-read or rewritten in full."""
    if not new_rows:
        return file_has_header
    df = pd.DataFrame(new_rows, columns=RESULT_COLS)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, mode="a", index=False, header=not file_has_header)
    return True


# ---------- return loading ----------

def _read_ret_header(path: Path) -> Optional[list[str]]:
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return None
    return [c for c in header if c.startswith("ret_")]


def load_side_returns(data_root: str,
                      exchanges: list[str],
                      massive_name: str,
                      date: str):
    """Load the return frame for one side.

    Returns a pandas DataFrame with the ret_* columns (and, when needed,
    a `_ts` column used only for sorting), or None if any input file is
    missing or malformed.

    - Single exchange: file order is used as-is; no timestamp required.
    - Multiple exchanges (massive_minus_*): needs a timestamp to interleave
      the constituents in the same order P08 used. Preference order:
      `ts_ms` (what current P06/P07 actually write, regardless of
      exchange) -> the exchange's native time column name, for older/
      foreign files -> the timestamp column read from the corresponding
      clean_data CSV, as a last resort.
    """
    ret_root = Path("linear_return") / data_root
    trades_root = Path("clean_data") / data_root

    need_ts = len(exchanges) > 1
    frames = []

    for exch in exchanges:
        coin_dir = from_massive_name(massive_name, exch)
        if coin_dir is None:
            return None
        rpath = ret_root / exch / coin_dir / f"{date}.csv"
        if not rpath.exists():
            return None

        ret_cols = _read_ret_header(rpath)
        if not ret_cols:
            return None

        header = pd.read_csv(rpath, nrows=0).columns.tolist()
        tc = time_col_for(exch)

        if "ts_ms" in header:
            df = pd.read_csv(rpath, usecols=ret_cols + ["ts_ms"])
            df = df.rename(columns={"ts_ms": "_ts"})
        elif tc in header:
            df = pd.read_csv(rpath, usecols=ret_cols + [tc])
            df = df.rename(columns={tc: "_ts"})
        elif need_ts:
            # last resort: read the timestamp from the trade CSV
            tpath = trades_root / exch / coin_dir / f"{date}.csv"
            if not tpath.exists():
                return None
            ts_df = pd.read_csv(tpath, usecols=[tc])
            df = pd.read_csv(rpath, usecols=ret_cols)
            if len(ts_df) != len(df):
                return None
            df["_ts"] = ts_df[tc].to_numpy()
        else:
            # no timestamp, single exchange: file order is all we need
            df = pd.read_csv(rpath, usecols=ret_cols)

        frames.append(df)

    if len(frames) == 1:
        out = frames[0]
    else:
        out = pd.concat(frames, ignore_index=True)
        if "_ts" not in out.columns:
            return None
        if not out["_ts"].is_monotonic_increasing:
            out = out.sort_values("_ts", kind="mergesort").reset_index(drop=True)

    return out


# ---------- regression ----------

def run_one_regression(lead_ret: np.ndarray,
                       lag_ret: np.ndarray,
                       lead_idx: np.ndarray,
                       lag_idx: np.ndarray,
                       found: np.ndarray,
                       no_dup: np.ndarray):
    n_lead = len(lead_idx)

    # found_counterpart: a lag trade existed in-window for this lead row.
    # no_duplications:  this lead row is the *first* one (in time) to claim
    #                    its matched lag row. Without this, lead rows that
    #                    cluster densely relative to lag rows would repeat
    #                    the same lag_ret[lag_idx] value across multiple
    #                    regression observations -- pseudo-replication that
    #                    shrinks standard errors and inflates significance.
    valid = found & no_dup & (lag_idx >= 0)
    if not valid.any():
        return None
    li = lead_idx[valid]
    lj = lag_idx[valid]
    n_found = int(valid.sum())

    in_bounds = (li < len(lead_ret)) & (lj < len(lag_ret))
    li = li[in_bounds]
    lj = lj[in_bounds]

    x = lead_ret[li]
    y = lag_ret[lj]

    ok = ~np.isnan(x) & ~np.isnan(y)
    x = x[ok]
    y = y[ok]
    n_used = int(len(x))

    if n_used < MIN_OBS:
        return {
            "n_lead": n_lead, "n_found": n_found, "n_used": n_used,
            "slope": np.nan, "intercept": np.nan,
            "tstat": np.nan, "pvalue": np.nan, "r2": np.nan,
        }

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    return {
        "n_lead": n_lead,
        "n_found": n_found,
        "n_used": n_used,
        "slope": float(model.params[1]),
        "intercept": float(model.params[0]),
        "tstat": float(model.tvalues[1]),
        "pvalue": float(model.pvalues[1]),
        "r2": float(model.rsquared),
    }


# ---------- per-(lead, lag, coin, date) ----------

def process_one_lead_lag_date(lead_label: str,
                              lag_label: str,
                              lead_df: pd.DataFrame,
                              lag_df: pd.DataFrame,
                              coin_label: str,
                              date: str,
                              idx_dir: Path,
                              done_keys: set[tuple],
                              label_pairs: list[tuple[str, str]]) -> list[dict]:
    """Return the list of newly-computed result rows (dicts) for this
    (lead, lag, coin, date). Mutates `done_keys` in place to include any
    new keys computed, so a repeated key within the same run is never
    double-counted."""
    prefix = f"{lead_label}_{lag_label}_"
    new_rows = []

    for idx_file in sorted(idx_dir.glob(f"{prefix}*.csv")):
        parsed = parse_idx_filename(idx_file.name, label_pairs)
        if parsed is None:
            continue
        p_lead, p_lag, a, b, c = parsed
        if p_lead != lead_label or p_lag != lag_label:
            continue

        key = (coin_label, date, lead_label, lag_label, a, b, c)
        if key in done_keys:
            continue

        col = f"ret_{a}"
        if col not in lead_df.columns or col not in lag_df.columns:
            continue

        try:
            idx_df = pd.read_csv(idx_file)
        except Exception:
            continue

        lead_col = f"lead_{lead_label}"
        lag_col = f"lag_{lag_label}"
        if lead_col not in idx_df.columns or lag_col not in idx_df.columns:
            continue
        if "no_duplications" not in idx_df.columns:
            # Should not happen with current P08 (it's always written and
            # is part of the cache-validity check), but fail loudly rather
            # than silently skipping the dedup step if an old/malformed
            # index file somehow slips through.
            raise ValueError(
                f"{idx_file} is missing the 'no_duplications' column; "
                "regenerate it with the current P08 before rerunning P09."
            )

        lead_idx = idx_df[lead_col].to_numpy(dtype=np.int64)
        lag_idx = idx_df[lag_col].to_numpy(dtype=np.int64)
        found = idx_df["found_counterpart"].to_numpy(dtype=bool)
        no_dup = idx_df["no_duplications"].to_numpy(dtype=bool)

        lead_ret = lead_df[col].to_numpy(dtype=float)
        lag_ret = lag_df[col].to_numpy(dtype=float)

        stats = run_one_regression(lead_ret, lag_ret, lead_idx, lag_idx,
                                   found, no_dup)
        if stats is None:
            continue

        new_rows.append({
            "coin": coin_label,
            "date": date,
            "lead": lead_label,
            "lag": lag_label,
            "a": a, "b": b, "c": c,
            **stats,
        })
        done_keys.add(key)

    return new_rows


# ---------- per-data_root ----------

def run_one_data_root(data_root: str,
                      label_pairs: list[tuple[str, str]],
                      done_keys: set[tuple],
                      out_file: Path,
                      file_has_header: bool) -> bool:
    """Returns the updated file_has_header flag."""
    index_root = Path("regression_index") / data_root
    if not index_root.exists():
        print(f"[skip] {index_root} missing")
        return file_has_header

    for lead_dir in sorted(d for d in index_root.iterdir() if d.is_dir()):
        lead_label = lead_dir.name
        lead_eff = label_to_effective(lead_label)
        if lead_eff is None:
            continue

        lag_candidates = []
        for ll, gl in label_pairs:
            if ll != lead_label:
                continue
            eff = label_to_effective(gl)
            if eff is not None:
                lag_candidates.append((gl, eff))

        print(f"\n=== {data_root} | lead={lead_label} "
              f"({len(lag_candidates)} lag labels) ===")

        for coin_dir in sorted(d for d in lead_dir.iterdir() if d.is_dir()):
            coin_label = coin_dir.name

            # coin folder label -> canonical massive name
            if lead_label.startswith("massive_minus_"):
                massive_name = (coin_label
                                if (coin_label.startswith("X_")
                                    and coin_label.endswith("USD"))
                                else None)
            else:
                massive_name = to_massive_name(coin_label, lead_label)
            if massive_name is None:
                continue

            for date_dir in sorted(d for d in coin_dir.iterdir() if d.is_dir()):
                date = date_dir.name

                lead_df = load_side_returns(data_root, lead_eff,
                                            massive_name, date)
                if lead_df is None:
                    continue

                for lag_label, lag_eff in lag_candidates:
                    lag_df = load_side_returns(data_root, lag_eff,
                                               massive_name, date)
                    if lag_df is None:
                        continue

                    new_rows = process_one_lead_lag_date(
                        lead_label, lag_label,
                        lead_df, lag_df,
                        coin_label, date,
                        date_dir, done_keys, label_pairs,
                    )

                    if new_rows:
                        file_has_header = append_results(
                            out_file, new_rows, file_has_header)
                        print(f"  {coin_label}/{date}  "
                              f"{lead_label} -> {lag_label}  +{len(new_rows)}")

    return file_has_header


# ---------- summary (streamed) ----------

def build_summary_streaming(out_file: Path, summary_file: Path,
                            chunksize: int = SUMMARY_CHUNK_SIZE) -> pd.DataFrame:
    """Compute the same grouped stats as a `groupby(...).agg(...)` would,
    via a single chunked pass over the regressions CSV, so peak memory is
    bounded by chunk size plus the number of distinct groups -- never the
    full row count, which is what a plain in-memory groupby would need.

    Per group (coin, lead, lag, a, b, c), accumulates: row count, sum of
    n_used, sum and sum-of-squares of slope (for mean/std), sum of tstat,
    sum of r2, and count of pvalue < 0.05. That's exactly what's needed to
    reconstruct n_days, n_used_sum, slope_mean, slope_std, tstat_mean,
    r2_mean, and frac_significant with a one-pass formula instead of a
    two-pass pandas aggregation.
    """
    if not out_file.exists() or out_file.stat().st_size == 0:
        empty = pd.DataFrame()
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
                "n_days": 0, "n_used_sum": 0,
                "slope_sum": 0.0, "slope_sumsq": 0.0,
                "tstat_sum": 0.0, "r2_sum": 0.0, "sig_sum": 0,
            })
            entry["n_days"]      += len(g)
            entry["n_used_sum"]  += int(g["n_used"].sum())
            entry["slope_sum"]   += float(g["slope"].sum())
            entry["slope_sumsq"] += float((g["slope"] ** 2).sum())
            entry["tstat_sum"]   += float(g["tstat"].sum())
            entry["r2_sum"]      += float(g["r2"].sum())
            entry["sig_sum"]     += int((g["pvalue"] < 0.05).sum())

    rows = []
    for (coin, lead, lag, a, b, c), v in acc.items():
        n = v["n_days"]
        mean = v["slope_sum"] / n
        if n > 1:
            var = (v["slope_sumsq"] - n * mean * mean) / (n - 1)
            std = float(np.sqrt(var)) if var > 0 else 0.0
        else:
            std = np.nan
        rows.append({
            "coin": coin, "lead": lead, "lag": lag, "a": a, "b": b, "c": c,
            "n_days": n,
            "n_used_sum": v["n_used_sum"],
            "slope_mean": mean,
            "slope_std": std,
            "tstat_mean": v["tstat_sum"] / n,
            "r2_mean": v["r2_sum"] / n,
            "frac_significant": v["sig_sum"] / n,
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["coin", "lead", "lag", "a", "b", "c"]).reset_index(drop=True)

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_file, index=False)
    return summary


# ---------- main ----------

def main():
    label_pairs = valid_label_pairs()
    print(f"Known (lead, lag) label pairs: {len(label_pairs)}")

    for data_root in DATA_ROOTS:
        out_dir, out_file, summary_file = _out_paths(data_root)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n########## data_root = {data_root} ##########")

        migrate_legacy_schema(out_file)
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
            pd.set_option("display.max_rows", None)
            pd.set_option("display.width", None)
            print(summary.head(20))


if __name__ == "__main__":
    main()