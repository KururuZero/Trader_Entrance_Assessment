from __future__ import annotations
"""
P08_regression_assign.py

Generalized cross-exchange lead-lag index assignment.

For every ordered pair of exchanges (lead, lag) with lead != lag, and every
(coin, day) pair that exists on both, find for each lead trade at time t the
lag trade nearest to (t + b) that also falls within:

    (t + max(2a, b - c),  t + b + c)

Both endpoints inclusive, all offsets in milliseconds. Only (a, b, c) triples
satisfying b > 2a are produced, so the two VWAP windows do not overlap.

Exchange set
------------
    binance, massive, Exchange1, Exchange2, Exchange6, Exchange23

Massive substitution
--------------------
`massive` is treated as an aggregate of its constituent exchanges
(Exchange1, Exchange2, Exchange6, Exchange23).  Whenever one side of a pair
is `massive` and the other side is one of those constituents, the `massive`
side is replaced by the *remaining* three constituent exchanges: their CSVs
are read, concatenated and sorted by timestamp.  In that case the side is
labelled `massive_minus_<counterpart>` in output paths and column names.

Coin-dir naming conventions
---------------------------
    binance      BTCUSDT
    massive      X_BTCUSD
    ExchangeN    N_BTCUSD

All three are mapped through a canonical Massive-format name (X_BTCUSD).

Data roots
----------
Three data roots are processed by default:
    clean_data, clean_open_data, clean_close_data

Outputs, one CSV per (data_root, lead, lag, coin, day, a, b, c):

    regression_index/{data_root}/{lead_label}/{coin}/{date}/
        {lead_label}_{lag_label}_{a}_{b}_{c}.csv

Columns:
    lead_{lead_label}  int    row index into the lead (combined) file
    lag_{lag_label}    int    row index into the lag (combined) file, -1 if none
    no_duplications    bool   True for the first lead row that claims a lag row
    found_counterpart  bool   True if a lag trade was found in the window

Caching
-------
Per (data_root, lead, lag, coin, day, a, b, c) output is cached when:
  - the file exists and is non-empty
  - its header contains the four expected columns
  - its row count equals the number of data rows in the lead (combined) file

Lead and lag data are loaded once per (data_root, lead, lag, coin, day) and
reused across all (a, b, c) combos.

Memory design
-------------
`nearest_in_range` -- the only thing that actually consumes lead/lag data --
needs nothing but the two timestamp arrays. The original version of this
script instead loaded each side with a bare `pd.read_csv(path)`, pulling in
every original column: a UUID-style string `id`/`conditions` field, price,
quantity, everything. That's the same failure mode that caused OOM kills in
P06/P07 on oversized files (e.g. a day's file that actually spans several
days of tick history) -- materializing wide, string-heavy rows that the
computation never touches. It's worse here specifically for a
`massive_minus_<X>` side, which concatenates three such files into one before
sorting.

`load_side_ts` below reads ONLY the timestamp column (via `usecols`), for
each constituent file, then concatenates and -- only if the result isn't
already sorted -- stable-sorts it with `mergesort`. Ties break in original
concatenation order, exactly matching what the old full-DataFrame
`sort_values(..., kind="mergesort")` produced and what P09 reproduces when it
reconstructs the same combined side from `linear_return` files to look the
indices back up. Unlike P06, no full-DataFrame fallback path is needed here:
P08's output never carries any of the original wide columns through, so
there's nothing that ever requires holding the full row set in memory in the
first place.

Per-combo outputs are written atomically (temp file + rename) so a process
killed mid-write can't leave a partially-written file that a later run might
mistake for something to validate against the cache checks above.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ---------- config ----------

DATA_ROOTS = [
    "clean_data",
    "clean_open_data",
    "clean_close_data",
]

EXCHANGES = [
    "binance",
    "massive",
    "Exchange1",
    "Exchange2",
    "Exchange6",
    "Exchange23",
]

# `massive` is treated as an aggregate of these constituent exchanges.
MASSIVE_CONSTITUENTS = ["Exchange1", "Exchange2", "Exchange6", "Exchange23"]

OUT_ROOT = Path("regression_index")

# intervals a, in milliseconds
A_VALUES = [200, 500, 1000, 5000, 10000, 30000]

# (b, c) pairs: b = lag in ms, c = tolerance in ms
PARAMS = [
    (5000,   500),
    (10000, 1000),
    (30000, 3000),
    (60000, 6000),
]

# optional filter: set to a list of (lead, lag) tuples to run a subset,
# or None to run every ordered pair.
PAIR_FILTER = None


# ---------- combo generation ----------

def valid_combos():
    """Yield (a, b, c) triples with b > 2a."""
    for a in A_VALUES:
        for b, c in PARAMS:
            if b > 2 * a:
                yield a, b, c


def ordered_pairs():
    """Yield (lead, lag) tuples for every lead != lag, honouring PAIR_FILTER."""
    if PAIR_FILTER is not None:
        for lead, lag in PAIR_FILTER:
            if lead in EXCHANGES and lag in EXCHANGES and lead != lag:
                yield lead, lag
        return
    for lead in EXCHANGES:
        for lag in EXCHANGES:
            if lead != lag:
                yield lead, lag


# ---------- massive substitution ----------

def effective_exchanges(exchange: str, counterpart: str) -> list[str]:
    """Determine which exchange(s) actually supply the data for one side
    of a (lead, lag) pair.

    If `exchange` is 'massive' and `counterpart` is one of massive's
    constituents, massive is replaced by the remaining constituents
    (i.e. massive minus its counterpart).  Otherwise the exchange
    supplies its own data.
    """
    if exchange == "massive" and counterpart in MASSIVE_CONSTITUENTS:
        return [e for e in MASSIVE_CONSTITUENTS if e != counterpart]
    return [exchange]


def side_label(original: str, effective: list[str], counterpart: str) -> str:
    """Label used in output paths and column names for one side."""
    if len(effective) == 1 and effective[0] == original:
        return original
    # only possible when `original == 'massive'` was substituted
    return f"{original}_minus_{counterpart}"


# ---------- coin-name mapping ----------

def to_massive_name(coin_dir: str, exchange: str) -> str | None:
    """Return the canonical Massive-format name X_BTCUSD from any exchange's
    coin directory name, or None if the name can't be mapped."""
    if exchange == "binance":
        if not coin_dir.endswith("USDT"):
            return None
        base = coin_dir[:-4]
        return f"X_{base}USD"

    if exchange == "massive":
        return coin_dir if (coin_dir.startswith("X_") and
                            coin_dir.endswith("USD")) else None

    if exchange.startswith("Exchange"):
        parts = coin_dir.split("_", 1)
        if len(parts) != 2:
            return None
        return f"X_{parts[1]}"

    return None


def from_massive_name(massive_name: str, exchange: str) -> str | None:
    """Return the coin directory name for the given exchange, from X_BTCUSD."""
    if not (massive_name.startswith("X_") and massive_name.endswith("USD")):
        return None
    base = massive_name[2:-3]

    if exchange == "binance":
        return f"{base}USDT"
    if exchange == "massive":
        return massive_name
    if exchange.startswith("Exchange"):
        num = exchange.replace("Exchange", "")
        return f"{num}_{base}USD"

    return None


# ---------- helpers ----------

def count_csv_rows(path: Path) -> int:
    """Fast byte-mode line count minus header."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def time_col_for(exchange: str) -> str:
    return "transact_time_ms" if exchange == "binance" else "participant_ts_ms"


def out_columns(lead_label: str, lag_label: str):
    return [
        f"lead_{lead_label}",
        f"lag_{lag_label}",
        "no_duplications",
        "found_counterpart",
    ]


def is_cached(path: Path, expected_rows: int,
              lead_label: str, lag_label: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return False
    required = out_columns(lead_label, lag_label)
    if not all(c in header for c in required):
        return False
    return count_csv_rows(path) == expected_rows


def coins_for_exchange(data_root: str, exchange: str) -> dict | None:
    """Return {massive_name: coin_dir_path} for an exchange, or None if the
    exchange directory is missing."""
    root = Path(data_root) / exchange
    if not root.exists():
        return None
    out: dict[str, Path] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        mn = to_massive_name(d.name, exchange)
        if mn is not None:
            out[mn] = d
    return out


# ---------- core alignment ----------

def nearest_in_range(lead_ts: np.ndarray,
                     lag_ts: np.ndarray,
                     b: int, c: int, a: int):
    """Return (lag_idx, found) for every lead row.

    Window: [t + max(2a, b - c), t + b + c], endpoints inclusive.
    Nearest to (t + b) wins; ties favour the earlier index.
    """
    n = len(lead_ts)
    m = len(lag_ts)
    if n == 0 or m == 0:
        return (np.full(n, -1, dtype=np.int64),
                np.zeros(n, dtype=bool))

    lower_off = max(2 * a, b - c)
    upper_off = b + c

    lead_lower  = lead_ts + lower_off
    lead_upper  = lead_ts + upper_off
    lead_target = lead_ts + b

    lo = np.searchsorted(lag_ts, lead_lower, side="left")
    hi = np.searchsorted(lag_ts, lead_upper, side="right")
    p  = np.searchsorted(lag_ts, lead_target, side="left")

    c1 = p - 1
    c2 = p

    BIG = np.iinfo(np.int64).max

    v1 = (c1 >= lo) & (c1 < hi) & (c1 >= 0) & (c1 < m)
    v2 = (c2 >= lo) & (c2 < hi) & (c2 >= 0) & (c2 < m)

    d1 = np.full(n, BIG, dtype=np.int64)
    d2 = np.full(n, BIG, dtype=np.int64)

    idx1 = np.where(v1)[0]
    if len(idx1):
        d1[idx1] = np.abs(lag_ts[c1[idx1]] - lead_target[idx1])

    idx2 = np.where(v2)[0]
    if len(idx2):
        d2[idx2] = np.abs(lag_ts[c2[idx2]] - lead_target[idx2])

    use1   = d1 <= d2
    chosen = np.where(use1, c1, c2)
    dist   = np.where(use1, d1, d2)

    found = dist < BIG
    lag_idx = np.full(n, -1, dtype=np.int64)
    lag_idx[found] = chosen[found]

    return lag_idx, found


def compute_no_dup(lag_idx: np.ndarray) -> np.ndarray:
    """True for the first lead row that claims a given lag row."""
    n = len(lag_idx)
    no_dup = np.zeros(n, dtype=bool)
    valid = lag_idx >= 0
    if not valid.any():
        return no_dup
    positions = np.where(valid)[0]
    values    = lag_idx[positions]
    _, first_idx = np.unique(values, return_index=True)
    no_dup[positions[first_idx]] = True
    return no_dup


# ---------- side loading ----------

def _side_paths(data_root: str, exchanges: list[str],
                massive_name: str, date: str):
    """Resolve the CSV path for each exchange in a side.  Returns a list of
    Paths (same length as `exchanges`) or None if any is missing."""
    paths = []
    for exch in exchanges:
        coin_dir = from_massive_name(massive_name, exch)
        if coin_dir is None:
            return None
        path = Path(data_root) / exch / coin_dir / f"{date}.csv"
        if not path.exists():
            return None
        paths.append(path)
    return paths


def load_side_ts(data_root: str, exchanges: list[str],
                  massive_name: str, date: str) -> np.ndarray | None:
    """Read ONLY the timestamp column needed for one side of a pair.

    Light-pass counterpart to the old `load_side`, which read every column
    (UUID-style `id`, `conditions`, price, quantity, ...) via a bare
    `pd.read_csv(path)` just to pull out one column at the end. That's the
    same wide-row-materialization problem that caused OOM kills in P06/P07,
    and it's worse here for a `massive_minus_<X>` side, which concatenates
    three such files before sorting.

    Concatenates (in `exchanges` order) if there's more than one constituent,
    then stable-sorts by timestamp with mergesort if the result isn't already
    monotonic. Ties break in original concatenation order -- identical
    ordering semantics to the previous full-DataFrame
    `sort_values(time_col, kind="mergesort")`, and identical to what P09
    reproduces when it rebuilds the same combined side from `linear_return`
    files to look these indices back up.

    Returns a sorted int64 ndarray, or None if any input file is missing.
    """
    paths = _side_paths(data_root, exchanges, massive_name, date)
    if paths is None:
        return None

    arrays = []
    for path, exch in zip(paths, exchanges):
        tc = time_col_for(exch)
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if tc not in header:
            raise ValueError(f"missing time column {tc} in {path}")
        col = pd.read_csv(path, usecols=[tc])[tc].to_numpy(dtype=np.int64)
        arrays.append(col)

    ts = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)

    if len(ts) > 1 and not np.all(ts[1:] >= ts[:-1]):
        ts = np.sort(ts, kind="mergesort")

    return ts


def count_side_rows(data_root: str, exchanges: list[str],
                    massive_name: str, date: str) -> int:
    paths = _side_paths(data_root, exchanges, massive_name, date)
    return sum(count_csv_rows(p) for p in paths)


# ---------- per-pair driver ----------

def process_one_pair(data_root: str,
                     lead_eff: list[str],
                     lag_eff: list[str],
                     lead_label: str,
                     lag_label: str,
                     massive_name: str,
                     date: str,
                     rel_dir: Path) -> dict:
    """Return dict {(a,b,c): 'skip'|'computed'} for one (lead, lag, coin, day)."""
    expected_rows = count_side_rows(data_root, lead_eff, massive_name, date)

    statuses = {}
    need_any = False
    for a, b, c in valid_combos():
        out_path = (OUT_ROOT / data_root / rel_dir
                    / f"{lead_label}_{lag_label}_{a}_{b}_{c}.csv")
        if is_cached(out_path, expected_rows, lead_label, lag_label):
            statuses[(a, b, c)] = ("skip", out_path)
        else:
            statuses[(a, b, c)] = ("compute", out_path)
            need_any = True

    if not need_any:
        return {k: "skip" for k in statuses}

    lead_ts = load_side_ts(data_root, lead_eff, massive_name, date)
    lag_ts  = load_side_ts(data_root, lag_eff,  massive_name, date)

    if lead_ts is None or lag_ts is None:
        # Shouldn't happen -- the caller only reaches here for dates already
        # confirmed present on every constituent file -- but guard against a
        # file disappearing between the presence check and this read.
        return {k: "skip" for k in statuses}

    lead_idx_arr = np.arange(len(lead_ts), dtype=np.int64)

    lead_col = f"lead_{lead_label}"
    lag_col  = f"lag_{lag_label}"

    results = {}
    for a, b, c in valid_combos():
        status, out_path = statuses[(a, b, c)]
        if status == "skip":
            results[(a, b, c)] = "skip"
            continue

        lag_idx, found = nearest_in_range(lead_ts, lag_ts, b, c, a)
        no_dup         = compute_no_dup(lag_idx)

        out = pd.DataFrame({
            lead_col:            lead_idx_arr,
            lag_col:             lag_idx,
            "no_duplications":   no_dup,
            "found_counterpart": found,
        })

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        out.to_csv(tmp_path, index=False)
        tmp_path.replace(out_path)
        results[(a, b, c)] = "computed"

    return results


# ---------- per-(data_root, lead, lag) driver ----------

def _dates_across(dirs: list[Path]) -> set:
    """Dates (CSV stems) present in every directory of `dirs`."""
    if not dirs:
        return set()
    sets = [set(f.stem for f in d.glob("*.csv")) for d in dirs]
    return set.intersection(*sets)


def run_exchange_pair(data_root: str,
                      lead_orig: str,
                      lag_orig: str,
                      counters: dict):
    lead_eff = effective_exchanges(lead_orig, lag_orig)
    lag_eff  = effective_exchanges(lag_orig, lead_orig)

    lead_label = side_label(lead_orig, lead_eff, lag_orig)
    lag_label  = side_label(lag_orig,  lag_eff,  lead_orig)

    # --- build coin maps for each effective exchange -----------------------
    lead_maps = {}
    for exch in lead_eff:
        m = coins_for_exchange(data_root, exch)
        if m is None:
            print(f"  [skip] {data_root}/{exch} missing")
            return
        lead_maps[exch] = m

    lag_maps = {}
    for exch in lag_eff:
        m = coins_for_exchange(data_root, exch)
        if m is None:
            print(f"  [skip] {data_root}/{exch} missing")
            return
        lag_maps[exch] = m

    # coins present on *every* exchange involved
    common_coins = set.intersection(*(set(m) for m in lead_maps.values()))
    common_coins &= set.intersection(*(set(m) for m in lag_maps.values()))

    print(f"\n=== {data_root} | {lead_label} leads {lag_label} ===")

    combos = list(valid_combos())

    for massive_name in sorted(common_coins):
        lead_dirs = [lead_maps[e][massive_name] for e in lead_eff]
        lag_dirs  = [lag_maps[e][massive_name]  for e in lag_eff]

        common_dates = _dates_across(lead_dirs) & _dates_across(lag_dirs)

        # coin folder label: keep the lead exchange's own coin-dir name when
        # the lead is a single exchange; otherwise fall back to the canonical
        # massive name (this is the case for a substituted massive lead).
        if len(lead_eff) == 1:
            coin_label = lead_dirs[0].name
        else:
            coin_label = massive_name

        for date in sorted(common_dates):
            rel_dir = Path(lead_label) / coin_label / date

            try:
                results = process_one_pair(
                    data_root, lead_eff, lag_eff,
                    lead_label, lag_label,
                    massive_name, date, rel_dir,
                )
            except Exception as e:
                print(f"    [error] {coin_label}/{date}: "
                      f"{type(e).__name__}: {e}")
                continue

            n_ok  = sum(1 for v in results.values() if v == "computed")
            n_skp = sum(1 for v in results.values() if v == "skip")
            counters["total"]    += len(combos)
            counters["computed"] += n_ok
            counters["skipped"]  += n_skp

            print(f"    {coin_label}/{date}  "
                  f"computed={n_ok:>2}  skipped={n_skp:>2}")


# ---------- main ----------

def main():
    combos = list(valid_combos())
    print(f"Combos (a, b, c) with b > 2a: {len(combos)}")

    pairs = list(ordered_pairs())
    print(f"Ordered (lead, lag) pairs: {len(pairs)}")

    counters = {"total": 0, "computed": 0, "skipped": 0}

    for data_root in DATA_ROOTS:
        for lead_exchange, lag_exchange in pairs:
            try:
                run_exchange_pair(data_root, lead_exchange, lag_exchange,
                                  counters)
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                raise

    print(f"\nDone.")
    print(f"  total:    {counters['total']:,}")
    print(f"  computed: {counters['computed']:,}")
    print(f"  skipped:  {counters['skipped']:,}")
    print(f"  output:   {OUT_ROOT}/")


if __name__ == "__main__":
    main()