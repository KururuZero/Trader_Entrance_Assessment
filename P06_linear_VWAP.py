"""
P06_linear_VWAP.py   (construction step)

Walk every file under clean_data/, clean_open_data/, clean_close_data/ and
produce an augmented copy under linear_vwap_construction/.

Files are discovered dynamically — this script never hardcodes a coin list
or exchange list. It walks each source dir's subfolders (exchange dirs,
e.g. binance/, massive/, Exchange1/, Exchange2/, ...), then each of those
folders' coin subfolders, then every *.csv inside. Whatever exists on disk
gets processed; nothing needs to be added here when a new coin or a new
Exchange{N} folder shows up.

Output layout is namespaced by source directory to avoid path collisions:
    linear_vwap_construction/clean_data/...
    linear_vwap_construction/clean_open_data/...
    linear_vwap_construction/clean_close_data/...

For each a (ms) in VWAP_interval, adds five columns:
    a_{a}_possible   bool
    a_{a}_5_trades   bool
    a_{a}_side       float  (1=buyers only, 2=sellers only, 3=mixed, NaN=empty)
    a_{a}_next       float  (row index of next possible row >= a ms later)
    a_{a}_vwap       float  (size-weighted price over the centered a ms window)

Caching
-------
For each output file:

1. Compare the constant VWAP_interval list against the intervals already
   present in the destination (from a_{a}_possible columns).
2. For every interval present in both, validate that a_{a}_next is non-NaN
   on:
       - the first row where a_{a}_possible == True
       - the second-to-last such row
   The last possible row is intentionally not checked — it may legitimately
   have no a_{a}_next if no further possible row exists within a ms.
3. Anything that fails step 1 or step 2 is recomputed and merged back.
   If the destination has a different row count than the source, the whole
   file is recomputed (a_{a}_next is row-positional and would be stale).

Logging
-------
Each file prints a compact status line:
    OVLP=[...]  CACHED=[...]  MISSING=[...]  ACTION=...
Status tokens:
    OVLP     intervals present in both source and destination
    CACHED   intervals in OVLP that passed row-level validation
    MISSING  intervals that must be computed
    ACTION   skip | partial | recompute | fresh

Memory design
-------------
Source rows carry several object/string columns (e.g. a UUID-style trade
id, exchange condition codes) that are expensive per-row in memory -- far
more so than the plain int64/float64 timestamp and flag data the interval
math actually needs. Materializing the *entire* row set (original columns
+ all new a_* columns + assorted temporary index/prefix-sum arrays) at
once is what causes OOM kills on unusually large files (e.g. a day's file
that actually spans several days of tick history).

To avoid that, this script splits each file into two passes:

  1. A light pass reads ONLY the timestamp column and the one column used
     to derive is_buyer (is_buyer_maker for binance, conditions
     otherwise), and computes every a_* column as plain numpy arrays.
     This working set is a small, fixed multiple of row count regardless
     of how wide or string-heavy the original columns are.

  2. A chunked write pass re-reads the source in fixed-size row chunks
     (WRITE_CHUNK_SIZE rows at a time) purely to carry the original
     columns through, slices the precomputed arrays to match each chunk,
     and appends the augmented chunk to the destination. At no point is
     the full, wide row set held in memory at once -- only one chunk's
     worth, plus the numeric a_* arrays for the whole file.

This assumes the source file is already sorted by timestamp (true for
clean_data/clean_open_data/clean_close_data, since P03 sorts before
writing them) so that on-disk row order in pass 2 lines up with the
row order the arrays were computed against in pass 1. That assumption is
checked explicitly; if it ever fails, the script falls back to the
original full-in-memory implementation for that one file rather than
silently producing misaligned output.

Writes are atomic (written to a .tmp file, then renamed into place), so a
process killed mid-write (e.g. by the OOM killer) can't leave a corrupt,
partially-written destination that a later run mistakes for a cached file.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- config ----------

SOURCE_DIRS = [Path("clean_data"),
               Path("clean_open_data"),
               Path("clean_close_data")]

OUT_DIR = Path("linear_vwap_construction")

VWAP_interval = [200, 500, 1000, 5000, 10000, 30000]     # ms

MIN_TRADES_IN_WINDOW = 5

# Rows per chunk during the write pass. Bounds peak memory to roughly this
# many full (wide, string-carrying) rows at once, regardless of total file
# size. Lower it if a single file still runs out of memory; raise it for
# fewer, larger I/O calls on files that are comfortably small.
WRITE_CHUNK_SIZE = 200_000

_A_POSSIBLE_RE = re.compile(r"^a_(\d+)_possible$")

INTERVAL_SUFFIXES = ("possible", "5_trades", "side", "next", "vwap")


# ---------- interval discovery ----------

def discover_output_intervals(columns) -> set:
    """Return set of a where the destination has a_{a}_possible column."""
    found = set()
    for c in columns:
        m = _A_POSSIBLE_RE.match(c)
        if m:
            found.add(int(m.group(1)))
    return found


# ---------- validation ----------

def validate_intervals(dst: Path, a_values) -> set:
    """Return subset of a_values that are correctly cached in dst."""
    if not a_values:
        return set()

    usecols = []
    for a in a_values:
        usecols.append(f"a_{a}_possible")
        usecols.append(f"a_{a}_next")

    try:
        df = pd.read_csv(dst, usecols=usecols)
    except Exception:
        return set()

    valid = set()
    for a in a_values:
        pos_col = f"a_{a}_possible"
        nxt_col = f"a_{a}_next"
        if pos_col not in df.columns or nxt_col not in df.columns:
            continue

        pos_mask = df[pos_col].fillna(False).astype(bool).to_numpy()
        pos_idx = np.where(pos_mask)[0]
        if len(pos_idx) < 2:
            valid.add(a)
            continue

        first_pos       = int(pos_idx[0])
        second_last_pos = int(pos_idx[-2])

        nxt_vals = df[nxt_col].to_numpy()
        if (not pd.isna(nxt_vals[first_pos]) and
            not pd.isna(nxt_vals[second_last_pos])):
            valid.add(a)

    return valid


def read_cached_arrays(dst: Path, valid_a) -> dict[str, np.ndarray]:
    """Read only the a_* columns for already-valid intervals from dst,
    without pulling in the (potentially huge) original row columns."""
    if not valid_a:
        return {}
    cols = [f"a_{a}_{suffix}" for a in valid_a for suffix in INTERVAL_SUFFIXES]
    try:
        df = pd.read_csv(dst, usecols=lambda c: c in cols)
    except Exception:
        return {}
    return {c: df[c].to_numpy() for c in df.columns}


# ---------- source column mapping ----------

def _source_columns(exchange: str):
    """(timestamp column, column used to derive is_buyer) for an exchange."""
    if exchange == "binance":
        return "transact_time_ms", "is_buyer_maker"
    return "participant_ts_ms", "conditions"


def _volume_column(exchange: str) -> str:
    return "quantity" if exchange == "binance" else "size"


def _derive_is_buyer(exchange: str, side_series: pd.Series) -> np.ndarray:
    if exchange == "binance":
        return ~side_series.astype(bool).to_numpy()
    cond = side_series.astype(str)
    has_buy  = cond.str.contains("2", regex=False)
    has_sell = cond.str.contains("1", regex=False)
    return (has_buy & ~has_sell).to_numpy()


# ---------- full-row loader (used by the full in-memory fallback only) ----------

def load_and_prepare(path: Path, exchange: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    ts_col, side_col = _source_columns(exchange)
    if side_col not in df.columns:
        raise ValueError(f"{path}: missing {side_col} column")
    if ts_col not in df.columns:
        raise ValueError(f"{path}: missing {ts_col} column")

    df["is_buyer"] = _derive_is_buyer(exchange, df[side_col])
    volume_col = _volume_column(exchange)
    if "price" not in df.columns or volume_col not in df.columns:
        raise ValueError(f"{path}: missing price or {volume_col} column")
    df = df.rename(columns={ts_col: "ts_ms"})
    df = df.sort_values("ts_ms", kind="mergesort").reset_index(drop=True)
    return df


# ---------- light loader (timestamp + side column only) ----------

def load_light(path: Path, exchange: str):
    """Read only the two columns needed to derive is_buyer and drive every
    interval computation -- never the full, potentially object-heavy row.

    Returns (ts_ms, price, volume, is_buyer, sorted_ok):
        ts_ms      int64 ndarray, in on-disk row order
        is_buyer   bool ndarray, in on-disk row order
        sorted_ok  True if ts_ms is already non-decreasing. The chunked
                   write pass assumes on-disk row order matches this
                   array's order; if this is False the caller must use
                   the full in-memory fallback instead.
    """
    ts_col, side_col = _source_columns(exchange)
    volume_col = _volume_column(exchange)

    header = pd.read_csv(path, nrows=0).columns.tolist()
    if ts_col not in header:
        raise ValueError(f"{path}: missing {ts_col} column")
    if side_col not in header:
        raise ValueError(f"{path}: missing {side_col} column")
    for col in ("price", volume_col):
        if col not in header:
            raise ValueError(f"{path}: missing {col} column")

    df = pd.read_csv(path, usecols=[ts_col, "price", volume_col, side_col])
    ts_ms = df[ts_col].to_numpy(dtype="int64")
    price = df["price"].to_numpy(dtype=float)
    volume = df[volume_col].to_numpy(dtype=float)
    is_buyer = _derive_is_buyer(exchange, df[side_col])

    sorted_ok = bool(np.all(ts_ms[1:] >= ts_ms[:-1])) if len(ts_ms) > 1 else True
    return ts_ms, price, volume, is_buyer, sorted_ok


# ---------- core interval math (single source of truth) ----------

def compute_interval_arrays(ts_ms: np.ndarray,
                            price: np.ndarray,
                            volume: np.ndarray,
                            is_buyer: np.ndarray,
                            a_values) -> dict[str, np.ndarray]:
    """Compute every a_{a}_{suffix} column as plain numpy arrays.

    Identical math to the original row-count-independent implementation --
    just returning bare arrays instead of appending to a wide DataFrame, so
    it never holds the string/object columns in memory at all.
    """
    n = len(ts_ms)
    out: dict[str, np.ndarray] = {}
    if n == 0 or not a_values:
        return out

    is_sell = ~is_buyer

    buy_pref  = np.concatenate([[0], np.cumsum(is_buyer.astype("int64"))])
    sell_pref = np.concatenate([[0], np.cumsum(is_sell.astype("int64"))])
    volume_pref = np.concatenate([[0.0], np.cumsum(volume)])
    dollar_pref = np.concatenate([[0.0], np.cumsum(price * volume)])

    first_ts = ts_ms[0]
    last_ts  = ts_ms[-1]

    for a in a_values:
        half = a // 2

        possible = (ts_ms - first_ts >= half) & (last_ts - ts_ms >= half)
        out[f"a_{a}_possible"] = possible

        left_idx  = np.searchsorted(ts_ms, ts_ms - half, side="left")
        right_idx = np.searchsorted(ts_ms, ts_ms + half, side="right")
        counts    = right_idx - left_idx

        out[f"a_{a}_5_trades"] = possible & (counts >= MIN_TRADES_IN_WINDOW)

        buy_in  = buy_pref[right_idx]  - buy_pref[left_idx]
        sell_in = sell_pref[right_idx] - sell_pref[left_idx]
        volume_in = volume_pref[right_idx] - volume_pref[left_idx]
        dollar_in = dollar_pref[right_idx] - dollar_pref[left_idx]

        side = np.full(n, np.nan)
        valid = counts > 0
        side[valid & (buy_in  > 0) & (sell_in == 0)] = 1
        side[valid & (sell_in > 0) & (buy_in  == 0)] = 2
        side[valid & (buy_in  > 0) & (sell_in >  0)] = 3
        out[f"a_{a}_side"] = side
        vwap = np.full(n, np.nan)
        positive_volume = volume_in > 0
        vwap[positive_volume] = dollar_in[positive_volume] / volume_in[positive_volume]
        out[f"a_{a}_vwap"] = vwap

        # ----- next_a (two-pointer) -----
        possible_idx = np.where(possible)[0]
        next_a = np.full(n, np.nan)
        if len(possible_idx) > 0:
            j = 0
            for pos, i in enumerate(possible_idx):
                if j <= pos:
                    j = pos + 1
                target = ts_ms[i] + a
                while j < len(possible_idx) and ts_ms[possible_idx[j]] < target:
                    j += 1
                if j < len(possible_idx):
                    next_a[i] = possible_idx[j]
        out[f"a_{a}_next"] = next_a

    return out


def add_interval_columns(df: pd.DataFrame, a_values) -> pd.DataFrame:
    """Full-DataFrame wrapper around compute_interval_arrays, kept for the
    full in-memory fallback path."""
    ts = df["ts_ms"].to_numpy(dtype="int64")
    is_buyer = df["is_buyer"].to_numpy(dtype=bool)
    price = df["price"].to_numpy(dtype=float)
    volume = df[_volume_column("binance")].to_numpy(dtype=float) if "quantity" in df.columns else df["size"].to_numpy(dtype=float)
    for col, arr in compute_interval_arrays(ts, price, volume, is_buyer, a_values).items():
        df[col] = arr
    return df


# ---------- chunked writer ----------

def write_chunked(src: Path, dst: Path, exchange: str,
                  is_buyer: np.ndarray, arrays: dict[str, np.ndarray],
                  n_rows: int) -> None:
    """Stream src in WRITE_CHUNK_SIZE-row chunks, attach the precomputed
    is_buyer flag and a_* arrays to each chunk by row offset, and append
    to dst. Only one chunk's worth of full (wide) rows is ever in memory
    at once. Writes to a temp file first and renames atomically so a
    process killed mid-write can't leave a corrupt destination."""
    ts_col, _side_col = _source_columns(exchange)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")

    offset = 0
    first_chunk = True
    with open(tmp_path, "w", newline="") as f:
        for chunk in pd.read_csv(src, chunksize=WRITE_CHUNK_SIZE):
            m = len(chunk)
            chunk = chunk.rename(columns={ts_col: "ts_ms"})
            chunk["is_buyer"] = is_buyer[offset:offset + m]
            for col, arr in arrays.items():
                chunk[col] = arr[offset:offset + m]
            chunk.to_csv(f, index=False, header=first_chunk)
            first_chunk = False
            offset += m

    if offset != n_rows:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"{src}: row count changed between passes "
            f"({offset} on write pass vs {n_rows} on light pass) -- "
            "refusing to write a possibly-misaligned output"
        )

    tmp_path.replace(dst)


# ---------- full in-memory fallback (only used if a source isn't sorted) ----------

def process_one_file_full_fallback(src: Path, dst: Path, exchange: str,
                                   missing, valid_a) -> int:
    df = load_and_prepare(src, exchange)

    if valid_a and dst.exists():
        try:
            existing = pd.read_csv(dst)
        except Exception as e:
            print(f"      [warn] cannot read existing dst ({e}); recomputing all")
            missing = list(VWAP_interval)
            valid_a = set()
        else:
            if len(existing) == len(df):
                for a in valid_a:
                    for suffix in INTERVAL_SUFFIXES:
                        col = f"a_{a}_{suffix}"
                        if col in existing.columns:
                            df[col] = existing[col].values
                print(f"      merged cached columns for {_fmt(valid_a)}")
            else:
                print(f"      [warn] row count mismatch "
                      f"(src={len(df)}, dst={len(existing)}); recomputing all")
                missing = list(VWAP_interval)

    df = add_interval_columns(df, missing)

    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(dst)
    return len(df)


# ---------- driver ----------

def _fmt(lst) -> str:
    return "[" + ",".join(str(x) for x in sorted(lst)) + "]"


def process_one_file(src: Path, rel: Path, exchange: str) -> int:
    """Returns row count, or -1 if skipped (all intervals already valid in dst)."""
    dst = OUT_DIR / rel

    src_intervals = list(VWAP_interval)

    if dst.exists() and dst.stat().st_size > 0:
        try:
            dst_header = pd.read_csv(dst, nrows=0).columns.tolist()
            dst_intervals = discover_output_intervals(dst_header)
        except Exception:
            dst_intervals = set()
    else:
        dst_intervals = set()

    overlap  = sorted(set(src_intervals) & dst_intervals)
    valid_a  = validate_intervals(dst, overlap)
    missing  = [a for a in src_intervals if a not in valid_a]

    if not dst.exists():
        action = "fresh"
    elif not missing:
        action = "skip"
    elif valid_a:
        action = "partial"
    else:
        action = "recompute"

    print(f"      OVLP={_fmt(overlap)}  "
          f"CACHED={_fmt(valid_a)}  "
          f"MISSING={_fmt(missing)}  "
          f"ACTION={action}")

    if not missing:
        return -1

    try:
        ts_ms, price, volume, is_buyer, sorted_ok = load_light(src, exchange)
    except Exception as e:
        print(f"      [skip] cannot load source ({e})")
        return -1

    if not sorted_ok:
        print("      [warn] source not time-sorted; using full in-memory fallback")
        try:
            return process_one_file_full_fallback(src, dst, exchange, missing, valid_a)
        except Exception as e:
            print(f"      [skip] fallback failed ({e})")
            return -1

    n_rows = len(ts_ms)

    cached_arrays: dict[str, np.ndarray] = {}
    if valid_a:
        cached_arrays = read_cached_arrays(dst, valid_a)
        expected_cached_cols = len(valid_a) * len(INTERVAL_SUFFIXES)
        incomplete = len(cached_arrays) != expected_cached_cols
        mismatched = any(len(arr) != n_rows for arr in cached_arrays.values())
        if incomplete or mismatched:
            print(f"      [warn] cached columns unreadable/mismatched "
                  f"(src={n_rows} rows); recomputing all")
            missing = list(src_intervals)
            cached_arrays = {}
        else:
            print(f"      merged cached columns for {_fmt(valid_a)}")

    computed_arrays = compute_interval_arrays(ts_ms, price, volume, is_buyer, missing)
    all_arrays = {**cached_arrays, **computed_arrays}

    try:
        write_chunked(src, dst, exchange, is_buyer, all_arrays, n_rows)
    except Exception as e:
        print(f"      [skip] write failed ({e})")
        return -1

    return n_rows


def walk_source(source_dir: Path, counters: dict):
    """Dynamically discover every exchange folder, coin folder, and csv file
    under source_dir — nothing here is hardcoded to a specific coin or
    exchange list, so new coins/exchanges just get picked up automatically."""
    if not source_dir.exists():
        print(f"[skip] {source_dir} missing")
        return

    source_name = source_dir.name

    for exchange_dir in sorted(source_dir.iterdir()):
        if not exchange_dir.is_dir():
            continue
        name = exchange_dir.name

        if name in ("binance", "massive"):
            inner = name
        elif name.startswith("Exchange"):
            inner = "massive"
        else:
            print(f"  [skip dir] {exchange_dir}: unrecognized exchange folder name")
            continue

        for coin_dir in sorted(exchange_dir.iterdir()):
            if not coin_dir.is_dir():
                continue
            for f in sorted(coin_dir.glob("*.csv")):
                rel = Path(source_name) / f.relative_to(source_dir)
                print(f"  [{counters['total']+1:>4}] {rel}")

                n = process_one_file(f, rel, inner)
                counters["total"] += 1

                if n == -1:
                    counters["skipped"] += 1
                    print(f"      -> SKIP")
                else:
                    counters["computed"] += 1
                    print(f"      -> OK ({n:,} rows)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    counters = {"total": 0, "skipped": 0, "computed": 0}

    for source_dir in SOURCE_DIRS:
        print(f"\n=== {source_dir} ===")
        walk_source(source_dir, counters)

    print(f"\nDone.")
    print(f"  total:    {counters['total']}")
    print(f"  computed: {counters['computed']}")
    print(f"  skipped:  {counters['skipped']}")
    print(f"  output:   {OUT_DIR}/")


if __name__ == "__main__":
    main()