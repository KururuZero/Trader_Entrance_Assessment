"""
P07_linear_return.py

Read augmented files from linear_vwap_construction/ and produce parallel copies
under linear_return/ with derived return columns for every interval `a` that
already exists in the source file.

The intervals are discovered dynamically from the source file's columns
(any column named `a_{N}_possible` defines an interval a = N).

Columns added per discovered `a` (force-zero convention throughout):

    ret_{a}              = vwap2 / vwap1 - 1
    logret_{a}           = sign(ret) * log(|ret|), forced to 0 when ret == 0
    ret_per_dt_{a}       = ret / dt
    log_ret_per_dt_{a}   = sign(ret/dt) * log(|ret/dt|), forced to 0 when ret/dt == 0
    logret_per_dt_{a}    = logret / dt, forced to 0 when ret == 0
    excess_dt_{a}        = dt - a
    dt_{a}               = dt

NOTE: earlier drafts of this script also emitted NaN-at-zero variants of the
log columns plus duplicate `*_force0_{a}` columns. Those have been removed —
only the force-zero convention is kept, so this script now writes
byte-for-byte-identical column names and semantics to P06. A `linear_return/`
directory can therefore safely contain a mix of files produced by either
script with no schema mismatch.

Caching rules
-------------
For each output file:

1. Compare the set of intervals in the source (from `a_{a}_possible` columns)
   with the set of intervals in the destination (from `ret_{a}` columns).
2. For every interval present in both, validate that `ret_{a}` is non-NaN on:
       - the first row where `a_{a}_possible == True`
       - the second-to-last such row
3. Anything that fails step 1 or step 2 is recomputed and merged back.
   If the destination has a different row count than the source, the whole
   file is recomputed (next_a indices are row-positional and would be stale).

Logging
-------
Each file prints a compact status line:
    SRC=[...] DST=[...] OVLP=[...] CACHED=[...] MISSING=[...] ACTION=...

Memory design
-------------
The source files here (P06's output) are even wider than the raw clean_data
files: they carry every original column (including string/object ones like
a UUID-style trade id and exchange condition codes) *plus* the 24 a_*
columns P06 added. Loading a whole such file for an unusually large day
(e.g. one that spans several days of tick history) is what causes OOM
kills on this script, same as it did on P06.

The actual computation here only ever needs three things per interval:
`ts_ms`, `a_{a}_vwap`, and that interval's `a_{a}_next` column — never the id,
conditions, quantity, price, or any of the other a_* columns. So, as with P06,
this script splits into:

  1. A light pass that reads ONLY `ts_ms`, the specific `a_{a}_vwap` and
      `a_{a}_next` columns needed for whatever intervals are missing, via
     `usecols`. This working set is a small, fixed multiple of row count
     regardless of how wide the source file actually is.

  2. A chunked write pass that re-reads the source in fixed-size row
     chunks (WRITE_CHUNK_SIZE rows at a time) purely to carry the original
     (wide) columns through unmodified, slices the precomputed return
     arrays to match each chunk's row range, and appends the augmented
     chunk to the destination. At no point is the full row set held in
     memory at once.

Unlike P06, no sortedness assumption is needed: `a_{a}_next` values are
plain row-positional indices into this same file, so as long as the light
pass and the chunked write pass see the same on-disk row order — which
they always do, since both simply read the same file front-to-back — the
indexing is correct regardless of how the data happens to be ordered.

Writes are atomic (written to a .tmp file, then renamed into place), so a
process killed mid-write can't leave a corrupt, partially-written
destination that a later run mistakes for a cached file.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path("linear_vwap_construction")
OUT_DIR = Path("linear_return")

RET_SUFFIXES = ("ret", "logret", "ret_per_dt",
                "log_ret_per_dt", "logret_per_dt",
                "excess_dt", "dt")

# Rows per chunk during the write pass. Bounds peak memory to roughly this
# many full (wide, string-carrying) rows at once, regardless of total file
# size. Lower it if a single file still runs out of memory.
WRITE_CHUNK_SIZE = 200_000

_A_POSSIBLE_RE = re.compile(r"^a_(\d+)_possible$")
_RET_RE        = re.compile(r"^ret_(\d+)$")


# ---------- interval discovery ----------

def discover_source_intervals(columns) -> list:
    """Return sorted list of `a` from the source's a_{a}_possible columns."""
    found = set()
    for c in columns:
        m = _A_POSSIBLE_RE.match(c)
        if m:
            found.add(int(m.group(1)))
    return sorted(found)


def discover_output_intervals(columns) -> set:
    """Return set of `a` where the destination has a ret_{a} column."""
    found = set()
    for c in columns:
        m = _RET_RE.match(c)
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
        usecols.append(f"ret_{a}")

    try:
        df = pd.read_csv(dst, usecols=usecols)
    except Exception:
        return set()

    valid = set()
    for a in a_values:
        pos_col = f"a_{a}_possible"
        ret_col = f"ret_{a}"
        if pos_col not in df.columns or ret_col not in df.columns:
            continue

        pos_mask = df[pos_col].fillna(False).astype(bool).to_numpy()
        pos_idx = np.where(pos_mask)[0]
        if len(pos_idx) < 2:
            valid.add(a)
            continue

        first_pos       = int(pos_idx[0])
        second_last_pos = int(pos_idx[-2])

        ret_vals = df[ret_col].to_numpy()
        if (not pd.isna(ret_vals[first_pos]) and
            not pd.isna(ret_vals[second_last_pos])):
            valid.add(a)

    return valid


def read_cached_arrays(dst: Path, valid_a) -> dict[str, np.ndarray]:
    """Read only the ret_*/logret_*/etc. columns for already-valid intervals
    from dst, without pulling in the (potentially huge) original columns."""
    if not valid_a:
        return {}
    cols = [f"{suffix}_{a}" for a in valid_a for suffix in RET_SUFFIXES]
    try:
        df = pd.read_csv(dst, usecols=lambda c: c in cols)
    except Exception:
        return {}
    return {c: df[c].to_numpy() for c in df.columns}


# ---------- light loader (ts_ms, VWAP, and needed next_a columns only) -----

def load_light_for_intervals(src: Path, header, a_list):
    """Read only ts_ms, a_{a}_vwap, and a_{a}_next columns for a_list (minus
    any that don't exist in this source) -- never the wide original columns.

    Returns (ts_ms, vwap_arrays, next_arrays, skipped, n_rows) where `skipped`
    is the subset of a_list missing either required VWAP or next-row column.
    """
    usable = [a for a in a_list
              if f"a_{a}_next" in header and f"a_{a}_vwap" in header]
    skipped = sorted(set(a_list) - set(usable))

    cols = ["ts_ms"]
    cols += [f"a_{a}_vwap" for a in usable]
    cols += [f"a_{a}_next" for a in usable]
    df = pd.read_csv(src, usecols=cols)

    ts_ms = df["ts_ms"].to_numpy(dtype="int64")
    vwap_arrays = {a: df[f"a_{a}_vwap"].to_numpy(dtype=float) for a in usable}
    next_arrays = {a: df[f"a_{a}_next"].to_numpy(dtype=float) for a in usable}

    return ts_ms, vwap_arrays, next_arrays, skipped, len(df)


# ---------- core computation ----------

def signed_log(x: np.ndarray) -> np.ndarray:
    """sign(x) * log(|x|). Returns NaN at x == 0 (caller overrides to 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sign(x) * np.log(np.abs(x))


def compute_return_arrays(ts_ms: np.ndarray,
                          vwap_arrays: dict[int, np.ndarray],
                          next_arrays: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute every {suffix}_{a} column as plain numpy arrays. Identical
    math to the original DataFrame-based implementation -- just returning
    bare arrays instead of appending to a wide DataFrame."""
    n = len(ts_ms)
    out: dict[str, np.ndarray] = {}
    if n == 0 or not next_arrays:
        return out

    for a, next_arr in next_arrays.items():
        vwap = vwap_arrays[a]
        valid_mask = ~np.isnan(next_arr) & ~np.isnan(vwap)

        t2 = np.full(n, np.nan)
        p2 = np.full(n, np.nan)

        if valid_mask.any():
            idx       = next_arr[valid_mask].astype(int)
            in_bounds = (idx >= 0) & (idx < n)
            positions = np.where(valid_mask)[0][in_bounds]
            idx_ok    = idx[in_bounds]
            t2[positions] = ts_ms[idx_ok].astype(float)
            p2[positions] = vwap[idx_ok]

        t1 = ts_ms.astype(float)
        dt = t2 - t1

        with np.errstate(divide="ignore", invalid="ignore"):
            ret    = np.where((vwap > 0) & ~np.isnan(p2), p2 / vwap - 1.0, np.nan)
            logret = np.where((vwap > 0) & (p2 > 0),     np.log(p2 / vwap), np.nan)

            ret_per_dt      = np.where(dt > 0, ret / dt, np.nan)
            log_ret_per_dt  = signed_log(ret_per_dt)
            logret_per_dt   = np.where(dt > 0, logret / dt, np.nan)
            excess_dt       = np.where(~np.isnan(dt), dt - a, np.nan)

        # ---- force 0 (never NaN) where the linear return is exactly 0 ----
        zero_ret   = (ret == 0)          # only True when ret is a real 0, not NaN
        zero_retdt = (ret_per_dt == 0)

        logret         = np.where(zero_ret,   0.0, logret)
        logret_per_dt  = np.where(zero_ret,   0.0, logret_per_dt)
        log_ret_per_dt = np.where(zero_retdt, 0.0, log_ret_per_dt)

        out[f"ret_{a}"]            = ret
        out[f"logret_{a}"]         = logret
        out[f"ret_per_dt_{a}"]     = ret_per_dt
        out[f"log_ret_per_dt_{a}"] = log_ret_per_dt
        out[f"logret_per_dt_{a}"]  = logret_per_dt
        out[f"excess_dt_{a}"]      = excess_dt
        out[f"dt_{a}"]             = dt

    return out


def add_return_columns(df: pd.DataFrame, a_values) -> pd.DataFrame:
    """Full-DataFrame wrapper around compute_return_arrays. Kept only for
    reference / potential ad-hoc use; the driver below no longer calls this
    since it would defeat the point of the light/chunked split."""
    vwap_arrays = {}
    next_arrays = {}
    for a in a_values:
        vwap_col = f"a_{a}_vwap"
        col = f"a_{a}_next"
        if vwap_col in df.columns and col in df.columns:
            vwap_arrays[a] = df[vwap_col].to_numpy(dtype=float)
            next_arrays[a] = df[col].to_numpy(dtype=float)
    ts_ms = df["ts_ms"].to_numpy(dtype="int64")
    for col, arr in compute_return_arrays(ts_ms, vwap_arrays, next_arrays).items():
        df[col] = arr
    return df


# ---------- chunked writer ----------

def write_chunked(src: Path, dst: Path,
                  arrays: dict[str, np.ndarray], n_rows: int) -> None:
    """Stream src in WRITE_CHUNK_SIZE-row chunks, attach the precomputed
    return arrays to each chunk by row offset, and append to dst. Only one
    chunk's worth of full (wide) rows is ever in memory at once. Writes to
    a temp file first and renames atomically so a process killed mid-write
    can't leave a corrupt destination."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")

    offset = 0
    first_chunk = True
    with open(tmp_path, "w", newline="") as f:
        for chunk in pd.read_csv(src, chunksize=WRITE_CHUNK_SIZE):
            m = len(chunk)
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


# ---------- driver ----------

def _fmt(lst) -> str:
    return "[" + ",".join(str(x) for x in sorted(lst)) + "]"


def process_one_file(src: Path, rel: Path) -> int:
    """Returns row count, or -1 if skipped (all intervals already valid in dst)."""
    dst = OUT_DIR / rel

    # --- 1. read source header, discover source intervals ---
    try:
        src_header = pd.read_csv(src, nrows=0).columns.tolist()
    except Exception as e:
        print(f"      [skip] cannot read source header ({e})")
        return -1

    src_intervals = discover_source_intervals(src_header)
    if not src_intervals:
        print(f"      [skip] no a_{{a}}_possible columns in source")
        return -1

    # --- 2. read destination header, discover destination intervals ---
    if dst.exists() and dst.stat().st_size > 0:
        try:
            dst_header = pd.read_csv(dst, nrows=0).columns.tolist()
            dst_intervals = discover_output_intervals(dst_header)
            dst_intervals &= {
                a for a in src_intervals if f"a_{a}_vwap" in dst_header
            }
        except Exception:
            dst_intervals = set()
    else:
        dst_intervals = set()

    overlap = sorted(set(src_intervals) & dst_intervals)
    valid_a = validate_intervals(dst, overlap)
    missing = [a for a in src_intervals if a not in valid_a]

    # --- 3. status line ---
    if not dst.exists():
        action = "fresh"
    elif not missing:
        action = "skip"
    elif valid_a:
        action = "partial"
    else:
        action = "recompute"

    print(f"      SRC={_fmt(src_intervals)}  "
          f"DST={_fmt(dst_intervals)}  "
          f"OVLP={_fmt(overlap)}  "
          f"CACHED={_fmt(valid_a)}  "
          f"MISSING={_fmt(missing)}  "
          f"ACTION={action}")

    if not missing:
        return -1

    # --- 4. light pass: ts_ms, VWAP, and needed a_{a}_next columns only ---
    try:
        ts_ms, vwap_arrays, next_arrays, skipped_no_next, n_rows = \
            load_light_for_intervals(src, src_header, missing)
    except Exception as e:
        print(f"      [skip] cannot read source columns ({e})")
        return -1

    if skipped_no_next:
        print(f"      [warn] no a_{{a}}_next column for {_fmt(skipped_no_next)}; "
              f"leaving those intervals unset")

    # --- 5. merge cached columns (usecols-only dst read) or recompute all ---
    cached_arrays: dict[str, np.ndarray] = {}
    if valid_a:
        cached_arrays = read_cached_arrays(dst, valid_a)
        expected_cached_cols = len(valid_a) * len(RET_SUFFIXES)
        incomplete = len(cached_arrays) != expected_cached_cols
        mismatched = any(len(arr) != n_rows for arr in cached_arrays.values())
        if incomplete or mismatched:
            print(f"      [warn] cached columns unreadable/mismatched "
                  f"(src={n_rows} rows); recomputing all")
            missing = list(src_intervals)
            ts_ms, vwap_arrays, next_arrays, skipped_no_next, n_rows = \
                load_light_for_intervals(src, src_header, missing)
            if skipped_no_next:
                print(f"      [warn] no a_{{a}}_next column for "
                      f"{_fmt(skipped_no_next)}; leaving those intervals unset")
            cached_arrays = {}
        else:
            print(f"      merged cached columns for {_fmt(valid_a)}")

    # --- 6. compute + chunked write ---
    computed_arrays = compute_return_arrays(ts_ms, vwap_arrays, next_arrays)
    all_arrays = {**cached_arrays, **computed_arrays}

    try:
        write_chunked(src, dst, all_arrays, n_rows)
    except Exception as e:
        print(f"      [skip] write failed ({e})")
        return -1

    return n_rows


def main():
    if not SRC_DIR.exists():
        print(f"Source {SRC_DIR} missing — nothing to do.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    counters = {"total": 0, "skipped": 0, "computed": 0}

    for f in sorted(SRC_DIR.rglob("*.csv")):
        rel = f.relative_to(SRC_DIR)
        print(f"  [{counters['total']+1:>4}] {rel}")

        n = process_one_file(f, rel)
        counters["total"] += 1

        if n == -1:
            counters["skipped"] += 1
            print(f"      -> SKIP (all intervals cached)")
        else:
            counters["computed"] += 1
            print(f"      -> OK ({n:,} rows)")

    print(f"\nDone.")
    print(f"  total files:    {counters['total']}")
    print(f"  computed:       {counters['computed']}")
    print(f"  skipped:        {counters['skipped']}")
    print(f"  output:         {OUT_DIR}/")


if __name__ == "__main__":
    main()