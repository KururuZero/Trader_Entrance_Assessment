"""
P11_focus_exchange23.py
=======================

Focused follow-up on the ONE result in P10 that is worth defending:

    <lead venue>  ->  Exchange23,   a = 1000 ms,  b = 5000 ms,   BTC / ETH / XRP

P10's A1 table ranks 1,440 specs by BH q-value.  The top of that table is not
the proposal's primary hypothesis (binance <-> massive, best q = 0.107, i.e.
nothing) -- it is a cluster of specs that all share the same lag venue:

    clean_data  binance                  -> Exchange23  a=1000 b=5000  corr .126  t=5.91  q=.0015
    clean_data  massive_minus_Exchange23 -> Exchange23  a=1000 b=5000  corr .214  t=6.51  q=.0015
    clean_data  Exchange1                -> Exchange23  a=1000 b=5000  corr .136  t=5.49  q=.0026
    clean_data  Exchange1                -> Exchange23  a=10000 b=30000 corr .045 t=5.03  q=.0026
    clean_data  binance                  -> Exchange1   a=1000 b=5000  corr .042  t=5.13  q=.0026

and the REVERSE of each of those is close to zero:

    Exchange23 -> binance                   corr .055  t=2.56  q=.22
    Exchange23 -> Exchange1                 corr .041  t=1.92  q=.40
    Exchange23 -> massive_minus_Exchange23  corr .022  t=1.02  q=.69   <-- the clean one

That asymmetry is the whole finding: Exchange23 follows the rest of the market
and leads nothing.  A spurious correlation (shared volatility, overlapping
windows, clock skew) would be symmetric.  This script is built to try to kill
that finding; if it survives, it is the result the report should be built on.

What this script does that P10 did not
--------------------------------------
  1. b is a FREE parameter.  P10 only ever tested b in {5000,10000,30000,60000}
     with c = b/10, because it consumed P08's pre-built index files.  This
     script rebuilds the lead->lag matching itself (identical window rule to
     P08) straight off the vwap_return/*.npy arrays, so it can trace the decay
     curve at b = 1500 .. 20000 and locate the half-life.  a is still limited
     to P06's grid {200,500,1000,5000,10000,30000}; finer a needs a P06 rerun.
  2. Per-coin results for the Exchange23 pairs.  A6 only covered binance/massive,
     so nothing in P10's output tells you whether this is a BTC-only effect.
  3. Hour-of-day, not just the US open/close day-split.  A4 compared
     clean_open_data vs clean_close_data; this buckets by UTC and US/Eastern
     hour within clean_data.
  4. Four placebos: reverse direction, negative b, wrong-day lead, wrong-coin
     lead.  All four should be flat if the effect is real.
  5. Staleness diagnostics on the LAG side.  P10's mean_dt_ratio is the LEAD
     side only.  If Exchange23 simply prints late, "prediction" is just a stale
     quote catching up and is not tradable.  This is the most likely way the
     result dies -- it is tested explicitly (F5).
  6. Continuation vs reversal: the same signal evaluated at 2b and 4b.  A real
     information transfer continues; transient impact reverses.
  7. A cost curve over a |signal| threshold, not just a flat gross number, plus
     a date-block bootstrap CI and a first-half / second-half OOS split.

Outputs (under ROOT/regression_result_vwap/focus/)
--------------------------------------------------
  F0_daily.csv        one row per (coin, date, lead, lag, a, b, variant)
  F1_spec.csv         per-spec aggregate + BH q over this (small) grid
  F2_by_coin.csv      per-coin aggregate at the focus specs
  F3_hours.csv        per hour-of-day bucket at the focus spec
  F4_placebo.csv      reverse / negative-b / shuffled-day / shuffled-coin
  F5_staleness.csv    lag-side dt ratios, lag idle time, conditional corr
  F6_horizon.csv      signal evaluated at b, 2b, 4b (continuation vs reversal)
  F7_cost_curve.csv   |x| threshold x cost scenario -> net bps, and capacity
  F8_oos.csv          first-half -> second-half out-of-sample
  focus_summary.md    the numbers, in the order the report needs them

Usage
-----
  python P11_focus_exchange23.py                     # everything
  python P11_focus_exchange23.py --stage sweep
  python P11_focus_exchange23.py --stage hours,placebo,cost
  python P11_focus_exchange23.py --root /path/to/project
  python P11_focus_exchange23.py --boot 2000

Memory: same discipline as P10 -- one (coin, date) side in memory at a time,
npy memmaps, float32, closed-form OLS.  Peak well under 400 MB on BTC days.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
except Exception:
    sps = None


# ==================================================================== config

ROOT = Path(os.environ.get("LEADLAG_ROOT", Path.home()))

VWAP_DIR = "vwap_return"
RESULT_DIR = "regression_result_vwap"
OUT_DIR = ROOT / RESULT_DIR / "focus"

MASSIVE_CONSTITUENTS = ["Exchange1", "Exchange2", "Exchange6", "Exchange23"]
A_VALUES = [200, 500, 1000, 5000, 10000, 30000]      # fixed by P06's output

# ---- the focus set, straight off P10's A1 ranking -------------------------
FOCUS_COINS = ["X_BTCUSD", "X_ETHUSD", "X_XRPUSD"]   # only coins Exchange23 fits
FOCUS_DATA_ROOT = "clean_data"                        # all hours; A4 split is redone here

# (lead_label, lag_label, role).  "signal" = the claim; "control" = must be flat.
FOCUS_PAIRS = [
    ("binance",                  "Exchange23",               "signal"),
    ("massive_minus_Exchange23", "Exchange23",               "signal"),
    ("Exchange1",                "Exchange23",               "signal"),
    ("binance",                  "Exchange1",                "signal"),
    ("Exchange23",               "binance",                  "control"),
    ("Exchange23",               "massive_minus_Exchange23", "control"),
    ("Exchange23",               "Exchange1",                "control"),
    ("Exchange1",                "binance",                  "control"),
]

FOCUS_A = [500, 1000, 5000]                 # around the a=1000 winner
B_GRID = [1500, 2000, 3000, 4000, 5000, 6000, 7500, 10000, 15000, 20000]
C_FRAC = 0.10                               # P08/P10 used c = b/10; kept

HEADLINE = dict(a=1000, b=5000)             # the spec everything else references

MIN_OBS = 30                                # same as P10
THIN_MULT = 2                               # same non-overlap thinning as P10
COST_BPS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]  # incl. sub-bp maker/rebate cases
THRESH_Q = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95]   # |x| quantile thresholds
N_BOOT = 1000
ET_OFFSET_H = -4                            # US/Eastern in September (EDT)

RNG = np.random.default_rng(20260914)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ==================================================================== naming

def label_to_effective(label: str) -> list[str] | None:
    if label in ["binance", "massive"] + MASSIVE_CONSTITUENTS:
        return [label]
    if label.startswith("massive_minus_"):
        counter = label[len("massive_minus_"):]
        if counter in MASSIVE_CONSTITUENTS:
            return [e for e in MASSIVE_CONSTITUENTS if e != counter]
    return None


def from_massive_name(massive_name: str, exchange: str) -> str | None:
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


def load_npy(dir_: Path, name: str):
    p = dir_ / f"{name}.npy"
    return np.load(p, mmap_mode="r") if p.exists() else None


def load_side(data_root: str, label: str, coin: str, date: str, a_list: list[int]):
    """ts_ms + vret_{a}/dt_{a}/imb_{a} for one side, concatenated over
    constituents and stable-sorted by ts, exactly as P08/P10 do."""
    eff = label_to_effective(label)
    if eff is None:
        return None
    dirs = []
    for exch in eff:
        cd = from_massive_name(coin, exch)
        if cd is None:
            return None
        d = ROOT / VWAP_DIR / data_root / exch / cd / date
        if not d.is_dir():
            return None
        dirs.append(d)
    keys = ["ts_ms"] + [f"{k}_{a}" for a in a_list for k in ("vret", "dt", "imb")]
    if len(dirs) == 1:
        out = {k: load_npy(dirs[0], k) for k in keys}
        return out if out.get("ts_ms") is not None else None
    ts = np.concatenate([np.asarray(load_npy(d, "ts_ms")) for d in dirs])
    order = None
    if len(ts) > 1 and not np.all(ts[1:] >= ts[:-1]):
        order = np.argsort(ts, kind="mergesort")      # stable, same as P08
    side = {"ts_ms": ts[order] if order is not None else ts}
    del ts
    for k in keys[1:]:
        parts = [load_npy(d, k) for d in dirs]
        if any(p is None for p in parts):
            return None
        arr = np.concatenate([np.asarray(p) for p in parts])
        side[k] = arr[order] if order is not None else arr
        del arr, parts
    return side


# ==================================================== matching (P08 rule, free b)

def match_lead_lag(lead_ts: np.ndarray, lag_ts: np.ndarray,
                   a: int, b: int, c: int) -> tuple[np.ndarray, np.ndarray]:
    """For each lead row i, the lag row nearest to t_i + b that also lies in
    [t_i + max(2a, b - c),  t_i + b + c].  Returns (lead_idx, lag_idx) for
    matched rows only, deduplicated so each lag row is claimed by the first
    (chronologically earliest) lead row -- P08's `no_duplications`.

    Identical window rule to P08 for b > 0; the only change is that b is not
    restricted to the four values P08 wrote index files for.

    b < 0 is supported for the placebo: the window is mirrored, so the lag row
    is searched at t_i + b (in the past) with the same 2a non-overlap floor
    applied on the other side.  Everything else is unchanged, which is the
    point -- the placebo must differ from the real test ONLY in the sign of b.
    """
    if len(lead_ts) == 0 or len(lag_ts) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    lead_ts = np.asarray(lead_ts, dtype=np.int64)
    lag_ts = np.asarray(lag_ts, dtype=np.int64)

    target = lead_ts + b
    if b >= 0:
        lo = lead_ts + max(2 * a, b - c)
        hi = lead_ts + b + c
    else:
        lo = lead_ts + (b - c)
        hi = lead_ts + min(-2 * a, b + c)
    if np.any(lo > hi):
        return np.empty(0, np.int64), np.empty(0, np.int64)

    p = np.searchsorted(lag_ts, target)
    c2 = np.clip(p, 0, len(lag_ts) - 1)
    c1 = np.clip(p - 1, 0, len(lag_ts) - 1)

    t1, t2 = lag_ts[c1], lag_ts[c2]
    ok1 = (t1 >= lo) & (t1 <= hi) & (p - 1 >= 0)
    ok2 = (t2 >= lo) & (t2 <= hi) & (p < len(lag_ts))
    # nearest of the two in-window candidates (two-pointer argument: if neither
    # immediate neighbour is in window, nothing else can be)
    d1 = np.where(ok1, np.abs(t1 - target), np.iinfo(np.int64).max)
    d2 = np.where(ok2, np.abs(t2 - target), np.iinfo(np.int64).max)
    pick = np.where(d1 <= d2, c1, c2)
    found = ok1 | ok2

    li = np.nonzero(found)[0]
    gi = pick[found]
    if len(gi) == 0:
        return li, gi
    # dedup: keep first lead row (lead_ts is sorted, so first index == earliest)
    _, keep = np.unique(gi, return_index=True)
    keep.sort()
    return li[keep], gi[keep]


def thin(lead_ts: np.ndarray, li: np.ndarray, a: int) -> np.ndarray:
    """P10's non-overlap thinning: first surviving row per THIN_MULT*a bucket."""
    if THIN_MULT <= 0 or len(li) == 0:
        return np.arange(len(li))
    t = np.asarray(lead_ts)[li]
    bucket = (t - t[0]) // (THIN_MULT * a)
    _, keep = np.unique(bucket, return_index=True)
    keep.sort()
    return keep


# ==================================================================== stats

def ols_stats(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    nan = dict(slope=np.nan, tstat=np.nan, pvalue=np.nan, r2=np.nan, corr=np.nan)
    if n < 3:
        return nan
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    dx, dy = x - x.mean(), y - y.mean()
    sxx, syy, sxy = float(dx @ dx), float(dy @ dy), float(dx @ dy)
    if sxx <= 0 or syy <= 0:
        return nan
    slope = sxy / sxx
    rss = max(syy - slope * sxy, 0.0)
    se = math.sqrt(rss / (n - 2) / sxx)
    t = slope / se if se > 0 else np.nan
    if sps is not None and np.isfinite(t):
        p = float(2 * sps.t.sf(abs(t), n - 2))
    else:
        p = np.nan
    return dict(slope=slope, tstat=t, pvalue=p, r2=1.0 - rss / syy,
                corr=sxy / math.sqrt(sxx * syy))


def one_sample_t(v: np.ndarray) -> tuple[float, float]:
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 3 or v.std(ddof=1) == 0:
        return np.nan, np.nan
    t = v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))
    p = float(2 * sps.t.sf(abs(t), len(v) - 1)) if sps is not None else np.nan
    return t, p


def bh_q(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg over the specs actually tested in THIS script.
    NOTE for the report: this is a much smaller family than P10's 1,440, so
    these q-values are NOT comparable to A1's.  A1's q is the honest one for
    'did anything survive the original search'; this one is for ranking
    within the already-selected focus set."""
    v = pd.to_numeric(p, errors="coerce")
    ok = v.notna()
    out = pd.Series(np.nan, index=p.index, dtype=float)
    if not ok.any():
        return out
    s = v[ok].sort_values()
    m = len(s)
    q = (s.to_numpy() * m / np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out.loc[s.index] = np.clip(q, 0, 1)
    return out


def block_bootstrap_mean(values: np.ndarray, groups: np.ndarray,
                         n_boot: int = N_BOOT) -> tuple[float, float]:
    """Resample whole DATES with replacement, not individual observations.
    Windows within a day are not independent; days plausibly are.  Returns a
    2.5/97.5 percentile CI for the mean of `values`."""
    values = np.asarray(values, np.float64)
    ok = np.isfinite(values)
    values, groups = values[ok], np.asarray(groups)[ok]
    if len(values) == 0:
        return np.nan, np.nan
    keys, inv = np.unique(groups, return_inverse=True)
    buckets = [values[inv == i] for i in range(len(keys))]
    if len(buckets) < 3:
        return np.nan, np.nan
    means = np.empty(n_boot)
    for k in range(n_boot):
        pick = RNG.integers(0, len(buckets), len(buckets))
        means[k] = np.concatenate([buckets[i] for i in pick]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ==================================================== per-(coin, date) core

def available_dates(data_root: str, label: str, coin: str) -> list[str]:
    eff = label_to_effective(label)
    if eff is None:
        return []
    sets = []
    for exch in eff:
        cd = from_massive_name(coin, exch)
        d = ROOT / VWAP_DIR / data_root / exch / (cd or "")
        sets.append({p.name for p in d.iterdir() if p.is_dir()} if d.is_dir() else set())
    return sorted(set.intersection(*sets)) if sets else []


def pair_series(lead: dict, lag: dict, a: int, b: int, c: int):
    """Matched, finite, thinned (x, y, lead_ts, lead_idx, lag_idx)."""
    li, gi = match_lead_lag(lead["ts_ms"], lag["ts_ms"], a, b, c)
    if len(li) == 0:
        return None
    xv, yv = lead[f"vret_{a}"], lag[f"vret_{a}"]
    inb = (li < len(xv)) & (gi < len(yv))
    li, gi = li[inb], gi[inb]
    x = np.asarray(xv[li], np.float64)
    y = np.asarray(yv[gi], np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    li, gi, x, y = li[ok], gi[ok], x[ok], y[ok]
    if len(x) == 0:
        return None
    keep = thin(lead["ts_ms"], li, a)
    li, gi, x, y = li[keep], gi[keep], x[keep], y[keep]
    return x, y, np.asarray(lead["ts_ms"])[li], li, gi


def day_row(x, y, ts, extra=None) -> dict:
    row = dict(n_used=len(x))
    if len(x) < MIN_OBS:
        return row
    row.update(ols_stats(x, y))
    sig = x != 0
    signed = np.sign(x[sig]) * y[sig]
    row["n_signals"] = int(sig.sum())
    row["signed_mean_bps"] = float(signed.mean() * 1e4) if sig.any() else np.nan
    both = sig & (y != 0)
    row["hit_rate"] = float((np.sign(x[both]) == np.sign(y[both])).mean()) if both.any() else np.nan
    row["mean_abs_x_bps"] = float(np.abs(x).mean() * 1e4)
    row["mean_abs_y_bps"] = float(np.abs(y).mean() * 1e4)
    if extra:
        row.update(extra)
    return row


# ==================================================================== stage: sweep

def stage_sweep(args) -> pd.DataFrame:
    """b-decay curve + per-coin, for every focus pair.  This is the file
    everything downstream reads."""
    rows = []
    for lead_l, lag_l, role in FOCUS_PAIRS:
        for coin in FOCUS_COINS:
            dates = sorted(set(available_dates(FOCUS_DATA_ROOT, lead_l, coin))
                           & set(available_dates(FOCUS_DATA_ROOT, lag_l, coin)))
            if not dates:
                log(f"[sweep] {lead_l}->{lag_l} {coin}: no common dates")
                continue
            for date in dates:
                lead = load_side(FOCUS_DATA_ROOT, lead_l, coin, date, FOCUS_A)
                lag = load_side(FOCUS_DATA_ROOT, lag_l, coin, date, FOCUS_A)
                if lead is None or lag is None:
                    continue
                for a in FOCUS_A:
                    for b in B_GRID:
                        if b <= 2 * a:          # P08's non-overlap constraint
                            continue
                        c = max(int(round(b * C_FRAC)), 1)
                        got = pair_series(lead, lag, a, b, c)
                        if got is None:
                            continue
                        x, y, ts, li, gi = got
                        r = day_row(x, y, ts)
                        rows.append(dict(coin=coin, date=date, lead=lead_l, lag=lag_l,
                                         role=role, a=a, b=b, c=c, variant="base", **r))
                del lead, lag
                gc.collect()
            log(f"[sweep] {lead_l}->{lag_l} {coin}: {len(dates)} dates done")
    df = pd.DataFrame(rows)
    save(df, "F0_daily.csv")
    return df


def aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per key group; the unit of observation is the COIN-DAY
    correlation, as in P10's agg_block -- never pooled raw windows."""
    out = []
    for k, g in df.groupby(keys, dropna=False):
        cor = g["corr"].to_numpy(float)
        t, p = one_sample_t(cor)
        ok = np.isfinite(cor)
        w = g["n_used"].to_numpy(float)
        rec = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        rec.update(
            n_coin_days=int(len(g)), n_fits=int(ok.sum()),
            n_used_sum=int(np.nansum(w)),
            corr_mean=float(np.nanmean(cor)) if ok.any() else np.nan,
            corr_wmean=float(np.nansum(cor[ok] * w[ok]) / np.nansum(w[ok])) if ok.any() else np.nan,
            corr_t=t, corr_t_p=p,
            frac_slope_pos=float((g["slope"] > 0).sum() / max(ok.sum(), 1)),
            hit_rate_mean=float(np.nanmean(g.get("hit_rate", np.nan))),
            signed_mean_bps=float(np.nanmean(g.get("signed_mean_bps", np.nan))),
            r2_mean=float(np.nanmean(g.get("r2", np.nan))),
        )
        lo, hi = block_bootstrap_mean(cor, g["date"].to_numpy(), n_boot=N_BOOT)
        rec["corr_boot_lo"], rec["corr_boot_hi"] = lo, hi
        out.append(rec)
    res = pd.DataFrame(out)
    if len(res):
        res["bh_q_focus"] = bh_q(res["corr_t_p"])
    return res


def stage_spec(df: pd.DataFrame):
    f1 = aggregate(df[df.variant == "base"], ["lead", "lag", "role", "a", "b"])
    save(f1.sort_values("corr_t_p"), "F1_spec.csv")

    h = df[(df.variant == "base") & (df.a == HEADLINE["a"]) & (df.b == HEADLINE["b"])]
    f2 = aggregate(h, ["lead", "lag", "role", "coin"])
    save(f2.sort_values(["lead", "lag", "coin"]), "F2_by_coin.csv")
    return f1, f2


# ==================================================================== stage: hours

def stage_hours(args) -> pd.DataFrame:
    """Hour-of-day at the headline spec.  P10's A4 only had the day-level
    US-open / US-closed split (clean_open_data vs clean_close_data); it found
    the effect larger in clean_close_data.  This localises that."""
    a, b = HEADLINE["a"], HEADLINE["b"]
    c = max(int(round(b * C_FRAC)), 1)
    rows = []
    for lead_l, lag_l, role in FOCUS_PAIRS:
        for coin in FOCUS_COINS:
            dates = sorted(set(available_dates(FOCUS_DATA_ROOT, lead_l, coin))
                           & set(available_dates(FOCUS_DATA_ROOT, lag_l, coin)))
            for date in dates:
                lead = load_side(FOCUS_DATA_ROOT, lead_l, coin, date, [a])
                lag = load_side(FOCUS_DATA_ROOT, lag_l, coin, date, [a])
                if lead is None or lag is None:
                    continue
                got = pair_series(lead, lag, a, b, c)
                if got is not None:
                    x, y, ts, li, gi = got
                    hr_utc = ((ts // 3_600_000) % 24).astype(int)
                    hr_et = (hr_utc + ET_OFFSET_H) % 24
                    for h in np.unique(hr_utc):
                        m = hr_utc == h
                        if m.sum() < MIN_OBS:
                            continue
                        r = day_row(x[m], y[m], ts[m])
                        rows.append(dict(coin=coin, date=date, lead=lead_l, lag=lag_l,
                                         role=role, hour_utc=int(h),
                                         hour_et=int((h + ET_OFFSET_H) % 24),
                                         us_equity_open=bool(9 <= (h + ET_OFFSET_H) % 24 < 16),
                                         **r))
                del lead, lag
                gc.collect()
    d = pd.DataFrame(rows)
    if len(d):
        agg = aggregate(d, ["lead", "lag", "role", "hour_utc", "hour_et", "us_equity_open"])
        save(agg.sort_values(["lead", "lag", "hour_utc"]), "F3_hours.csv")
        # and the open/closed contrast, paired by coin-day
        sig = d[d.role == "signal"]
        if len(sig):
            piv = (sig.groupby(["lead", "lag", "coin", "date", "us_equity_open"])["corr"]
                      .mean().unstack("us_equity_open"))
            if piv.shape[1] == 2:
                diff = (piv[True] - piv[False]).dropna()
                t, p = one_sample_t(diff.to_numpy())
                log(f"[hours] US-open minus US-closed corr, paired by coin-day: "
                    f"mean={diff.mean():+.4f}  t={t:.2f}  p={p:.3f}  n={len(diff)}")
    return d


# ==================================================================== stage: placebo

def stage_placebo(args) -> pd.DataFrame:
    """Four ways this should break if it is not a real lead-lag.

      reverse       lag venue leads.  Already in FOCUS_PAIRS as role=control.
      negb          b < 0: 'predict' the lag venue BEFORE the lead moved.  If
                    this is as strong as b>0, the result is contemporaneous
                    co-movement leaking through the window edges, not a lead.
      shuffle_day   lead returns from a DIFFERENT date, same clock time.  Kills
                    any real link; anything left is a within-day seasonal
                    artefact (both venues quiet at the same hours, etc).
      shuffle_coin  lead returns from a different COIN, same date/time.  Kills
                    the coin-specific link; anything left is market-wide beta.
    """
    a, b = HEADLINE["a"], HEADLINE["b"]
    c = max(int(round(b * C_FRAC)), 1)
    rows = []
    sig_pairs = [(l, g) for l, g, r in FOCUS_PAIRS if r == "signal"]

    for lead_l, lag_l in sig_pairs:
        for coin in FOCUS_COINS:
            dates = sorted(set(available_dates(FOCUS_DATA_ROOT, lead_l, coin))
                           & set(available_dates(FOCUS_DATA_ROOT, lag_l, coin)))
            if len(dates) < 2:
                continue
            for i, date in enumerate(dates):
                lag = load_side(FOCUS_DATA_ROOT, lag_l, coin, date, [a])
                if lag is None:
                    continue

                # --- negative b: same day, lead shifted the wrong way
                lead = load_side(FOCUS_DATA_ROOT, lead_l, coin, date, [a])
                if lead is not None:
                    for bb in (-b, -2 * b):
                        # window rule with |b|; the lag search simply looks back.
                        # c must scale with |bb| (not the headline b) so this
                        # differs from a genuine +|bb| test ONLY in the sign of b.
                        c_bb = max(int(round(abs(bb) * C_FRAC)), 1)
                        li, gi = match_lead_lag(lead["ts_ms"], lag["ts_ms"],
                                                a, bb, c_bb)
                        if len(li) >= MIN_OBS:
                            x = np.asarray(lead[f"vret_{a}"][li], float)
                            y = np.asarray(lag[f"vret_{a}"][gi], float)
                            m = np.isfinite(x) & np.isfinite(y)
                            keep = thin(lead["ts_ms"], li[m], a)
                            r = day_row(x[m][keep], y[m][keep],
                                        np.asarray(lead["ts_ms"])[li[m]][keep])
                            rows.append(dict(coin=coin, date=date, lead=lead_l,
                                             lag=lag_l, a=a, b=bb,
                                             variant=f"negb_{bb}", **r))
                    del lead

                # --- wrong-day lead (same clock time, previous available date)
                other = dates[i - 1] if i > 0 else dates[i + 1]
                lead_o = load_side(FOCUS_DATA_ROOT, lead_l, coin, other, [a])
                if lead_o is not None:
                    r = _cross_source(lead_o, lag, a, b, c, date, other)
                    if r:
                        rows.append(dict(coin=coin, date=date, lead=lead_l,
                                         lag=lag_l, a=a, b=b,
                                         variant="shuffle_day", **r))
                    del lead_o

                # --- wrong-coin lead (same date, another focus coin)
                other_coin = next((cc for cc in FOCUS_COINS if cc != coin), None)
                if other_coin:
                    lead_c = load_side(FOCUS_DATA_ROOT, lead_l, other_coin, date, [a])
                    if lead_c is not None:
                        r = _cross_source(lead_c, lag, a, b, c, date, date)
                        if r:
                            rows.append(dict(coin=coin, date=date, lead=lead_l,
                                             lag=lag_l, a=a, b=b,
                                             variant="shuffle_coin", **r))
                        del lead_c
                del lag
                gc.collect()

    d = pd.DataFrame(rows)
    if len(d):
        agg = aggregate(d, ["lead", "lag", "variant", "a", "b"])
        save(agg.sort_values(["lead", "lag", "variant"]), "F4_placebo.csv")
    return d


def _cross_source(lead: dict, lag: dict, a: int, b: int, c: int,
                  lag_date: str, lead_date: str) -> dict | None:
    """Match a lead side from a different day/coin onto this lag side by
    ALIGNING TIME-OF-DAY, not absolute timestamp."""
    lt = np.asarray(lead["ts_ms"], np.int64)
    gt = np.asarray(lag["ts_ms"], np.int64)
    if len(lt) == 0 or len(gt) == 0:
        return None
    day = 86_400_000
    shift = (gt[0] // day) * day - (lt[0] // day) * day
    lt_shift = lt + shift
    order = np.argsort(lt_shift, kind="mergesort")
    li, gi = match_lead_lag(lt_shift[order], gt, a, b, c)
    if len(li) < MIN_OBS:
        return None
    src = order[li]
    x = np.asarray(lead[f"vret_{a}"][src], float)
    y = np.asarray(lag[f"vret_{a}"][gi], float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < MIN_OBS:
        return None
    ts = lt_shift[order][li][m]
    keep = thin(ts, np.arange(m.sum()), a)
    return day_row(x[m][keep], y[m][keep], ts[keep])


# ==================================================================== stage: staleness

def stage_stale(args) -> pd.DataFrame:
    """THE most likely way this result dies.

    If Exchange23 prints rarely, the 'lag VWAP at t+b' may be built from trades
    that are themselves minutes old, in which case the lead venue is not
    predicting a future Exchange23 price -- it is predicting the arrival of a
    catch-up print at a price that was already public.  You cannot trade that:
    by the time the print happens the quote has moved.

    Diagnostics per coin-day at the headline spec:
      lag_dt_ratio   lag side's own gap between its two VWAP window centres / a.
                     ~1 means dense; >>1 means the 'window' spans far more than a.
      lag_idle_ms    median time since the lag venue's previous trade at match.
      corr_dense     the same correlation restricted to matches where the lag
                     side is dense (lag_dt_ratio <= 2).  If corr_dense collapses
                     toward zero, the effect IS staleness.
    """
    a, b = HEADLINE["a"], HEADLINE["b"]
    c = max(int(round(b * C_FRAC)), 1)
    rows = []
    for lead_l, lag_l, role in FOCUS_PAIRS:
        for coin in FOCUS_COINS:
            dates = sorted(set(available_dates(FOCUS_DATA_ROOT, lead_l, coin))
                           & set(available_dates(FOCUS_DATA_ROOT, lag_l, coin)))
            for date in dates:
                lead = load_side(FOCUS_DATA_ROOT, lead_l, coin, date, [a])
                lag = load_side(FOCUS_DATA_ROOT, lag_l, coin, date, [a])
                if lead is None or lag is None:
                    continue
                got = pair_series(lead, lag, a, b, c)
                if got is not None:
                    x, y, ts, li, gi = got
                    lag_dt = np.asarray(lag[f"dt_{a}"][gi], float) / a
                    gts = np.asarray(lag["ts_ms"])
                    prev = np.where(gi > 0, gts[np.maximum(gi - 1, 0)], np.nan)
                    idle = gts[gi].astype(float) - prev
                    dense = np.isfinite(lag_dt) & (lag_dt <= 2.0)
                    rec = dict(coin=coin, date=date, lead=lead_l, lag=lag_l, role=role,
                               n_used=len(x),
                               lag_dt_ratio_med=float(np.nanmedian(lag_dt)),
                               lag_dt_ratio_p90=float(np.nanpercentile(lag_dt, 90)),
                               lag_idle_ms_med=float(np.nanmedian(idle)),
                               frac_dense=float(dense.mean()),
                               corr_all=ols_stats(x, y)["corr"])
                    rec["corr_dense"] = (ols_stats(x[dense], y[dense])["corr"]
                                         if dense.sum() >= MIN_OBS else np.nan)
                    rec["corr_stale"] = (ols_stats(x[~dense], y[~dense])["corr"]
                                         if (~dense).sum() >= MIN_OBS else np.nan)
                    rows.append(rec)
                del lead, lag
                gc.collect()
    d = pd.DataFrame(rows)
    if len(d):
        g = (d.groupby(["lead", "lag", "role"])
               .agg(n_coin_days=("n_used", "size"),
                    lag_dt_ratio_med=("lag_dt_ratio_med", "median"),
                    lag_idle_ms_med=("lag_idle_ms_med", "median"),
                    frac_dense=("frac_dense", "mean"),
                    corr_all=("corr_all", "mean"),
                    corr_dense=("corr_dense", "mean"),
                    corr_stale=("corr_stale", "mean")).reset_index())
        for col in ("corr_all", "corr_dense", "corr_stale"):
            sub = d[col].to_numpy(float)
            g[f"{col}_t"] = np.nan
        for i, r in g.iterrows():
            m = (d.lead == r.lead) & (d.lag == r.lag)
            for col in ("corr_all", "corr_dense", "corr_stale"):
                g.loc[i, f"{col}_t"] = one_sample_t(d.loc[m, col].to_numpy(float))[0]
        save(g, "F5_staleness.csv")
    return d


# ==================================================================== stage: horizon

def stage_horizon(args) -> pd.DataFrame:
    """Same signal, lag return measured at b, 2b and 4b.

    Information transfer  -> the move at b persists (corr stays >= 0 at 2b/4b).
    Transient impact/noise -> it reverses (corr flips negative).  A reversal
    means any gross edge at b is not capturable: you would be buying the top of
    a temporary dislocation.
    """
    a, b0 = HEADLINE["a"], HEADLINE["b"]
    rows = []
    for lead_l, lag_l, role in FOCUS_PAIRS:
        if role != "signal":
            continue
        for coin in FOCUS_COINS:
            dates = sorted(set(available_dates(FOCUS_DATA_ROOT, lead_l, coin))
                           & set(available_dates(FOCUS_DATA_ROOT, lag_l, coin)))
            for date in dates:
                lead = load_side(FOCUS_DATA_ROOT, lead_l, coin, date, [a])
                lag = load_side(FOCUS_DATA_ROOT, lag_l, coin, date, [a])
                if lead is None or lag is None:
                    continue
                for mult in (1, 2, 4):
                    b = b0 * mult
                    c = max(int(round(b * C_FRAC)), 1)
                    got = pair_series(lead, lag, a, b, c)
                    if got is None:
                        continue
                    x, y, ts, li, gi = got
                    rows.append(dict(coin=coin, date=date, lead=lead_l, lag=lag_l,
                                     horizon_mult=mult, b=b,
                                     **day_row(x, y, ts)))
                del lead, lag
                gc.collect()
    d = pd.DataFrame(rows)
    if len(d):
        save(aggregate(d, ["lead", "lag", "horizon_mult", "b"]).sort_values(
            ["lead", "lag", "horizon_mult"]), "F6_horizon.csv")
    return d


# ==================================================================== stage: cost

def stage_cost(args) -> pd.DataFrame:
    """Threshold x cost surface.

    P10's A7 reported one gross number per spec (binance->Exchange23 @1000/5000:
    0.471 bps/signal, breakeven 0.471 bps).  That is the average over ALL
    signals including near-zero ones.  A real strategy only fires on large lead
    moves, so the relevant question is whether the edge grows faster than it
    loses signal count as the threshold rises.

    Reports, for each |x| quantile threshold:
      gross_bps, n_signals, signals_per_day, net at each COST_BPS,
      and a date-block bootstrap CI on gross.
    Round-trip cost reference: taker fees on major venues are ~2-10 bps
    round trip before spread; half-spread on Exchange23 is unknown from trade
    data alone and must be added.  0.5-1.0 bps is only reachable with passive
    execution on both legs, which this signal's 5-second horizon does not allow.
    """
    a, b = HEADLINE["a"], HEADLINE["b"]
    c = max(int(round(b * C_FRAC)), 1)
    rows = []
    for lead_l, lag_l, role in FOCUS_PAIRS:
        recs = []
        for coin in FOCUS_COINS:
            dates = sorted(set(available_dates(FOCUS_DATA_ROOT, lead_l, coin))
                           & set(available_dates(FOCUS_DATA_ROOT, lag_l, coin)))
            for date in dates:
                lead = load_side(FOCUS_DATA_ROOT, lead_l, coin, date, [a])
                lag = load_side(FOCUS_DATA_ROOT, lag_l, coin, date, [a])
                if lead is None or lag is None:
                    continue
                got = pair_series(lead, lag, a, b, c)
                if got is not None:
                    x, y, _, _, _ = got
                    m = x != 0
                    recs.append(pd.DataFrame(dict(coin=coin, date=date,
                                                  x=x[m], y=y[m])))
                del lead, lag
                gc.collect()
        if not recs:
            continue
        d = pd.concat(recs, ignore_index=True)
        del recs
        ax = np.abs(d["x"].to_numpy())
        signed = np.sign(d["x"].to_numpy()) * d["y"].to_numpy() * 1e4
        n_days = d.groupby(["coin", "date"]).ngroups
        for q in THRESH_Q:
            thr = np.quantile(ax, q) if q > 0 else 0.0
            m = ax >= thr
            if m.sum() < MIN_OBS:
                continue
            s = signed[m]
            lo, hi = block_bootstrap_mean(s, d["date"].to_numpy()[m])
            rec = dict(lead=lead_l, lag=lag_l, role=role, a=a, b=b,
                       thresh_q=q, thresh_bps=float(thr * 1e4),
                       n_signals=int(m.sum()),
                       signals_per_coin_day=float(m.sum() / max(n_days, 1)),
                       gross_bps=float(s.mean()),
                       gross_boot_lo=lo, gross_boot_hi=hi,
                       hit_rate=float((s > 0).mean()),
                       gross_t=one_sample_t(s)[0])
            for cst in COST_BPS:
                rec[f"net_bps_cost{cst}"] = float(s.mean() - cst)
            rec["breakeven_cost_bps"] = float(s.mean())
            rows.append(rec)
        del d
        gc.collect()
    out = pd.DataFrame(rows)
    if len(out):
        save(out.sort_values(["lead", "lag", "thresh_q"]), "F7_cost_curve.csv")
    return out


# ==================================================================== stage: oos

def stage_oos(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Split the date range in half.  Pick the single best (a, b) for each
    signal pair using ONLY the first half, then report that same spec's
    performance in the second half, with no further selection.

    With ~10-14 dates this is weak -- say so in the report.  It still catches
    the worst failure mode: a spec that only 'works' because it was chosen
    after seeing the whole sample.
    """
    d = df_daily[(df_daily.variant == "base") & (df_daily.role == "signal")].copy()
    if not len(d):
        return pd.DataFrame()
    dates = sorted(d["date"].unique())
    cut = dates[len(dates) // 2]
    d["half"] = np.where(d["date"] < cut, "IS", "OOS")
    rows = []
    for (lead_l, lag_l), g in d.groupby(["lead", "lag"]):
        is_ = aggregate(g[g.half == "IS"], ["a", "b"])
        oos = aggregate(g[g.half == "OOS"], ["a", "b"])
        if not len(is_) or not len(oos):
            continue
        best = is_.sort_values("corr_t", ascending=False).iloc[0]
        m = oos[(oos.a == best.a) & (oos.b == best.b)]
        rows.append(dict(
            lead=lead_l, lag=lag_l, cut_date=cut,
            a_selected=int(best.a), b_selected=int(best.b),
            is_corr_mean=best.corr_mean, is_corr_t=best.corr_t,
            is_n_coin_days=best.n_coin_days,
            oos_corr_mean=float(m.corr_mean.iloc[0]) if len(m) else np.nan,
            oos_corr_t=float(m.corr_t.iloc[0]) if len(m) else np.nan,
            oos_signed_bps=float(m.signed_mean_bps.iloc[0]) if len(m) else np.nan,
            oos_n_coin_days=int(m.n_coin_days.iloc[0]) if len(m) else 0,
            headline_oos_corr=float(
                oos[(oos.a == HEADLINE["a"]) & (oos.b == HEADLINE["b"])].corr_mean.iloc[0])
            if len(oos[(oos.a == HEADLINE["a"]) & (oos.b == HEADLINE["b"])]) else np.nan,
        ))
    out = pd.DataFrame(rows)
    if len(out):
        save(out, "F8_oos.csv")
    return out


# ==================================================================== output

def save(df: pd.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / name, index=False)
    log(f"[save] {name} ({len(df)} rows)")


def write_summary(f1, f2, f3, f4, f5, f6, f7, f8):
    lines = ["# Focus investigation: the Exchange23 lag effect", ""]

    def block(title, df, cols, sort=None, n=25):
        lines.append(f"## {title}")
        if df is None or not len(df):
            lines.append("_no rows_\n"); return
        t = df.copy()
        cols = [c for c in cols if c in t.columns]
        if sort:
            t = t.sort_values([c for c in sort if c in t.columns])
        lines.append(t[cols].head(n).to_markdown(index=False))
        lines.append("")

    block("F1 b-decay by pair", f1,
          ["lead", "lag", "role", "a", "b", "n_coin_days", "corr_mean",
           "corr_boot_lo", "corr_boot_hi", "corr_t", "bh_q_focus"],
          ["role", "lead", "a", "b"], 60)
    block("F2 per coin at headline spec", f2,
          ["lead", "lag", "role", "coin", "n_coin_days", "corr_mean",
           "corr_t", "corr_t_p", "hit_rate_mean", "signed_mean_bps"],
          ["role", "lead", "coin"], 40)
    block("F3 hour of day (signal pairs)",
          f3[f3.role == "signal"] if f3 is not None and len(f3) else f3,
          ["lead", "lag", "hour_utc", "hour_et", "us_equity_open",
           "n_coin_days", "corr_mean", "corr_t", "signed_mean_bps"],
          ["lead", "hour_utc"], 100)
    block("F4 placebos (all should be flat)", f4,
          ["lead", "lag", "variant", "b", "n_coin_days", "corr_mean",
           "corr_t", "corr_t_p"], ["lead", "variant"], 40)
    block("F5 lag-side staleness", f5,
          ["lead", "lag", "role", "lag_dt_ratio_med", "lag_idle_ms_med",
           "frac_dense", "corr_all", "corr_dense", "corr_dense_t",
           "corr_stale"], ["role", "lead"], 20)
    block("F6 continuation vs reversal", f6,
          ["lead", "lag", "horizon_mult", "b", "corr_mean", "corr_t",
           "signed_mean_bps"], ["lead", "horizon_mult"], 30)
    block("F7 cost curve", f7,
          ["lead", "lag", "role", "thresh_q", "thresh_bps", "n_signals",
           "signals_per_coin_day", "gross_bps", "gross_boot_lo",
           "gross_boot_hi", "hit_rate", "net_bps_cost2", "net_bps_cost5"],
          ["role", "lead", "thresh_q"], 60)
    block("F8 out of sample", f8,
          ["lead", "lag", "cut_date", "a_selected", "b_selected",
           "is_corr_mean", "is_corr_t", "oos_corr_mean", "oos_corr_t",
           "oos_signed_bps", "headline_oos_corr"], None, 20)

    lines += [
        "## How to read this",
        "",
        "The result stands only if ALL of these hold:",
        "",
        "1. F1: signal pairs have corr_boot_lo > 0; control pairs (Exchange23 as",
        "   lead) do not. The b-decay should fall monotonically to zero by",
        "   b ~ 15-20s. A flat curve means a spurious common factor.",
        "2. F2: the effect is present in more than one coin. If it is BTC only,",
        "   the effective sample is ~10-14 observations and the t-stats in P10's",
        "   A1 are pooling coin-days that are not independent.",
        "3. F4: every placebo near zero. negb in particular -- if negative b is",
        "   as strong as positive b, this is contemporaneous co-movement bleeding",
        "   through the window edges, not a lead-lag.",
        "4. F5: corr_dense should be comparable to corr_all. If corr_dense",
        "   collapses, the effect is Exchange23 printing stale prices and there",
        "   is nothing to trade.",
        "5. F6: corr at 2b/4b should decay toward zero from above, not flip",
        "   negative. A negative sign at 2b means reversal.",
        "6. F7: net_bps must clear a realistic round-trip cost. Taker fees alone",
        "   are ~2-10 bps round trip; Exchange23's spread is not observable from",
        "   trade data and must be added on top. If the honest answer is that",
        "   nothing clears 2 bps, say so -- the brief treats 'exists gross, not",
        "   tradable' as a valid conclusion.",
        "7. F8: OOS corr should keep the sign and order of magnitude.",
        "",
        "Also note for the report: bh_q_focus here is over ~200 specs, not",
        "P10's 1,440. A1's bh_q is the correct number to quote for 'did the",
        "original search find anything'; this one only ranks within the focus set.",
        "",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "focus_summary.md").write_text("\n".join(lines))
    log(f"[save] focus_summary.md")


# ==================================================================== main

def main():
    global N_BOOT, ROOT, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--stage", default="all",
                    help="comma list: sweep,hours,placebo,stale,horizon,cost,oos")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    if args.root:
        ROOT = Path(args.root)
        OUT_DIR = ROOT / RESULT_DIR / "focus"
    N_BOOT = args.boot

    stages = ([s.strip() for s in args.stage.split(",")]
              if args.stage != "all" else
              ["sweep", "hours", "placebo", "stale", "horizon", "cost", "oos"])

    daily_path = OUT_DIR / "F0_daily.csv"
    df = None
    if "sweep" in stages:
        df = stage_sweep(args)
    elif daily_path.exists():
        df = pd.read_csv(daily_path, dtype={"date": str})

    f1 = f2 = f3 = f4 = f5 = f6 = f7 = f8 = None
    if df is not None and len(df):
        f1, f2 = stage_spec(df)
    if "hours" in stages:
        f3 = stage_hours(args)
        if f3 is not None and len(f3):
            f3 = aggregate(f3, ["lead", "lag", "role", "hour_utc", "hour_et",
                                "us_equity_open"])
    if "placebo" in stages:
        d4 = stage_placebo(args)
        f4 = aggregate(d4, ["lead", "lag", "variant", "a", "b"]) if len(d4) else None
    if "stale" in stages:
        d5 = stage_stale(args)
        f5 = pd.read_csv(OUT_DIR / "F5_staleness.csv") if (OUT_DIR / "F5_staleness.csv").exists() else None
    if "horizon" in stages:
        d6 = stage_horizon(args)
        f6 = aggregate(d6, ["lead", "lag", "horizon_mult", "b"]) if len(d6) else None
    if "cost" in stages:
        f7 = stage_cost(args)
    if "oos" in stages and df is not None and len(df):
        f8 = stage_oos(df)

    write_summary(f1, f2, f3, f4, f5, f6, f7, f8)
    log("focus run complete -> " + str(OUT_DIR))


if __name__ == "__main__":
    main()