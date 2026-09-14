#!/usr/bin/env python3
"""
P10_vwap_pipeline.py

The analysis layer on top of P06-P09: non-overlapping-subsample regressions
of lag VWAP return on lead VWAP return, plus the A1-A8 study tables.


What this script no longer does
-------------------------------
The previous version carried a full second implementation of the pipeline:

  * its own `vwap` stage, which re-read every trade file, rebuilt the prefix
    sums, recomputed VWAP(W2)/VWAP(W1)-1 and the taker imbalance, and wrote
    them to a private `vwap_return/` .npy store.  It existed because P07 was
    then computing returns against the raw trade print rather than the
    window VWAP, so P10 could not use P07's output.  P07 is fixed, and P07
    now applies the same `5_trades` gate P10 always enforced, so the two
    computations produce the same number.  `vret_{a}` is `ret_{a}`,
    `dt_{a}` is `dt_{a}`, and the imbalance now comes from P06 as
    `a_{a}_imb` -- computed there off prefix sums it was building anyway.
    The whole stage, its cache, its `vwap_return/` tree and the
    `--csv`/`LAZY_ROWS`/npy-memmap machinery are gone.

  * its own copies of `effective_exchanges`, `side_label`,
    `label_to_effective`, `to_massive_name`, `from_massive_name`,
    `label_pairs`, `count_data_rows`, `read_columns`,
    `write_columns_chunked`, `parse_index_name`, `read_index`,
    `load_done_keys` and `ols_stats`.  Every one of those existed verbatim
    in P06, P08 or P09.  They are imported now.

What is left here is what is genuinely only P10's: the non-overlap thinning,
the tradability statistics, the imbalance regression, and the A1-A8 study.


Stages
------
  volume   optional: sum price*qty per coin from the raw trade files, as a
           notional-volume proxy for A3 when neither ols_result.csv nor
           volume_0830_0905.csv exists.
  regress  regression_index/ + linear_return/
           -> regression_result_vwap/{data_root}/regressions.csv
           one row per (coin, date, lead, lag, a, b, c): OLS of lag return on
           lead return, correlation, hit rate, signed mean return (gross P&L
           per signal in bps), and OLS on the lead taker imbalance.  Main
           stats use a NON-OVERLAPPING subsample (first valid lead row per
           THIN_MULT*a bucket); corr_all / tstat_all / n_all are the
           overlapping full-sample numbers, directly comparable to P09's but
           with the same overstated significance.
  analyze  regression_result_vwap/analysis/*.csv + summary.md
           A1 spec summary (+ one-sample t-test on daily corr, BH q-values)
           A2 volume-tier summary            A3 tier / notional rank test
           A4 US-open vs US-close (paired)   A5 persistence over b
           A6 direction (B->M vs M->B)       A7 tradability after costs
           A8 coverage


Layout (all under ROOT = $LEADLAG_ROOT, default ~)
--------------------------------------------------
  {ROOT}/{clean_data,clean_open_data,clean_close_data}/{exch}/{coin}/{date}.csv
  {ROOT}/linear_vwap_construction/{data_root}/{exch}/{coin}/{date}.csv   P06
  {ROOT}/linear_return/{data_root}/{exch}/{coin}/{date}.csv              P07
  {ROOT}/regression_index/{data_root}/{lead}/{coin}/{date}/...csv        P08
  {ROOT}/coins.csv       M Tick Name, B Tick Name, NT Tier
  {ROOT}/ols_result.csv  Tick Name, Notional Volume


Stale results
-------------
Result rows carry `pipeline_version` (P09.RESULT_VERSION).  A regressions
file written under a different version -- in particular one built from the
old print-referenced returns -- is renamed aside on startup rather than
being topped up, because the resume key (coin, date, lead, lag, a, b, c)
says nothing about how the returns underneath it were computed.


Usage
-----
  python P10_vwap_pipeline.py                      # regress + analyze
  python P10_vwap_pipeline.py --stage regress --pairs primary
  python P10_vwap_pipeline.py --stage analyze
  python P10_vwap_pipeline.py --root /path/to/project
"""

from __future__ import annotations

import argparse
import gc
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
except Exception:
    sps = None

import P06_linear_VWAP as P06
import P08_regression_assign as P08
import P09_regression_run as P09
from P06_linear_VWAP import get_root, read_columns, set_root
from P08_regression_assign import (
    DATA_ROOTS, EXCHANGES, INDEX_DIR, label_to_effective, to_massive_name,
    valid_label_pairs,
)
from P09_regression_run import (
    append_results, load_done_keys, load_side_returns, ols_stats,
    parse_idx_filename, quarantine_stale_results, read_index, select_pairs,
)


# ============================================================ config

A_VALUES = list(P06.VWAP_INTERVALS)

RESULT_DIR = "regression_result_vwap"
VOLUME_FILE = "computed_notional_volume.csv"

MIN_OBS = P09.MIN_OBS
THIN_MULT = 2                 # keep the first valid lead row per THIN_MULT*a ms
                              # bucket, so kept windows are >= a apart (0 = off)
MIN_TRADES_TRADABILITY = 30
COST_BPS = [0, 2, 5, 10, 20]  # round-trip cost scenarios (bps) for A7

RESULT_COLS = [
    "coin", "coin_label", "date", "lead", "lag", "a", "b", "c",
    "n_lead", "n_matched", "n_all", "n_used",
    "slope", "intercept", "tstat", "pvalue", "r2", "corr",
    "corr_all", "tstat_all",
    "hit_rate", "signed_mean_bps", "signed_mean_bps_top50", "mean_abs_y_bps",
    "n_signals", "mean_dt_ratio",
    "imb_n", "imb_slope", "imb_tstat", "imb_r2", "imb_corr",
    "pipeline_version",
]

_IMB_RE = re.compile(r"^a_(\d+)_imb$")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _wanted_column(c: str) -> bool:
    """Columns P10 needs out of a linear_return file."""
    return (c.startswith("ret_") or c.startswith("dt_")
            or bool(_IMB_RE.match(c)))


def load_side(data_root: str, exchanges: list[str],
              massive_name: str, date: str):
    """One side's return/dt/imbalance frame, in P08's row order.

    Thin wrapper over P09.load_side_returns -- the concatenation and stable
    sort for a `massive_minus_*` side live there, once, instead of being
    reimplemented here against a parallel .npy store.
    """
    return load_side_returns(data_root, exchanges, massive_name, date,
                             col_filter=_wanted_column, dtype="float32")


# ============================================================ stage: regress

def regress_one(lead: pd.DataFrame, lag: pd.DataFrame,
                idx: dict, a: int) -> dict | None:
    """Full-sample and thinned regressions plus tradability stats."""
    rcol = f"ret_{a}"
    if rcol not in lead.columns or rcol not in lag.columns:
        return None

    lead_ret = lead[rcol].to_numpy(dtype=np.float64)
    lag_ret = lag[rcol].to_numpy(dtype=np.float64)

    x, y, li, _gi, n_lead, n_matched = select_pairs(
        lead_ret, lag_ret, idx["lead"], idx["lag"], idx["found"], idx["nodup"])
    n_all = int(len(x))

    row: dict = {"n_lead": n_lead, "n_matched": n_matched, "n_all": n_all}
    if n_matched == 0:
        return None
    if n_all >= 3:
        s_all = ols_stats(x, y)
        row["corr_all"], row["tstat_all"] = s_all["corr"], s_all["tstat"]

    # ---- non-overlap thinning ------------------------------------------
    # Consecutive lead rows are milliseconds apart, so their VWAP windows
    # overlap almost completely and full-sample t-stats are badly overstated.
    # Keep the first surviving lead row in each bucket of width THIN_MULT*a.
    if THIN_MULT > 0 and n_all and "_ts" in lead.columns:
        t = lead["_ts"].to_numpy()[li]
        _, keep_i = np.unique((t - t[0]) // (THIN_MULT * a), return_index=True)
        x, y, li = x[keep_i], y[keep_i], li[keep_i]

    n_used = int(len(x))
    row["n_used"] = n_used
    if n_used < MIN_OBS:
        return row

    row.update(ols_stats(x, y))

    sig = x != 0
    signed = np.sign(x[sig]) * y[sig]
    row["n_signals"] = int(sig.sum())
    row["signed_mean_bps"] = float(signed.mean() * 1e4) if sig.any() else np.nan
    if sig.sum() >= 4:
        strong = np.abs(x[sig]) >= np.median(np.abs(x[sig]))
        row["signed_mean_bps_top50"] = float(signed[strong].mean() * 1e4)
    else:
        row["signed_mean_bps_top50"] = np.nan
    both = sig & (y != 0)
    row["hit_rate"] = (float((np.sign(x[both]) == np.sign(y[both])).mean())
                       if both.any() else np.nan)
    row["mean_abs_y_bps"] = float(np.abs(y).mean() * 1e4)

    dcol = f"dt_{a}"
    if dcol in lead.columns:
        dt = lead[dcol].to_numpy(dtype=np.float64)[li]
        dt = dt[np.isfinite(dt)]
        row["mean_dt_ratio"] = float(dt.mean() / a) if len(dt) else np.nan

    icol = f"a_{a}_imb"
    if icol in lead.columns:
        imb = lead[icol].to_numpy(dtype=np.float64)[li]
        okb = np.isfinite(imb)
        row["imb_n"] = int(okb.sum())
        if okb.sum() >= MIN_OBS:
            s = ols_stats(imb[okb], y[okb])
            row.update(imb_slope=s["slope"], imb_tstat=s["tstat"],
                       imb_r2=s["r2"], imb_corr=s["corr"])
    return row


def stage_regress(pairs: list[tuple[str, str]], roots: list[str]) -> None:
    for data_root in roots:
        index_root = get_root() / INDEX_DIR / data_root
        if not index_root.exists():
            log(f"[regress] {index_root} missing, skip")
            continue

        out_file = get_root() / RESULT_DIR / data_root / "regressions.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        quarantine_stale_results(out_file, version=P09.RESULT_VERSION,
                                 siblings=())
        done = load_done_keys(out_file)
        has_header = out_file.exists() and out_file.stat().st_size > 0
        log(f"[regress] {data_root}: {len(done):,} rows already done")

        for lead_label, lag_label in pairs:
            lead_dir = index_root / lead_label
            lead_eff = label_to_effective(lead_label)
            lag_eff = label_to_effective(lag_label)
            if not lead_dir.exists() or lead_eff is None or lag_eff is None:
                continue

            for coin_dir in sorted(p for p in lead_dir.iterdir() if p.is_dir()):
                coin_label = coin_dir.name
                massive_name = (coin_label
                                if lead_label.startswith("massive_minus_")
                                else to_massive_name(coin_label, lead_label))
                if massive_name is None:
                    continue

                for date_dir in sorted(p for p in coin_dir.iterdir() if p.is_dir()):
                    date = date_dir.name
                    todo = []
                    for f in sorted(date_dir.glob(f"{lead_label}_{lag_label}_*.csv")):
                        parsed = parse_idx_filename(f.name, pairs)
                        if parsed is None:
                            continue
                        p_lead, p_lag, a, b, c = parsed
                        if p_lead != lead_label or p_lag != lag_label:
                            continue
                        if (massive_name, date, lead_label, lag_label,
                                a, b, c) not in done:
                            todo.append((f, (a, b, c)))
                    if not todo:
                        continue

                    lead = load_side(data_root, lead_eff, massive_name, date)
                    lag = load_side(data_root, lag_eff, massive_name, date)
                    if lead is None or lag is None:
                        log(f"[regress] {data_root} {massive_name}/{date} "
                            f"{lead_label}->{lag_label}: return files missing, skip")
                        continue

                    rows = []
                    for f, (a, b, c) in todo:
                        idx = read_index(f, lead_label, lag_label)
                        if idx is None:
                            continue
                        r = regress_one(lead, lag, idx, a)
                        del idx
                        if r is None:
                            continue
                        rows.append({"coin": massive_name,
                                     "coin_label": coin_label, "date": date,
                                     "lead": lead_label, "lag": lag_label,
                                     "a": a, "b": b, "c": c,
                                     "pipeline_version": P09.RESULT_VERSION,
                                     **r})
                        done.add((massive_name, date, lead_label, lag_label,
                                  a, b, c))
                    del lead, lag
                    gc.collect()

                    if rows:
                        has_header = append_results(out_file, rows, has_header,
                                                    columns=RESULT_COLS)
                        log(f"[regress] {data_root} {massive_name}/{date} "
                            f"{lead_label}->{lag_label}  +{len(rows)}")
        log(f"[regress] {data_root} done -> {out_file}")


# ============================================================ stage: volume

def stage_volume(roots: list[str]) -> None:
    """Sum price*qty per canonical coin straight from the raw trade files.

    Only reads one data root (clean_data by default): the open/close roots
    are the same trades restricted to a narrower window, so summing all three
    would double-count.
    """
    use_roots = [r for r in roots if r == "clean_data"] or roots[:1]
    if not use_roots:
        log("[volume] no data roots given, nothing to do")
        return
    if use_roots != ["clean_data"]:
        log(f"[volume] 'clean_data' not in --roots; using {use_roots} instead "
            f"(one root only, to avoid double-counting trades that also "
            f"appear in the open/close subsets)")

    notional: dict[str, float] = {}
    n_trades: dict[str, int] = {}
    n_dates: dict[str, set] = {}

    for data_root in use_roots:
        for exch in EXCHANGES:
            base = get_root() / data_root / exch
            if not base.exists():
                continue
            for coin_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                massive_name = to_massive_name(coin_dir.name, exch)
                if massive_name is None:
                    continue
                for f in sorted(coin_dir.glob("*.csv")):
                    cols = read_columns(f, {"price": "float64",
                                            "quantity": "float64",
                                            "size": "float64"})
                    qty = cols.get("quantity", cols.get("size"))
                    if "price" not in cols or qty is None or len(qty) == 0:
                        continue
                    notional[massive_name] = notional.get(massive_name, 0.0) + \
                        float(np.nansum(cols["price"] * qty))
                    n_trades[massive_name] = n_trades.get(massive_name, 0) + \
                        len(cols["price"])
                    n_dates.setdefault(massive_name, set()).add(f.stem)
                    del cols, qty

    if not notional:
        log(f"[volume] no trade files found under "
            f"{[str(get_root() / r) for r in use_roots]}; nothing written")
        return

    out = pd.DataFrame([{"Tick Name": name.replace("_", ":", 1),
                         "Notional Volume": notional[name],
                         "n_trades": n_trades[name],
                         "n_dates": len(n_dates[name])}
                        for name in sorted(notional)]
                       ).sort_values("Notional Volume", ascending=False)
    out_path = get_root() / VOLUME_FILE
    out.to_csv(out_path, index=False, float_format="%.6g")
    log(f"[volume] wrote {out_path}: {len(out)} coins, "
        f"{sum(n_trades.values()):,} trades total, from {use_roots}")


# ============================================================ stage: analyze

def load_coin_meta() -> pd.DataFrame:
    """coin (X_BTCUSD) -> tier, notional."""
    rows = []
    cpath = get_root() / "coins.csv"
    if cpath.exists():
        c = pd.read_csv(cpath)
        for _, r in c.iterrows():
            rows.append({"coin": str(r["M Tick Name"]).replace(":", "_"),
                         "tier": int(r["NT Tier"])})
    else:
        fallback = ["BTC", "ETH", "XRP", "SYRUP", "SEI", "RENDER", "SUPER",
                    "AUCTION", "XTZ", "GLM", "BLUR", "BIGTIME", "ADX", "LSK",
                    "CHR"]
        rows = [{"coin": f"X_{b}USD", "tier": i // 3 + 1}
                for i, b in enumerate(fallback)]

    meta = pd.DataFrame(rows).drop_duplicates("coin")
    meta["notional"] = np.nan

    notional_src = None
    for cand in ("ols_result.csv", "volume_0830_0905.csv", VOLUME_FILE):
        p = get_root() / cand
        if p.exists():
            v = pd.read_csv(p, usecols=["Tick Name", "Notional Volume"])
            v["coin"] = v["Tick Name"].str.replace(":", "_", regex=False)
            meta = meta.drop(columns="notional").merge(
                v[["coin", "Notional Volume"]].rename(
                    columns={"Notional Volume": "notional"}),
                on="coin", how="left")
            notional_src = cand
            break

    meta["log_notional"] = np.log(meta["notional"].where(meta["notional"] > 0))

    n_have = int(meta["log_notional"].notna().sum())
    if notional_src is None:
        log(f"[analyze] coin_meta: no notional-volume file found (looked for "
            f"ols_result.csv, volume_0830_0905.csv, {VOLUME_FILE} under "
            f"{get_root()}); A3's log_notional test will be skipped, tier only")
    elif n_have == 0:
        log(f"[analyze] coin_meta: WARNING found {notional_src} but 0/{len(meta)} "
            f"coins matched a positive Notional Volume -- check its 'Tick Name' "
            f"values against coin ids like 'X_BTCUSD' (colon vs underscore, "
            f"casing, suffix); A3's log_notional test will be skipped")
    else:
        log(f"[analyze] coin_meta: {n_have}/{len(meta)} coins have notional "
            f"volume from {notional_src}")
    return meta


def load_results(roots: list[str]) -> pd.DataFrame:
    frames = []
    for dr in roots:
        p = get_root() / RESULT_DIR / dr / "regressions.csv"
        if p.exists() and p.stat().st_size > 0:
            df = pd.read_csv(p, dtype={"date": str})
            df["data_root"] = dr
            frames.append(df)
    if not frames:
        raise SystemExit("no regression results found; run --stage regress first")
    df = pd.concat(frames, ignore_index=True)
    df["fit_ok"] = df["slope"].notna()
    df["sig05"] = (df["pvalue"] < 0.05).astype(float).where(df["fit_ok"])
    df["slope_pos"] = (df["slope"] > 0).astype(float).where(df["fit_ok"])
    return df


def one_sample_t(v: pd.Series):
    v = pd.Series(v).dropna().to_numpy(dtype=float)
    n = len(v)
    if n < 3 or v.std(ddof=1) == 0:
        return n, np.nan, np.nan
    t = v.mean() / (v.std(ddof=1) / math.sqrt(n))
    p = float(2 * sps.t.sf(abs(t), n - 1)) if sps is not None else np.nan
    return n, t, p


def bh_qvalues(p: pd.Series) -> pd.Series:
    q = pd.Series(np.nan, index=p.index)
    m = p.notna()
    pv = p[m].to_numpy()
    k = len(pv)
    if k == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * k / (np.arange(k) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(k)
    out[order] = np.clip(ranked, 0, 1)
    q[m] = out
    return q


SPEC = ["data_root", "lead", "lag", "a", "b", "c"]


def agg_block(g: pd.DataFrame) -> dict:
    fit = g[g["fit_ok"]]
    n, t, p = one_sample_t(fit["corr"])
    w = fit["n_used"].to_numpy(dtype=float)
    pooled = float((fit["corr"] * w).sum() / w.sum()) if w.sum() > 0 else np.nan
    return {
        "n_coin_days": int(len(g)), "n_fits": int(len(fit)),
        "n_used_sum": int(g["n_used"].fillna(0).sum()),
        "corr_mean": fit["corr"].mean(), "corr_median": fit["corr"].median(),
        "corr_wmean": pooled, "corr_t": t, "corr_t_p": p,
        "r2_mean": fit["r2"].mean(), "r2_median": fit["r2"].median(),
        "slope_mean": fit["slope"].mean(), "slope_median": fit["slope"].median(),
        "tstat_mean": fit["tstat"].mean(),
        "frac_sig05": fit["sig05"].mean(), "frac_slope_pos": fit["slope_pos"].mean(),
        "hit_rate_mean": fit["hit_rate"].mean(),
        "imb_corr_mean": fit["imb_corr"].mean(),
        "imb_tstat_mean": fit["imb_tstat"].mean(),
        "mean_dt_ratio": fit["mean_dt_ratio"].mean(),
    }


def groupped(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys, sort=True):
        k = k if isinstance(k, tuple) else (k,)
        rows.append({**dict(zip(keys, k)), **agg_block(g)})
    return pd.DataFrame(rows)


def stage_analyze(roots: list[str]) -> None:
    out_dir = get_root() / RESULT_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(roots)
    df = df.merge(load_coin_meta(), on="coin", how="left")
    log(f"[analyze] {len(df):,} daily regressions, {df['coin'].nunique()} coins, "
        f"{df['date'].nunique()} dates")

    def save(name, table):
        table.to_csv(out_dir / name, index=False, float_format="%.6g")
        log(f"[analyze] saved {name} ({len(table)} rows)")

    # ---- A1: per-spec summary + multiple testing
    a1 = groupped(df, SPEC)
    a1["bh_q"] = bh_qvalues(a1["corr_t_p"])
    save("A1_spec_summary.csv", a1)

    # ---- A2: by volume tier
    d_t = df[df["tier"].notna()].copy()
    if len(d_t):
        d_t["tier"] = d_t["tier"].astype(int)
        save("A2_tier_by_spec.csv", groupped(d_t, SPEC + ["tier"]))
        save("A2_tier_pooled.csv",
             groupped(d_t, ["data_root", "lead", "lag", "tier"]))

    # ---- A3: does predictability rise as coin volume falls?
    rows = []
    for k, g in df[df["fit_ok"]].groupby(SPEC):
        per_coin = g.groupby("coin").agg(
            corr=("corr", "mean"), r2=("r2", "mean"), tier=("tier", "first"),
            log_notional=("log_notional", "first")).dropna(subset=["corr"])
        rec = dict(zip(SPEC, k))
        rec["n_coins"] = len(per_coin)
        for xcol in ("log_notional", "tier"):
            sub = per_coin.dropna(subset=[xcol])
            if len(sub) >= 5 and sps is not None and sub[xcol].nunique() > 1:
                for ycol in ("corr", "r2"):
                    rho, p = sps.spearmanr(sub[xcol], sub[ycol])
                    rec[f"spearman_{ycol}_vs_{xcol}"] = rho
                    rec[f"p_{ycol}_vs_{xcol}"] = p
        rows.append(rec)
    a3 = pd.DataFrame(rows)
    save("A3_tier_rank_tests.csv", a3)

    # ---- A4: US-open vs US-close, paired by coin within spec
    rows = []
    both = df[df["fit_ok"] & df["data_root"].isin(
        ["clean_open_data", "clean_close_data"])]
    for k, g in both.groupby(["lead", "lag", "a", "b", "c"]):
        pc = g.groupby(["coin", "data_root"])[
            ["corr", "r2", "signed_mean_bps"]].mean().unstack("data_root")
        rec = dict(zip(["lead", "lag", "a", "b", "c"], k))
        for m in ("corr", "r2", "signed_mean_bps"):
            if ("clean_open_data" in pc[m].columns
                    and "clean_close_data" in pc[m].columns):
                pair = pc[m].dropna()
                diff = pair["clean_open_data"] - pair["clean_close_data"]
                rec[f"{m}_open"] = pair["clean_open_data"].mean()
                rec[f"{m}_close"] = pair["clean_close_data"].mean()
                rec[f"{m}_open_minus_close"] = diff.mean()
                n, t, p = one_sample_t(diff)
                rec[f"{m}_n_coins"] = n
                rec[f"{m}_paired_t"] = t
                rec[f"{m}_paired_p"] = p
                if sps is not None and n >= 6 and (diff != 0).any():
                    try:
                        rec[f"{m}_wilcoxon_p"] = float(sps.wilcoxon(diff).pvalue)
                    except Exception:
                        pass
        rows.append(rec)
    if rows:
        save("A4_open_vs_close.csv", pd.DataFrame(rows))
        save("A4_open_vs_close_by_tier.csv",
             groupped(both[both["tier"].notna()],
                      ["data_root", "lead", "lag", "tier"]))

    # ---- A5: persistence over the lag b
    a5 = a1.pivot_table(index=["data_root", "lead", "lag", "a"], columns="b",
                        values="corr_mean").reset_index()
    a5.columns = [f"corr_b{c}" if isinstance(c, (int, np.integer)) else c
                  for c in a5.columns]
    save("A5_persistence_over_b.csv", a5)

    # ---- A6: direction, per coin & spec
    d6 = df[df["fit_ok"]
            & (((df["lead"] == "binance") & (df["lag"] == "massive"))
               | ((df["lead"] == "massive") & (df["lag"] == "binance")))]
    if len(d6):
        d6 = d6.assign(direction=np.where(d6["lead"] == "binance",
                                          "B_to_M", "M_to_B"))
        p6 = d6.groupby(["data_root", "coin", "tier", "a", "b", "c", "direction"],
                        dropna=False)[["corr", "r2", "tstat"]].mean().unstack("direction")
        p6.columns = [f"{m}_{d}" for m, d in p6.columns]
        p6 = p6.reset_index()
        if "corr_B_to_M" in p6 and "corr_M_to_B" in p6:
            p6["corr_BtoM_minus_MtoB"] = p6["corr_B_to_M"] - p6["corr_M_to_B"]
        save("A6_direction_by_coin.csv", p6)

        rows = []
        for k, g in p6.groupby(["data_root", "a", "b", "c"]):
            if "corr_BtoM_minus_MtoB" in g:
                n, t, p = one_sample_t(g["corr_BtoM_minus_MtoB"])
                rows.append({**dict(zip(["data_root", "a", "b", "c"], k)),
                             "n_coins": n,
                             "mean_diff": g["corr_BtoM_minus_MtoB"].mean(),
                             "t": t, "p": p})
        save("A6_direction_tests.csv", pd.DataFrame(rows))

    # ---- A7: tradability after costs
    def trad(g: pd.DataFrame) -> dict:
        g = g[g["n_signals"].fillna(0) >= MIN_TRADES_TRADABILITY]
        w = g["n_signals"].to_numpy(dtype=float)
        if len(w) == 0 or w.sum() == 0:
            return {"n_signals": 0}
        gross = float((g["signed_mean_bps"] * w).sum() / w.sum())
        top = g["signed_mean_bps_top50"]
        wt = w[top.notna().to_numpy()]
        gross_top = (float((top.dropna() * wt).sum() / wt.sum())
                     if wt.sum() else np.nan)
        rec = {"n_signals": int(w.sum()), "n_coin_days": int(len(g)),
               "gross_bps_per_signal": gross, "gross_bps_top50": gross_top,
               "hit_rate": float((g["hit_rate"] * w).sum() / w.sum()),
               "frac_days_gross_pos": float((g["signed_mean_bps"] > 0).mean()),
               "mean_abs_lag_ret_bps": float((g["mean_abs_y_bps"] * w).sum() / w.sum())}
        for cb in COST_BPS:
            rec[f"net_bps_cost{cb}"] = gross - cb
        rec["breakeven_cost_bps"] = gross
        return rec

    a7 = pd.DataFrame([{**dict(zip(SPEC, k)), **trad(g)}
                       for k, g in df.groupby(SPEC)])
    save("A7_tradability.csv", a7[a7["n_signals"] > 0])
    a7t = pd.DataFrame([{**dict(zip(SPEC + ["tier"], k)), **trad(g)}
                        for k, g in df[df["tier"].notna()].groupby(SPEC + ["tier"])])
    save("A7_tradability_by_tier.csv",
         a7t[a7t["n_signals"] > 0] if len(a7t) else a7t)

    # ---- A8: coverage
    a8 = df.groupby(["data_root", "lead", "lag", "coin", "tier"],
                    dropna=False).agg(
        n_dates=("date", "nunique"), n_specs=("a", "size"),
        n_fits=("fit_ok", "sum"), n_used_sum=("n_used", "sum"),
        n_used_mean=("n_used", "mean"),
        mean_dt_ratio=("mean_dt_ratio", "mean")).reset_index()
    save("A8_coverage.csv", a8)

    write_summary_md(out_dir, df, a1, a3)


def write_summary_md(out_dir: Path, df: pd.DataFrame,
                     a1: pd.DataFrame, a3: pd.DataFrame) -> None:
    L = ["# VWAP lead-lag: headline numbers\n"]
    L.append(f"- daily regressions: {len(df):,}  "
             f"(fits with n>={MIN_OBS}: {int(df['fit_ok'].sum()):,})")
    L.append(f"- coins: {df['coin'].nunique()}, dates: {df['date'].nunique()}, "
             f"lead/lag labels: {sorted(set(df['lead']) | set(df['lag']))}\n")

    prim = a1[a1["lead"].isin(["binance", "massive"])
              & a1["lag"].isin(["binance", "massive"])]
    if len(prim):
        L.append("## Primary directions "
                 "(mean daily corr, frac p<0.05, BH q of t-test on corr)\n")
        L.append("| data_root | lead->lag | a | b | corr_mean | corr_wmean | "
                 "frac_sig05 | slope_mean | bh_q |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in prim[prim["n_fits"] > 0].sort_values(
                ["data_root", "lead", "a", "b"]).iterrows():
            L.append(f"| {r.data_root} | {r.lead}->{r.lag} | {r.a} | {r.b} | "
                     f"{r.corr_mean:.4f} | {r.corr_wmean:.4f} | "
                     f"{r.frac_sig05:.2f} | {r.slope_mean:.4f} | {r.bh_q:.3g} |")
        L.append("")

    if len(a3):
        prim3 = a3[a3["lead"].isin(["binance", "massive"])
                   & a3["lag"].isin(["binance", "massive"])]
        proxy, cols, pcol = None, [], None
        for cand in ("log_notional", "tier"):
            c = [c for c in a3.columns if c.startswith(f"spearman_corr_vs_{cand}")]
            if c and len(prim3) and prim3[c[0]].notna().any():
                proxy, cols, pcol = cand, c, f"p_corr_vs_{cand}"
                break
        if proxy:
            label = ("log(notional volume)" if proxy == "log_notional"
                     else "volume tier (1=lowest)")
            note = ("" if proxy == "log_notional" else
                    " -- log_notional was unavailable this run (see the "
                    "[analyze] coin_meta log line); showing the coarser tier "
                    "test instead")
            L.append(f"## Volume hypothesis (Spearman of coin-mean corr vs "
                     f"{label}; negative = lower-volume coins more "
                     f"predictable){note}\n")
            for _, r in prim3.dropna(subset=cols).sort_values(
                    ["data_root", "lead", "a", "b"]).iterrows():
                L.append(f"- {r.data_root} {r.lead}->{r.lag} a={r.a} b={r.b}: "
                         f"rho={r[cols[0]]:.3f} (p={r[pcol]:.3g}, n={r.n_coins})")
        else:
            L.append("## Volume hypothesis\n")
            L.append("(no volume-proxy test could be computed for the primary "
                     "binance/massive pairs this run -- see the [analyze] "
                     "coin_meta log line above for why)")
        L.append("")

    L.append("## Files\n")
    for f in sorted(out_dir.glob("A*.csv")):
        L.append(f"- {f.name}")
    L.append("\nInterpretation notes: `signed_mean_bps` is E[sign(lead return) "
             "x lag return] in bps per signal, i.e. gross P&L assuming "
             "execution at the lag venue's window VWAP at t+b; compare against "
             "the COST_BPS scenarios in A7. dt_ratio = actual gap between "
             "window centres / a (values >>1 mean sparse trading). `corr` and "
             "`tstat` are on the thinned, non-overlapping subsample; "
             "`corr_all`/`tstat_all` are the overlapping full sample and match "
             "P09's convention, including its overstated significance.")
    (out_dir / "summary.md").write_text("\n".join(L))
    log("[analyze] saved summary.md")


# ============================================================ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "regress", "analyze", "volume"])
    ap.add_argument("--pairs", default="all", choices=["all", "primary"])
    ap.add_argument("--root", default=None,
                    help="project root (default: $LEADLAG_ROOT or ~)")
    ap.add_argument("--roots", default=",".join(DATA_ROOTS),
                    help="comma-separated data roots to process")
    args = ap.parse_args()
    if args.root:
        set_root(args.root)

    roots = [r for r in args.roots.split(",") if r]
    pairs = valid_label_pairs(args.pairs)
    log(f"ROOT={get_root()}  stage={args.stage}  pairs={args.pairs} "
        f"({len(pairs)} label pairs)  roots={roots}")

    if args.stage == "volume":
        stage_volume(roots)
    if args.stage in ("all", "regress"):
        stage_regress(pairs, roots)
    if args.stage in ("all", "analyze"):
        stage_analyze(roots)
    log("all done")


if __name__ == "__main__":
    main()
