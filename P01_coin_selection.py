from trader_api import API
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from massive import RESTClient
from datetime import date, timedelta
import time
import os
import requests


START = date(2026, 8, 30)
END   = date(2026, 9, 5)
VOLUME     = "volume_0830_0905.csv"
OLS_RESULT = "ols_result.csv"
REF        = "coins.csv"

STABLES = {"X:USDTUSD", "X:USDCUSD", "X:USTUSD", "X:PAXUSD",
           "X:DAIUSD", "X:TUSDUSD", "X:BUSDUSD", "X:FRAXUSD", "X:TUSD"}


# ---------- Binance helpers ----------

def get_binance_active_spot_symbols():
    """Return a set of Binance spot symbols that are actively trading."""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    active_spot = set()
    for s in data["symbols"]:
        if s.get("status") != "TRADING":
            continue

        # Primary check: explicit boolean flag from the API
        if s.get("isSpotTradingAllowed") is True:
            active_spot.add(s["symbol"])
            continue

        # Fallback: permission arrays. Note the field is
        # "permissionSets" (singular "permission", capital "Sets").
        perms = set()
        flat = s.get("permissions")
        if flat:
            perms.update(flat)
        nested = s.get("permissionSets")
        if nested:
            for group in nested:
                perms.update(group)
        if "SPOT" in perms:
            active_spot.add(s["symbol"])

    return active_spot
def binance_to_massive(symbol: str) -> str | None:
    """Convert a Binance spot symbol to Massive's X:BASEUSD format."""
    if symbol.endswith("USDT"):
        base = symbol[:-4]
    elif symbol.endswith("USD"):
        base = symbol[:-3]
    else:
        return None
    return f"X:{base}USD"

def massive_to_binance(massive_ticker, binance_symbols):
    """Return the Binance spot symbol for a Massive ticker.
    Prefers USDT quote, falls back to USD. Returns None if no match."""
    if not massive_ticker.startswith("X:") or not massive_ticker.endswith("USD"):
        return None
    base = massive_ticker[2:-3]  # strip "X:" and "USD"

    for quote in ("USDT", "USD"):
        candidate = f"{base}{quote}"
        if candidate in binance_symbols:
            return candidate
    return None

def filter_to_common_universe(massive_tickers, binance_symbols):
    """Keep only Massive tickers that have a matching Binance spot pair."""
    binance_as_massive = {binance_to_massive(s) for s in binance_symbols}
    binance_as_massive.discard(None)
    return [t for t in massive_tickers if t in binance_as_massive]


# ---------- data fetch ----------

def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_volume_series(client, ticker, start, end):
    """Return {date: (volume, notional_volume)} for one ticker over the range."""
    bars = client.get_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=start.isoformat(),
        to=end.isoformat(),
        limit=5000,
        adjusted="true",
    )
    out = {}
    for bar in bars:
        d = date.fromtimestamp(bar.timestamp / 1000)
        vw = getattr(bar, "vw", None)
        if vw is None:
            vw = (bar.high + bar.low) / 2
        out[d] = (bar.volume, bar.volume * vw)
    return out


def _cache_is_valid(path):
    """Re-fetch if the file is missing, empty, header-only, or lacks required columns."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    required = {"Tick Name", "Volume", "VWAP", "Notional Volume"}
    return required.issubset(df.columns) and len(df) > 0


def build_universe(client):
    """Fetch 7-day unit volume, aggregate VWAP, and notional volume per ticker.
    Only tickers present on both Massive and Binance spot are included.
    Returns None if the intersection is empty (caller should abort)."""
    # 1) Massive: active USD pairs
    massive_tickers = []
    for t in client.list_tickers(
        market="crypto",
        active="true",
        order="asc",
        limit="1000",
        sort="ticker",
    ):
        if t.currency_symbol == "USD" and t.ticker not in STABLES:
            massive_tickers.append(t.ticker)
    print(f"Massive universe: {len(massive_tickers)} active USD pairs (excl. stables)")

    # 2) Binance: active spot symbols
    binance_symbols = get_binance_active_spot_symbols()
    print(f"Binance universe: {len(binance_symbols)} active spot symbols")

    # 3) Intersection
    tickers = filter_to_common_universe(massive_tickers, binance_symbols)
    print(f"Common universe: {len(tickers)} tickers on both Massive and Binance")

    if not tickers:
        print("WARNING: no common tickers found — check mapping logic")
        return None

    days = list(daterange(START, END))
    print(f"Fetching volume for {len(tickers)} common tickers, {START} -> {END}")

    per_ticker = {}
    for i, tick in enumerate(tickers, 1):
        try:
            per_ticker[tick] = fetch_volume_series(client, tick, START, END)
            missing = [d for d in days if d not in per_ticker[tick]]
            msg = f"  [{i}/{len(tickers)}] {tick}: {len(per_ticker[tick])}/{len(days)} days"
            if missing:
                msg += f"  MISSING {missing}"
            print(msg)
        except Exception as e:
            print(f"  [{i}/{len(tickers)}] {tick}: ERROR {e}")
            per_ticker[tick] = {}

    rows = []
    for tick in tickers:
        per_day = per_ticker.get(tick, {})
        if all(d in per_day for d in days):
            total_vol      = sum(per_day[d][0] for d in days)
            total_notional = sum(per_day[d][1] for d in days)
            agg_vwap       = total_notional / total_vol if total_vol > 0 else np.nan
        else:
            total_vol      = np.nan
            total_notional = np.nan
            agg_vwap       = np.nan
        rows.append({
            "Tick Name":       tick,
            "Volume":          total_vol,
            "VWAP":            agg_vwap,
            "Notional Volume": total_notional,
        })

    return pd.DataFrame(rows)


# ---------- regressions ----------

def run_ols(df, y_col="Notional Volume", x_col="Volume"):
    """Linear: y = b0 + b1 * x."""
    d = df.dropna(subset=[y_col, x_col]).copy()

    x = d[x_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    model = OLS(y, X).fit()

    d["y_pred"]       = model.fittedvalues
    d["residual"]     = model.resid
    d["abs_residual"] = np.abs(model.resid)

    print("\n=== Linear OLS ===")
    print(model.summary())
    print("\nTop 20 residuals:")
    print(d.sort_values("abs_residual", ascending=False)
           .head(20)[["Tick Name", "Volume", "Notional Volume", "y_pred", "residual"]])
    return d, model


def build_ref(df_clean, binance_symbols, col="Notional Volume", n_per_tier=3):
    """Pick n_per_tier coins nearest each quantile point. Tier 1 = largest.
    Returns columns: M Tick Name, B Tick Name, NT Tier."""
    quantiles = [1.00, 0.75, 0.50, 0.25, 0.00]

    rows = []
    for tier, q in enumerate(quantiles, start=1):
        qv = df_clean[col].quantile(q)
        closest = (df_clean[col] - qv).abs().nsmallest(n_per_tier).index
        for i in closest:
            m_tick = df_clean.loc[i, "Tick Name"]
            b_tick = massive_to_binance(m_tick, binance_symbols)
            rows.append({
                "M Tick Name": m_tick,
                "B Tick Name": b_tick,
                "NT Tier":     tier,
            })
    return pd.DataFrame(rows)


# ---------- main ----------

def main():
    client = RESTClient(API)

    # 0) Binance symbol set — needed for the M -> B mapping in coins.csv
    binance_symbols = get_binance_active_spot_symbols()
    print(f"Binance symbols fetched: {len(binance_symbols)}")

    # 1) Build or load clean universe
    if not _cache_is_valid(VOLUME):
        df = build_universe(client)
        if df is None or len(df) == 0:
            print("Universe is empty — aborting before writing cache.")
            return
        df.to_csv(VOLUME, index=False)
        print(f"Saved -> {VOLUME}")
    else:
        df = pd.read_csv(VOLUME)
        print(f"Loaded cached {VOLUME}")

    # 2) Clean universe
    df_clean = (df.dropna()
                  .sort_values("Notional Volume", ascending=False)
                  .reset_index(drop=True))
    print(f"\nClean universe: {len(df_clean)} of {len(df)} tickers "
          f"({len(df) - len(df_clean)} dropped for missing days)")
    print(df_clean.head(10))

    # 3) Linear OLS
    df_lin, model_lin = run_ols(df_clean)
    df_lin.to_csv(OLS_RESULT, index=False)
    print(f"Saved -> {OLS_RESULT}")

    # 4) Quantile picks -> coins.csv  (now with Binance ticker)
    ref = build_ref(df_clean, binance_symbols, col="Notional Volume", n_per_tier=3)
    print("\n=== coins.csv ===")
    print(ref)
    ref.to_csv(REF, index=False)
    print(f"Saved -> {REF}")


if __name__ == "__main__":
    main()