"""
data_quality.py
---------------
Classifies all 500 tickers into Tier A / B / C / X based on OHLCV coverage,
gap%, average daily value, and F&O eligibility.

Writes to: stock_data_quality table
Run: python data/data_quality.py
Safe to re-run: TRUNCATE + reinsert on every run (fully recomputable table)
"""

import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import text
from datetime import date

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TICKERS, DATA_QUALITY, TABLES,
    PROJECT_ROOT, FO_LIST_CSV
)
from data.db import get_engine

# ── Config ────────────────────────────────────────────────────────────────────

FO_LIST_PATH   = FO_LIST_CSV
TABLE_OUT      = TABLES["data_quality"]
TABLE_OHLCV    = TABLES["ohlcv"]
ADV_WINDOW     = 90          # trading days for ADV calculation
TODAY          = date.today()

# Tier thresholds (sourced from config.DATA_QUALITY)
MIN_DAYS_TIER_X  = DATA_QUALITY["tier_x_min_days"]        # 252
MIN_YEARS_TIER_A = DATA_QUALITY["tier_a"]["min_years"]    # 8
MIN_COV_TIER_A   = DATA_QUALITY["tier_a"]["min_coverage"] # 0.93
MIN_ADV_TIER_A   = DATA_QUALITY["tier_a"]["min_adv_cr"]   # 5.0 Cr
MIN_YEARS_TIER_B = DATA_QUALITY["tier_b"]["min_years"]    # 4
MIN_COV_TIER_B   = DATA_QUALITY["tier_b"]["min_coverage"] # 0.75
MIN_ADV_TIER_B   = DATA_QUALITY["tier_b"]["min_adv_cr"]   # 1.0 Cr


# ── Load F&O list ─────────────────────────────────────────────────────────────

def load_fo_set(path: str) -> set:
    """
    Parse fo_mktlots.csv. Symbol is column index 1.
    Excludes index derivatives (NIFTY, BANKNIFTY, etc.).
    Returns a set of clean symbols (no .NS suffix).
    """
    INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "NIFTYNXT50", "FINNIFTY",
                     "MIDCPNIFTY", "Symbol", "SYMBOL"}
    df = pd.read_csv(path, header=0)
    symbols = df.iloc[:, 1].astype(str).str.strip()
    fo_set = {s for s in symbols if s not in INDEX_SYMBOLS and s and s != "nan"}
    print(f"[F&O] Loaded {len(fo_set)} F&O-eligible symbols from {path}")
    return fo_set


# ── Load OHLCV from DB ────────────────────────────────────────────────────────

def load_ohlcv(engine) -> pd.DataFrame:
    """
    Pulls Date, Ticker, Close, Volume from nifty500_ohlcv.
    Only columns needed for coverage + ADV calculation.
    """
    query = f"SELECT Date, Ticker, Close, Volume FROM {TABLE_OHLCV} ORDER BY Ticker, Date"
    print("[OHLCV] Loading from DB...")
    df = pd.read_sql(query, engine, parse_dates=["Date"])
    print(f"[OHLCV] {len(df):,} rows, {df['Ticker'].nunique()} tickers")
    return df


# ── Expected trading days estimator ──────────────────────────────────────────

def expected_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """
    Approximates expected NSE trading days between two dates.
    Uses 252 trading days/year ratio (standard Indian market assumption).
    Calendar days × (252/365) gives a close enough estimate for gap% calculation.
    """
    calendar_days = (end - start).days
    return max(1, round(calendar_days * 252 / 365))


# ── Per-ticker classification ─────────────────────────────────────────────────

def classify_ticker(symbol: str, df_ticker: pd.DataFrame, fo_set: set) -> dict:
    """
    Given a single ticker's OHLCV rows, compute all quality metrics and assign tier.
    """
    df_ticker = df_ticker.sort_values("Date").reset_index(drop=True)

    trading_days_total = len(df_ticker)
    data_start         = df_ticker["Date"].iloc[0].date()
    data_end           = df_ticker["Date"].iloc[-1].date()

    coverage_years = round(trading_days_total / 252, 2)

    # Coverage ratio — actual / expected (0.93 = Tier A threshold in config)
    expected       = expected_trading_days(df_ticker["Date"].iloc[0], df_ticker["Date"].iloc[-1])
    coverage_ratio = round(min(1.0, trading_days_total / expected), 4)
    gap_pct        = round((1 - coverage_ratio) * 100, 2)   # stored for reference

    # ADV — last ADV_WINDOW trading days, in Crores (1 Cr = 1e7)
    recent = df_ticker.tail(ADV_WINDOW)
    daily_value        = recent["Close"] * recent["Volume"]
    avg_daily_value_cr = round(daily_value.mean() / 1e7, 4)

    f_and_o = 1 if symbol.replace(".NS", "") in fo_set else 0

    # Tier assignment — coverage_ratio matches config thresholds
    if trading_days_total < MIN_DAYS_TIER_X:
        tier = "X"
    elif (coverage_years    >= MIN_YEARS_TIER_A
          and coverage_ratio >= MIN_COV_TIER_A
          and avg_daily_value_cr >= MIN_ADV_TIER_A):
        tier = "A"
    elif (coverage_years    >= MIN_YEARS_TIER_B
          and coverage_ratio >= MIN_COV_TIER_B
          and avg_daily_value_cr >= MIN_ADV_TIER_B):
        tier = "B"
    else:
        tier = "C"

    return {
        "Ticker"             : symbol,
        "Trading_Days_Total" : trading_days_total,
        "Data_Start_Date"    : data_start,
        "Coverage_Years"     : coverage_years,
        "Gap_Pct"            : gap_pct,
        "Avg_Daily_Volume_Cr" : avg_daily_value_cr,
        "F_and_O_Listed"     : f_and_o,
        "Data_Tier"          : tier,
        "Tier_Assigned_Date" : TODAY,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    engine = get_engine()
    fo_set = load_fo_set(FO_LIST_PATH)
    df_ohlcv = load_ohlcv(engine)

    results = []
    tickers_in_db = df_ohlcv["Ticker"].unique()
    print(f"[CLASSIFY] Processing {len(tickers_in_db)} tickers...")

    for ticker in tickers_in_db:
        df_t = df_ohlcv[df_ohlcv["Ticker"] == ticker]
        row  = classify_ticker(ticker, df_t, fo_set)
        results.append(row)

    df_out = pd.DataFrame(results)

    # ── Summary ───────────────────────────────────────────────────────────────
    tier_counts = df_out["Data_Tier"].value_counts().to_dict()
    print("\n[RESULTS] Tier distribution:")
    for tier in ["A", "B", "C", "X"]:
        print(f"  Tier {tier}: {tier_counts.get(tier, 0)}")
    print(f"  Total  : {len(df_out)}")

    fo_count = df_out["F_and_O_Listed"].sum()
    print(f"\n[RESULTS] F&O listed: {fo_count} / {len(df_out)}")
    print(f"[RESULTS] Avg coverage (Tier A): "
          f"{df_out[df_out['Data_Tier']=='A']['Coverage_Years'].mean():.1f} years")

    # ── Write to DB ───────────────────────────────────────────────────────────
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {TABLE_OUT}"))
        print(f"\n[DB] Truncated {TABLE_OUT}")

    df_out.to_sql(TABLE_OUT, engine, if_exists="append", index=False)
    print(f"[DB] Inserted {len(df_out)} rows into {TABLE_OUT}")

    # ── Tickers missing from OHLCV ────────────────────────────────────────────
    config_tickers = set(TICKERS)
    db_tickers     = set(tickers_in_db)
    missing        = config_tickers - db_tickers
    if missing:
        print(f"\n[WARN] {len(missing)} config tickers have no OHLCV data:")
        print("  ", sorted(missing))

    print("\n[DONE] data_quality.py complete.")


if __name__ == "__main__":
    main()