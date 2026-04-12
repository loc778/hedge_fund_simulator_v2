# data/macro.py — hedge_v2
# ═══════════════════════════════════════════════════════════
# MACRO DATA INGESTION
# Sources:
#   yfinance  — India VIX, USDINR, Crude Oil, Gold (daily)
#   FRED API  — GDP India, CPI India, Fed Rate, US CPI, US 10Y Bond
#
# Writes to macro_indicators table.
# FII/DII and RBI columns (Repo_Rate, IIP_Growth, Forex_Reserves_USD)
# are written by fii_dii.py and rbi_macro.py respectively.
#
# RESUME SUPPORT:
# Checks latest date in DB and only fetches what's missing.
# Safe to re-run at any time.
# ═══════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import time

from config import DATA_START, MACRO_YFINANCE, MACRO_FRED, TABLES
from data.db import get_engine, save_to_db

load_dotenv()
engine = get_engine()
fred   = Fred(api_key=os.getenv("FRED_API_KEY"))


# ── Resume: find latest date already in DB ────────────────────────────
def get_latest_date_in_db() -> str:
    """Returns latest date in macro_indicators, or DATA_START if empty."""
    try:
        result = pd.read_sql(
            f"SELECT MAX(Date) as max_date FROM {TABLES['macro']}",
            con=engine
        )
        max_date = result["max_date"].iloc[0]
        if pd.isna(max_date):
            return DATA_START
        # Move back 5 days to re-fetch recent dates (handles late data arrival)
        return (pd.to_datetime(str(max_date)) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    except Exception:
        return DATA_START


# ── yfinance download with retry ──────────────────────────────────────
def download_yfinance(symbol: str, start: str, retries: int = 3) -> pd.Series | None:
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol, start=start, interval="1d",
                progress=False, timeout=20
            )
            if df.empty:
                return None
            # Handle MultiIndex columns from newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df["Close"]
        except Exception as e:
            if attempt < retries:
                time.sleep(5 * attempt)
            else:
                print(f"    ❌ {symbol} failed after {retries} attempts: {e}")
                return None


def main():
    print("=" * 60)
    print("MACRO DATA INGESTION — hedge_v2")
    print("=" * 60)

    fetch_start = get_latest_date_in_db()
    print(f"\nFetching from: {fetch_start} → today\n")

    # ── Step 1: yfinance daily data ───────────────────────────────────
    print("📥 Fetching daily market data from yfinance...")
    daily_frames = []

    for col_name, symbol in MACRO_YFINANCE.items():
        print(f"  {col_name} ({symbol})...", end=" ")
        series = download_yfinance(symbol, fetch_start)
        if series is not None:
            series.name = col_name
            daily_frames.append(series)
            print(f"✅ {len(series)} rows")
        else:
            print("⚠️  failed — will be NULL")

    if not daily_frames:
        print("\n❌ All yfinance fetches failed — check network/API")
        return

    daily = pd.concat(daily_frames, axis=1, sort=True)
    daily.index = pd.to_datetime(
        [str(d)[:10] for d in daily.index]
    )
    daily.index.name = "Date"

    # ── Step 2: FRED monthly/quarterly data ──────────────────────────
    print("\n📥 Fetching macro indicators from FRED...")
    fred_frames = []

    for col_name, series_id in MACRO_FRED.items():
        try:
            series = fred.get_series(series_id, observation_start=fetch_start)
            series.name = col_name
            series.index = pd.to_datetime(
                [str(d)[:10] for d in series.index]
            )
            series.index.name = "Date"
            fred_frames.append(series)
            print(f"  ✅ {col_name} ({series_id}) — {len(series)} observations")
        except Exception as e:
            print(f"  ❌ {col_name} ({series_id}) — {e}")

    if not fred_frames:
        print("❌ No FRED data fetched — check FRED_API_KEY in .env")
        return

    fred_df = pd.concat(fred_frames, axis=1, sort=True).round(4)

    # Forward fill FRED monthly/quarterly onto daily trading dates
    fred_df = fred_df.ffill()
    fred_daily = fred_df.reindex(daily.index, method="ffill")

    # ── Step 3: Combine ───────────────────────────────────────────────
    combined = pd.concat([daily, fred_daily], axis=1)
    combined.reset_index(inplace=True)
    combined["Date"] = pd.to_datetime(
        combined["Date"].astype(str).str[:10]
    ).dt.date

    # Drop rows where ALL yfinance columns are null (non-trading days)
    market_cols = [c for c in list(MACRO_YFINANCE.keys()) if c in combined.columns]
    combined.dropna(how="all", subset=market_cols, inplace=True)
    combined = combined.round(4)

    print(f"\n✅ Combined macro data: {len(combined)} rows")

    # ── Step 4: Save ──────────────────────────────────────────────────
    save_to_db(combined, TABLES["macro"], engine)
    print("\n📋 Next: python data/fii_dii.py")


if __name__ == "__main__":
    main()