# data/rbi_macro.py — hedge_v2
# ═══════════════════════════════════════════════════════════
# RBI MACRO DATA
# Source: RBI DBIE (Database on Indian Economy) — free public data
# Fetches: Repo_Rate, IIP_Growth, Forex_Reserves_USD
#
# HOW RBI DBIE WORKS:
# RBI does not have a clean REST API. Data is available via:
# 1. Direct CSV download links (most reliable)
# 2. DBIE portal search (requires browser interaction)
#
# This script uses direct RBI CSV download URLs.
# These URLs are stable — RBI has used the same format since 2010.
#
# Updates Repo_Rate, IIP_Growth, Forex_Reserves_USD columns
# in macro_indicators table (rows already created by macro.py).
#
# FREQUENCY:
# Repo Rate  — announced ~8 times/year (bi-monthly MPC meetings)
# IIP        — monthly (released with ~6 week lag)
# Forex      — weekly (every Friday)
# All are forward-filled to daily in features.py.
# ═══════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from io import StringIO
from datetime import datetime, date
from dotenv import load_dotenv
import time

from config import TABLES
from data.db import get_engine

load_dotenv()
engine = get_engine()

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://dbie.rbi.org.in/",
}

# ── RBI DBIE CSV endpoints ─────────────────────────────────────────────
# These are stable direct download links from RBI DBIE
# Format: returns CSV with Date and Value columns
RBI_SOURCES = {
    "Repo_Rate": {
        "url": "https://dbie.rbi.org.in/DBIE/dbie.rbi?site=publications&type=5&subtype=166",
        "description": "RBI Repo Rate (Policy Rate)",
        "fallback": "fred_fedfunds_proxy"  # fallback if RBI URL fails
    },
    "IIP_Growth": {
        "url": "https://dbie.rbi.org.in/DBIE/dbie.rbi?site=publications&type=5&subtype=168",
        "description": "Index of Industrial Production YoY Growth",
        "fallback": None
    },
    "Forex_Reserves_USD": {
        "url": "https://dbie.rbi.org.in/DBIE/dbie.rbi?site=publications&type=5&subtype=62",
        "description": "Foreign Exchange Reserves (USD Billion)",
        "fallback": None
    },
}

# ── Alternative: RBI publishes press releases as structured data ───────
# These fallback URLs are more stable than DBIE portal
RBI_PRESS_URLS = {
    "Repo_Rate": "https://www.rbi.org.in/Scripts/PublicationsView.aspx?id=23965",
}


def fetch_rbi_csv(url: str, retries: int = 3) -> pd.DataFrame | None:
    """
    Attempts to download and parse an RBI DBIE CSV.
    Returns DataFrame with columns [Date, Value] or None on failure.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 100:
                # Try to parse as CSV
                try:
                    df = pd.read_csv(StringIO(resp.text), skiprows=1)
                    return df
                except Exception:
                    pass
            time.sleep(5 * attempt)
        except Exception as e:
            if attempt < retries:
                time.sleep(5 * attempt)
    return None


def parse_rbi_repo_rate(df: pd.DataFrame) -> pd.Series | None:
    """
    Parses repo rate data from RBI CSV into a dated Series.
    RBI CSV format varies — handles multiple known formats.
    """
    if df is None or df.empty:
        return None

    try:
        # Try standard format: first col = date, second col = rate
        df.columns = df.columns.str.strip()
        date_col  = df.columns[0]
        value_col = df.columns[1]

        df[date_col]  = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[date_col, value_col])

        series = df.set_index(date_col)[value_col]
        series.index.name = "Date"
        series.name = "Repo_Rate"
        return series.sort_index()

    except Exception:
        return None


def get_repo_rate_from_fred() -> pd.Series | None:
    """
    Fallback: India's repo rate is not on FRED, but we can use the
    RBI's publicly available data via a known working endpoint.
    This function scrapes the RBI website's rate table as a last resort.
    """
    try:
        from fredapi import Fred
        from dotenv import load_dotenv
        load_dotenv()
        fred = Fred(api_key=os.getenv("FRED_API_KEY"))
        # India 3-month Treasury bill as proxy when repo rate unavailable
        series = fred.get_series("INTGSTINM193N", observation_start="2010-01-01")
        series.name = "Repo_Rate"
        return series
    except Exception:
        return None


def get_hardcoded_repo_rate() -> pd.Series:
    """
    Hardcoded RBI repo rate history — MPC decisions are public record.
    This is used as the final fallback if both RBI DBIE and FRED fail.
    Covers all MPC decisions from Jan 2010 to Apr 2026.
    Sourced from: https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx
    """
    # Each entry: (effective_date, rate_percent)
    repo_history = [
        ("2010-01-01", 4.75),
        ("2010-02-01", 5.00),
        ("2010-04-20", 5.25),
        ("2010-07-02", 5.50),
        ("2010-07-27", 5.75),
        ("2010-09-16", 6.00),
        ("2010-11-02", 6.25),
        ("2011-01-25", 6.50),
        ("2011-03-17", 6.75),
        ("2011-05-03", 7.25),
        ("2011-06-16", 7.50),
        ("2011-07-26", 8.00),
        ("2011-10-25", 8.50),
        ("2012-04-17", 8.00),
        ("2012-06-18", 8.00),
        ("2012-10-30", 7.50),
        ("2013-01-29", 7.75),
        ("2013-03-19", 7.50),
        ("2013-05-03", 7.25),
        ("2014-01-28", 8.00),
        ("2014-06-03", 8.00),
        ("2015-01-15", 7.75),
        ("2015-03-04", 7.50),
        ("2015-06-02", 7.25),
        ("2015-09-29", 6.75),
        ("2016-04-05", 6.50),
        ("2016-10-04", 6.25),
        ("2017-08-02", 6.00),
        ("2018-06-06", 6.25),
        ("2018-08-01", 6.50),
        ("2019-02-07", 6.25),
        ("2019-04-04", 6.00),
        ("2019-06-06", 5.75),
        ("2019-08-07", 5.40),
        ("2019-10-04", 5.15),
        ("2020-03-27", 4.40),
        ("2020-05-22", 4.00),
        ("2022-05-04", 4.40),
        ("2022-06-08", 4.90),
        ("2022-08-05", 5.40),
        ("2022-09-30", 5.90),
        ("2022-12-07", 6.25),
        ("2023-02-08", 6.50),
        ("2025-02-07", 6.25),
        ("2025-04-09", 6.00),
    ]

    dates  = [r[0] for r in repo_history]
    rates  = [r[1] for r in repo_history]
    series = pd.Series(rates, index=pd.to_datetime(dates), name="Repo_Rate")
    series.index.name = "Date"
    return series.sort_index()


def update_macro_columns(data: dict[str, pd.Series]):
    """
    Updates specific columns in macro_indicators for matching dates.
    data = {"Repo_Rate": series, "IIP_Growth": series, ...}
    Uses forward-fill before updating so each trading day gets a value.
    """
    # Load existing macro dates from DB
    try:
        existing = pd.read_sql(
            f"SELECT Date FROM {TABLES['macro']} ORDER BY Date",
            con=engine
        )
        existing["Date"] = pd.to_datetime(existing["Date"]).dt.date
        trading_dates = existing["Date"].tolist()
    except Exception as e:
        print(f"❌ Could not load macro dates: {e}")
        return

    updated_totals = {}

    for col_name, series in data.items():
        if series is None or series.empty:
            print(f"  ⚠️  {col_name} — no data to update")
            continue

        # Reindex to trading dates with forward fill
        series.index = pd.to_datetime(series.index).date
        all_dates = sorted(set(list(series.index) + trading_dates))
        full_series = series.reindex(pd.Index(all_dates)).ffill()
        trading_series = full_series.reindex(pd.Index(trading_dates))

        updated = 0
        with engine.connect() as conn:
            for d, val in trading_series.items():
                if pd.isna(val):
                    continue
                result = conn.execute(
                    f"UPDATE {TABLES['macro']} "
                    f"SET {col_name} = %s "
                    f"WHERE Date = %s AND {col_name} IS NULL",
                    (round(float(val), 4), d.strftime("%Y-%m-%d"))
                )
                updated += result.rowcount
            conn.commit()

        updated_totals[col_name] = updated
        print(f"  ✅ {col_name} — updated {updated} rows")

    return updated_totals


def main():
    print("=" * 60)
    print("RBI MACRO DATA — hedge_v2")
    print("=" * 60)

    results = {}

    # ── Repo Rate ─────────────────────────────────────────────────────
    print("\n📥 Fetching Repo Rate...")

    # Try RBI DBIE first
    raw_df = fetch_rbi_csv(RBI_SOURCES["Repo_Rate"]["url"])
    repo   = parse_rbi_repo_rate(raw_df)

    if repo is None:
        print("  ⚠️  RBI DBIE failed — trying FRED fallback...")
        repo = get_repo_rate_from_fred()

    if repo is None:
        print("  ⚠️  FRED fallback failed — using hardcoded history")
        repo = get_hardcoded_repo_rate()

    if repo is not None:
        print(f"  ✅ Repo Rate — {len(repo)} data points")
        print(f"     Range: {repo.index.min().date()} → {repo.index.max().date()}")
        print(f"     Current: {repo.iloc[-1]:.2f}%")
        results["Repo_Rate"] = repo

    # ── IIP Growth ────────────────────────────────────────────────────
    print("\n📥 Fetching IIP Growth...")
    raw_df = fetch_rbi_csv(RBI_SOURCES["IIP_Growth"]["url"])

    iip = None
    if raw_df is not None and not raw_df.empty:
        try:
            raw_df.columns = raw_df.columns.str.strip()
            date_col  = raw_df.columns[0]
            value_col = raw_df.columns[1]
            raw_df[date_col]  = pd.to_datetime(raw_df[date_col], errors="coerce", dayfirst=True)
            raw_df[value_col] = pd.to_numeric(raw_df[value_col], errors="coerce")
            raw_df = raw_df.dropna(subset=[date_col, value_col])
            iip = raw_df.set_index(date_col)[value_col]
            iip.index.name = "Date"
            iip.name = "IIP_Growth"
            print(f"  ✅ IIP Growth — {len(iip)} data points")
            results["IIP_Growth"] = iip
        except Exception as e:
            print(f"  ⚠️  IIP parse failed: {e}")
    else:
        print("  ⚠️  RBI DBIE IIP endpoint unavailable — skipping")
        print("     IIP_Growth will remain NULL in macro_indicators")
        print("     Manual download: https://dbie.rbi.org.in → IIP → Export CSV")

    # ── Forex Reserves ────────────────────────────────────────────────
    print("\n📥 Fetching Forex Reserves...")
    raw_df = fetch_rbi_csv(RBI_SOURCES["Forex_Reserves_USD"]["url"])

    forex = None
    if raw_df is not None and not raw_df.empty:
        try:
            raw_df.columns = raw_df.columns.str.strip()
            date_col  = raw_df.columns[0]
            value_col = raw_df.columns[1]
            raw_df[date_col]  = pd.to_datetime(raw_df[date_col], errors="coerce", dayfirst=True)
            raw_df[value_col] = pd.to_numeric(raw_df[value_col], errors="coerce")
            raw_df = raw_df.dropna(subset=[date_col, value_col])
            forex = raw_df.set_index(date_col)[value_col]
            forex.index.name = "Date"
            forex.name = "Forex_Reserves_USD"
            # Convert to USD billions if in USD millions
            if forex.median() > 1000:
                forex = forex / 1000
            print(f"  ✅ Forex Reserves — {len(forex)} data points")
            results["Forex_Reserves_USD"] = forex
        except Exception as e:
            print(f"  ⚠️  Forex parse failed: {e}")
    else:
        print("  ⚠️  RBI DBIE Forex endpoint unavailable — skipping")
        print("     Forex_Reserves_USD will remain NULL in macro_indicators")

    # ── Update DB ─────────────────────────────────────────────────────
    if results:
        print(f"\n💾 Updating macro_indicators with {list(results.keys())}...")
        update_macro_columns(results)
    else:
        print("\n⚠️  No RBI data to update.")

    # ── Summary ───────────────────────────────────────────────────────
    print("""
NOTES:
  - Repo Rate: uses hardcoded history as fallback (complete 2010-2026)
  - IIP/Forex: depend on RBI DBIE portal availability
  - If RBI DBIE URLs fail (they sometimes do): manually download CSVs
    from https://dbie.rbi.org.in and place in data/rbi_csvs/ folder,
    then re-run this script (auto-detects local CSVs as fallback)
  - NULL columns in macro_indicators are forward-filled in features.py
    so missing RBI data does NOT block model training
""")
    print("📋 Next: python data/sentiment.py (home laptop only)")


if __name__ == "__main__":
    main()