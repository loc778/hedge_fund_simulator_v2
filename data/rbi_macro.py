# data/rbi_macro.py — hedge_v2
# ═══════════════════════════════════════════════════════════
# RBI MACRO DATA
# Fetches: Repo_Rate, IIP_Growth, Forex_Reserves_USD
#
# SOURCES:
#   Repo_Rate          — Hardcoded MPC history (RBI DBIE URLs are
#                        session-based, not programmatically accessible)
#   IIP_Growth         — FRED: INDPROINDMISMEI (OECD India industrial
#                        production index, monthly from 1994).
#                        YoY % computed here.
#   Forex_Reserves_USD — FRED: TRESEGINM052N (IMF total reserves excl.
#                        gold for India, USD millions → /1000 = billions)
#
# All forward-filled to daily in features.py.
# ═══════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from config import TABLES, RBI_REPO_HISTORY
from data.db import get_engine

load_dotenv()
engine = get_engine()

START_DATE = "2010-01-01"

FRED_IIP_SERIES   = "INDPROINDMISMEI"  # India industrial production index
FRED_FOREX_SERIES = "TRESEGINM052N"    # India total reserves excl. gold (USD mn)


# ══════════════════════════════════════════════════════════════════════
# REPO RATE — hardcoded MPC history
# ══════════════════════════════════════════════════════════════════════

def get_repo_rate() -> pd.Series:
    """
    Reads RBI repo rate history from config.RBI_REPO_HISTORY.
    To update: edit RBI_REPO_HISTORY in config.py after each MPC meeting.
    """
    series = pd.Series(
        [r[1] for r in RBI_REPO_HISTORY],
        index=pd.to_datetime([r[0] for r in RBI_REPO_HISTORY]),
        name="Repo_Rate"
    )
    series.index.name = "Date"
    return series.sort_index()


# ══════════════════════════════════════════════════════════════════════
# IIP GROWTH — FRED INDPROINDMISMEI
# ══════════════════════════════════════════════════════════════════════

def get_iip_growth() -> pd.Series | None:
    try:
        from fredapi import Fred
        fred = Fred(api_key=os.getenv("FRED_API_KEY"))

        # Fetch extra year for YoY computation
        raw = fred.get_series(FRED_IIP_SERIES, observation_start="2009-01-01")
        raw = raw.dropna()

        iip = raw.pct_change(periods=12) * 100
        iip = iip[iip.index >= START_DATE].dropna()
        iip.name = "IIP_Growth"
        iip.index.name = "Date"

        print(f"  ✅ IIP Growth ({FRED_IIP_SERIES}) — {len(iip)} monthly points")
        print(f"     Range: {iip.index.min().date()} → {iip.index.max().date()}")
        print(f"     Latest: {iip.iloc[-1]:.2f}%")
        return iip

    except Exception as e:
        print(f"  ❌ IIP Growth FRED fetch failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# FOREX RESERVES — FRED TRESEGINM052N
# ══════════════════════════════════════════════════════════════════════

def get_forex_reserves() -> pd.Series | None:
    try:
        from fredapi import Fred
        fred = Fred(api_key=os.getenv("FRED_API_KEY"))

        raw = fred.get_series(FRED_FOREX_SERIES, observation_start=START_DATE)
        raw = raw.dropna()

        forex = raw / 1000  # USD mn → USD bn
        forex.name = "Forex_Reserves_USD"
        forex.index.name = "Date"

        print(f"  ✅ Forex Reserves ({FRED_FOREX_SERIES}) — {len(forex)} monthly points")
        print(f"     Range: {forex.index.min().date()} → {forex.index.max().date()}")
        print(f"     Latest: ${forex.iloc[-1]:.1f}B")
        return forex

    except Exception as e:
        print(f"  ❌ Forex Reserves FRED fetch failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# DB UPDATE — forward-fill monthly → daily, update NULLs only
# ══════════════════════════════════════════════════════════════════════

def update_macro_columns(data: dict):
    """
    Writes Repo_Rate / IIP_Growth / Forex_Reserves_USD into macro_indicators.
    Only updates rows where column IS NULL (safe to re-run).
    Forward-fills each series across all trading dates before writing.
    """
    try:
        existing = pd.read_sql(
            f"SELECT Date FROM {TABLES['macro']} ORDER BY Date",
            con=engine
        )
        existing["Date"] = pd.to_datetime(existing["Date"])
        trading_dates = existing["Date"].tolist()
    except Exception as e:
        print(f"❌ Could not load macro dates: {e}")
        return

    if not trading_dates:
        print("❌ macro_indicators is empty — run macro.py first")
        return

    updated_totals = {}

    for col_name, series in data.items():
        if series is None or series.empty:
            print(f"  ⚠️  {col_name} — no data, skipping")
            continue

        # Normalize index to Timestamp (no time component)
        series.index = pd.to_datetime(series.index).normalize()

        # Merge series dates + all trading dates, then forward-fill
        all_dates   = sorted(set(list(series.index) + trading_dates))
        full_series = series.reindex(pd.Index(all_dates)).ffill()
        trading_series = full_series.reindex(pd.Index(trading_dates))

        sql = text(
            f"UPDATE {TABLES['macro']} "
            f"SET `{col_name}` = :val "
            f"WHERE Date = :dt AND `{col_name}` IS NULL"
        )

        updated = 0
        with engine.connect() as conn:
            for ts, val in trading_series.items():
                if pd.isna(val):
                    continue
                result = conn.execute(sql, {
                    "val": round(float(val), 4),
                    "dt":  ts.strftime("%Y-%m-%d")
                })
                updated += result.rowcount
            conn.commit()

        updated_totals[col_name] = updated
        print(f"  ✅ {col_name} — updated {updated} rows")

    return updated_totals


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("RBI MACRO DATA — hedge_v2")
    print("=" * 60)

    results = {}

    # Repo Rate
    print("\n📥 Repo Rate (hardcoded MPC history)...")
    repo = get_repo_rate()
    print(f"  ✅ {len(repo)} change points | Current: {repo.iloc[-1]:.2f}%")
    results["Repo_Rate"] = repo

    # IIP Growth
    print("\n📥 IIP Growth (FRED)...")
    iip = get_iip_growth()
    if iip is not None:
        results["IIP_Growth"] = iip
    else:
        print("     Check FRED_API_KEY in .env")

    # Forex Reserves
    print("\n📥 Forex Reserves (FRED)...")
    forex = get_forex_reserves()
    if forex is not None:
        results["Forex_Reserves_USD"] = forex
    else:
        print("     Check FRED_API_KEY in .env")

    # Write to DB
    print(f"\n💾 Updating macro_indicators: {list(results.keys())}...")
    update_macro_columns(results)

    # Null check
    print("\n📊 Null check after update:")
    try:
        null_check = pd.read_sql(
            f"""
            SELECT
                SUM(CASE WHEN Repo_Rate         IS NULL THEN 1 ELSE 0 END) AS Repo_nulls,
                SUM(CASE WHEN IIP_Growth         IS NULL THEN 1 ELSE 0 END) AS IIP_nulls,
                SUM(CASE WHEN Forex_Reserves_USD IS NULL THEN 1 ELSE 0 END) AS Forex_nulls,
                COUNT(*) AS total_rows
            FROM {TABLES['macro']}
            """,
            con=engine
        )
        print(null_check.to_string(index=False))
    except Exception as e:
        print(f"  ⚠️  Verification failed: {e}")



if __name__ == "__main__":
    main()