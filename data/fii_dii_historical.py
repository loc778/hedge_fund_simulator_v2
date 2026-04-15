# data/fii_dii_historical.py — hedge_v2
# ═══════════════════════════════════════════════════════════
# FII/FPI HISTORICAL DATA LOADER — NSDL FILES
#
# SETUP:
#   Place all downloaded NSDL .xls files in:
#   hedge_fund_simulator_v2/files/fii_nsdl/
#
# C5 FIX — UNIT MISMATCH:
# NSDL provides monthly totals only. Previously each trading day
# in a month received the full month's total — mixing monthly
# units with daily units in the same column.
#
# Fix: divide monthly total by number of NSE trading days in that
# month to produce a unit-consistent daily approximation.
# Written to FII_Monthly_Net_Cr (separate column from FII_Daily_Net_Cr).
# This is an approximation — each trading day gets an equal share
# of the monthly flow. Not exact daily values, but unit-consistent.
#
# Recent daily precision is handled by fii_dii.py → FII_Daily_Net_Cr.
# features.py will prefer FII_Daily_Net_Cr when available (post-2025),
# and fall back to FII_Monthly_Net_Cr for historical rows.
# ═══════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import re
from sqlalchemy import text

from config import TABLES, PROJECT_ROOT, NSE_HOLIDAYS
from data.db import get_engine

engine = get_engine()

FII_FILES_DIR = os.path.join(PROJECT_ROOT, "files", "fii_nsdl")

MONTH_MAP = {
    "january": 1,   "february": 2,  "march": 3,    "april": 4,
    "may": 5,       "june": 6,      "july": 7,      "august": 8,
    "september": 9, "october": 10,  "november": 11, "december": 12
}


def get_trading_days_in_month(year: int, month: int) -> int:
    """
    Returns the count of NSE trading days in a given year/month.
    Uses NSE_HOLIDAYS from config to exclude known holidays.
    """
    from datetime import date, timedelta
    first = date(year, month, 1)
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)

    count   = 0
    current = first
    while current <= last:
        if current.weekday() < 5 and current not in NSE_HOLIDAYS:
            count += 1
        current += timedelta(days=1)
    return max(count, 1)   # avoid division by zero


def parse_fii_file(filepath: str):
    """
    Parses one NSDL FPI .xls file.
    Returns (DataFrame[Year, Month, FII_Equity_Cr], error_string)
    """
    try:
        dfs = pd.read_html(filepath)
    except Exception as e:
        return None, f"read error: {e}"

    if not dfs:
        return None, "no tables found"

    df = max(dfs, key=len)

    year = None
    for col in df.columns:
        m = re.search(r"Calendar Year - (\d{4})", str(col))
        if m:
            year = int(m.group(1))
            break

    if year is None:
        return None, "year not found in column headers"

    equity_col = df.columns[1]
    month_col  = df.columns[0]

    records = []
    for _, row in df.iterrows():
        month_raw = str(row[month_col]).strip().lower()

        if (month_raw.startswith("total") or
                month_raw.startswith("*") or
                month_raw in ("nan", "calendar year", "month", "financial year")):
            continue

        month_num = MONTH_MAP.get(month_raw)
        if month_num is None:
            continue

        try:
            val = float(str(row[equity_col]).replace(",", "").strip())
        except (ValueError, TypeError):
            continue

        records.append({
            "Year":          year,
            "Month":         month_num,
            "FII_Equity_Cr": val,
        })

    if not records:
        return None, "no monthly records parsed"

    return pd.DataFrame(records), None


def expand_to_daily(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands monthly totals to daily rows in macro_indicators.
    Each trading day receives (monthly_total / trading_days_in_month).
    This makes the column unit-consistent: Cr per trading day.

    Written to FII_Monthly_Net_Cr (not FII_Net_Buy_Cr).
    """
    trading_dates = pd.read_sql(
        f"SELECT Date FROM {TABLES['macro']} ORDER BY Date",
        con=engine
    )
    trading_dates["Date"]  = pd.to_datetime(trading_dates["Date"])
    trading_dates["Year"]  = trading_dates["Date"].dt.year
    trading_dates["Month"] = trading_dates["Date"].dt.month

    # Compute per-day value = monthly total / trading days in that month
    monthly_df = monthly_df.copy()
    monthly_df["Trading_Days"] = monthly_df.apply(
        lambda r: get_trading_days_in_month(int(r["Year"]), int(r["Month"])),
        axis=1
    )
    monthly_df["FII_Monthly_Net_Cr"] = (
        monthly_df["FII_Equity_Cr"] / monthly_df["Trading_Days"]
    ).round(4)

    merged = trading_dates.merge(
        monthly_df[["Year", "Month", "FII_Monthly_Net_Cr"]],
        on=["Year", "Month"],
        how="left"
    ).dropna(subset=["FII_Monthly_Net_Cr"])

    merged["Date"] = merged["Date"].dt.date
    return merged[["Date", "FII_Monthly_Net_Cr"]]


def update_db(daily_df: pd.DataFrame) -> int:
    """
    UPDATEs FII_Monthly_Net_Cr — only fills NULLs, never overwrites.
    (Historical data doesn't change once NSDL files are finalized.)
    """
    updated = 0
    with engine.connect() as conn:
        for _, row in daily_df.iterrows():
            result = conn.execute(
                text(
                    f"UPDATE {TABLES['macro']} "
                    f"SET FII_Monthly_Net_Cr = :val "
                    f"WHERE Date = :date AND FII_Monthly_Net_Cr IS NULL"
                ),
                {"val":  float(row["FII_Monthly_Net_Cr"]),
                 "date": str(row["Date"])}
            )
            updated += result.rowcount
        conn.commit()
    return updated


def main():
    print("=" * 60)
    print("FII/FPI HISTORICAL LOADER — NSDL files")
    print("=" * 60)

    if not os.path.exists(FII_FILES_DIR):
        print(f"❌ Folder not found: {FII_FILES_DIR}")
        print(f"   Create it and place NSDL .xls files there.")
        return

    xls_files = sorted([
        os.path.join(FII_FILES_DIR, f)
        for f in os.listdir(FII_FILES_DIR)
        if f.lower().endswith(".xls") or f.lower().endswith(".xlsx")
    ])

    if not xls_files:
        print(f"❌ No .xls/.xlsx files found in {FII_FILES_DIR}")
        return

    print(f"\n📂 {len(xls_files)} files found\n")

    all_monthly  = []
    failed_files = []

    for filepath in xls_files:
        fname  = os.path.basename(filepath)
        df, err = parse_fii_file(filepath)

        if err:
            print(f"  ❌ {fname} — {err}")
            failed_files.append(fname)
            continue

        year  = df["Year"].iloc[0]
        total = df["FII_Equity_Cr"].sum()
        print(f"  ✅ {fname} → {year} | {len(df)} months | net: {total:+,.0f} Cr")
        all_monthly.append(df)

    if not all_monthly:
        print("\n❌ No data parsed from any file.")
        return

    combined = pd.concat(all_monthly, ignore_index=True)

    dupes = combined.groupby("Year")["Month"].count()
    dupes = dupes[dupes > 12]
    if not dupes.empty:
        print(f"\n⚠️  Duplicate files for years: {dupes.index.tolist()} — keeping first")
        combined = combined.drop_duplicates(subset=["Year", "Month"], keep="first")

    print(f"\n✅ {len(combined)} monthly records | "
          f"{combined['Year'].min()}–{combined['Year'].max()}")

    print("\n🔧 Expanding to daily with per-trading-day approximation...")
    daily_df = expand_to_daily(combined)
    print(f"   {len(daily_df)} trading days matched")

    if daily_df.empty:
        print("❌ No dates matched — run macro.py first.")
        return

    print("\n💾 Updating FII_Monthly_Net_Cr in macro_indicators...")
    updated = update_db(daily_df)
    print(f"✅ {updated} rows updated")

    # Verification
    result = pd.read_sql(
        f"""SELECT
                COUNT(*) as total_rows,
                SUM(CASE WHEN FII_Monthly_Net_Cr IS NULL  THEN 1 ELSE 0 END) as nulls,
                SUM(CASE WHEN FII_Monthly_Net_Cr IS NOT NULL THEN 1 ELSE 0 END) as filled,
                MIN(CASE WHEN FII_Monthly_Net_Cr IS NOT NULL THEN Date END) as earliest,
                MAX(CASE WHEN FII_Monthly_Net_Cr IS NOT NULL THEN Date END) as latest
            FROM {TABLES['macro']}""",
        con=engine
    )
    r = result.iloc[0]
    print(f"\n   macro_indicators total rows      : {r['total_rows']}")
    print(f"   FII_Monthly_Net_Cr filled        : {r['filled']}")
    print(f"   FII_Monthly_Net_Cr NULLs left    : {r['nulls']}")
    print(f"   FII data range                   : {r['earliest']} → {r['latest']}")

    if failed_files:
        print(f"\n⚠️  Failed: {failed_files}")

    print("\n📋 Next: python data/rbi_macro.py")


if __name__ == "__main__":
    main()