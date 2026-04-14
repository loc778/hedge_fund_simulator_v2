# data/fii_dii_historical.py — hedge_v2
# ═══════════════════════════════════════════════════════════
# FII/FPI HISTORICAL DATA LOADER — NSDL FILES
#
# SETUP:
#   Place all downloaded NSDL .xls files in:
#   hedge_fund_simulator_v2/files/fii_nsdl/
#   Files can have any name — year is auto-detected from content.
#
# WHAT IT DOES:
#   1. Reads all .xls files in files/fii_nsdl/
#   2. Parses FPI Equity net flows (handles both pre/post 2022 formats)
#   3. Matches each month's value to exact trading dates in macro_indicators
#   4. UPDATEs FII_Net_Buy_Cr — only fills NULLs, never overwrites
#
# GRANULARITY NOTE:
#   NSDL provides monthly totals only. Each trading day in a given
#   month receives that month's total. This is standard practice —
#   FII flow is a monthly trend signal for model training purposes.
#   Recent daily precision is handled by fii_dii.py (NSE API).
# ═══════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import re
from sqlalchemy import text

from config import TABLES, PROJECT_ROOT
from data.db import get_engine

engine = get_engine()

FII_FILES_DIR = os.path.join(PROJECT_ROOT, "files", "fii_nsdl")

MONTH_MAP = {
    "january": 1,   "february": 2,  "march": 3,    "april": 4,
    "may": 5,       "june": 6,      "july": 7,      "august": 8,
    "september": 9, "october": 10,  "november": 11, "december": 12
}


def parse_fii_file(filepath: str):
    """
    Parses one NSDL FPI .xls file (HTML-disguised-as-xls).

    Handles both formats:
      Pre-2022 : 5 cols  — Month | Equity | Debt | Total | (empty)
      Post-2022: 13 cols — Month | Equity | Debt-General | Debt-VRR |
                           Debt-FAR | Hybrid | MF cols... | AIF | Total

    In both formats: Equity is always column index 1.
    Year is extracted from the column header title string.

    Returns (DataFrame[Year, Month, FII_Equity_Cr], error_string)
    """
    try:
        dfs = pd.read_html(filepath)
    except Exception as e:
        return None, f"read error: {e}"

    if not dfs:
        return None, "no tables found"

    df = max(dfs, key=len)

    # Extract year from column headers
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
    Loads all dates from macro_indicators and assigns each trading day
    the FII equity value for its (Year, Month).
    Only dates that exist in macro_indicators are matched —
    ensures no orphaned rows and exact date alignment.
    """
    trading_dates = pd.read_sql(
        f"SELECT Date FROM {TABLES['macro']} ORDER BY Date",
        con=engine
    )
    trading_dates["Date"]  = pd.to_datetime(trading_dates["Date"])
    trading_dates["Year"]  = trading_dates["Date"].dt.year
    trading_dates["Month"] = trading_dates["Date"].dt.month

    merged = trading_dates.merge(
        monthly_df[["Year", "Month", "FII_Equity_Cr"]],
        on=["Year", "Month"],
        how="left"
    ).dropna(subset=["FII_Equity_Cr"])

    merged["Date"] = merged["Date"].dt.date
    return merged[["Date", "FII_Equity_Cr"]].rename(
        columns={"FII_Equity_Cr": "FII_Net_Buy_Cr"}
    )


def update_db(daily_df: pd.DataFrame) -> int:
    """UPDATE macro_indicators — only fills NULLs, never overwrites."""
    updated = 0
    with engine.connect() as conn:
        for _, row in daily_df.iterrows():
            result = conn.execute(
                text(
                    f"UPDATE {TABLES['macro']} "
                    f"SET FII_Net_Buy_Cr = :val "
                    f"WHERE Date = :date AND FII_Net_Buy_Cr IS NULL"
                ),
                {"val":  round(float(row["FII_Net_Buy_Cr"]), 4),
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
        print(f"   Create it and place your NSDL .xls files there.")
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

    # Parse all files
    all_monthly  = []
    failed_files = []

    for filepath in xls_files:
        fname = os.path.basename(filepath)
        df, err = parse_fii_file(filepath)

        if err:
            print(f"  ❌ {fname} — {err}")
            failed_files.append(fname)
            continue

        year  = df["Year"].iloc[0]
        total = df["FII_Equity_Cr"].sum()
        print(f"  ✅ {fname} → {year} | "
              f"{len(df)} months | net: {total:+,.0f} Cr")
        all_monthly.append(df)

    if not all_monthly:
        print("\n❌ No data parsed from any file.")
        return

    combined = pd.concat(all_monthly, ignore_index=True)

    # Guard against duplicate year files
    dupes = combined.groupby("Year")["Month"].count()
    dupes = dupes[dupes > 12]
    if not dupes.empty:
        print(f"\n⚠️  Duplicate files for years: {dupes.index.tolist()} — keeping first")
        combined = combined.drop_duplicates(subset=["Year", "Month"], keep="first")

    print(f"\n✅ {len(combined)} monthly records | "
          f"{combined['Year'].min()}–{combined['Year'].max()}")

    # Expand monthly → daily using exact macro_indicators dates
    print("\n🔧 Matching to trading dates in macro_indicators...")
    daily_df = expand_to_daily(combined)
    print(f"   {len(daily_df)} trading days matched")

    if daily_df.empty:
        print("❌ No dates matched — run macro.py first to populate macro_indicators.")
        return

    # Update DB
    print("\n💾 Updating FII_Net_Buy_Cr...")
    updated = update_db(daily_df)
    print(f"✅ {updated} rows updated")

    # Verification summary
    result = pd.read_sql(
        f"""SELECT
                COUNT(*) as total_rows,
                SUM(CASE WHEN FII_Net_Buy_Cr IS NULL  THEN 1 ELSE 0 END) as nulls,
                SUM(CASE WHEN FII_Net_Buy_Cr IS NOT NULL THEN 1 ELSE 0 END) as filled,
                MIN(CASE WHEN FII_Net_Buy_Cr IS NOT NULL THEN Date END) as earliest,
                MAX(CASE WHEN FII_Net_Buy_Cr IS NOT NULL THEN Date END) as latest
            FROM {TABLES['macro']}""",
        con=engine
    )
    r = result.iloc[0]
    print(f"\n   macro_indicators total rows : {r['total_rows']}")
    print(f"   FII_Net_Buy_Cr filled        : {r['filled']}")
    print(f"   FII_Net_Buy_Cr NULLs left    : {r['nulls']}")
    print(f"   FII data range               : {r['earliest']} → {r['latest']}")

    if failed_files:
        print(f"\n⚠️  Failed: {failed_files}")

    print("\n📋 Next: python data/rbi_macro.py")


if __name__ == "__main__":
    main()