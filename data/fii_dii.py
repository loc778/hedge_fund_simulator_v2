# data/fii_dii.py — hedge_v2
# ═══════════════════════════════════════════════════════════
# FII/DII DAILY FLOWS
# Source: NSE public API (no auth required)
# Endpoint: https://www.nseindia.com/api/fiidiiTradeReact
#
# C5 FIX: Writes to FII_Daily_Net_Cr (not FII_Net_Buy_Cr).
# This separates actual daily values (this script, last ~30 days)
# from the monthly approximation (fii_dii_historical.py).
# features.py uses FII_Daily_Net_Cr when available, falls back
# to FII_Monthly_Net_Cr for pre-2025 historical rows.
# ═══════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
import requests
import pandas as pd
from datetime import date, datetime
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
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer":         "https://www.nseindia.com/",
    "Connection":      "keep-alive",
}

NSE_FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
NSE_HOME_URL    = "https://www.nseindia.com"


def get_nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get(NSE_HOME_URL, timeout=10)
        time.sleep(2)
    except Exception:
        pass
    return session


def fetch_fii_dii_page(session: requests.Session,
                       retries: int = 3) -> list[dict] | None:
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(NSE_FII_DII_URL, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (401, 403):
                session = get_nse_session()
                time.sleep(3)
                continue
            time.sleep(5 * attempt)
        except Exception as e:
            if attempt < retries:
                time.sleep(5 * attempt)
            else:
                print(f"  ❌ NSE API failed: {e}")
                return None
    return None


def parse_fii_dii(raw: list[dict]) -> pd.DataFrame:
    """
    Parses NSE FII/DII JSON response.
    Returns DataFrame: Date, FII_Daily_Net_Cr, DII_Net_Buy_Cr
    """
    records = []
    for item in raw:
        try:
            date_str = item.get("date", "")
            fii_val  = item.get("fiiNetTrade", "0") or "0"
            dii_val  = item.get("diiNetTrade", "0") or "0"

            parsed_date = datetime.strptime(date_str.strip(), "%d-%b-%Y").date()
            fii_net     = float(str(fii_val).replace(",", ""))
            dii_net     = float(str(dii_val).replace(",", ""))

            records.append({
                "Date":              parsed_date,
                "FII_Daily_Net_Cr":  round(fii_net, 4),
                "DII_Net_Buy_Cr":    round(dii_net, 4),
            })
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def get_dates_missing_fii_daily() -> set:
    """Returns dates in macro_indicators that have NULL FII_Daily_Net_Cr."""
    try:
        result = pd.read_sql(
            f"SELECT Date FROM {TABLES['macro']} WHERE FII_Daily_Net_Cr IS NULL",
            con=engine
        )
        return set(pd.to_datetime(result["Date"]).dt.date.tolist())
    except Exception:
        return set()


def update_macro_table(df: pd.DataFrame):
    updated = 0
    with engine.connect() as conn:
        for _, row in df.iterrows():
            result = conn.execute(
                text(f"UPDATE {TABLES['macro']} "
                     f"SET FII_Daily_Net_Cr = :fii, DII_Net_Buy_Cr = :dii "
                     f"WHERE Date = :date"),
                {"fii":  row["FII_Daily_Net_Cr"],
                 "dii":  row["DII_Net_Buy_Cr"],
                 "date": row["Date"].strftime("%Y-%m-%d")}
            )
            updated += result.rowcount
        conn.commit()
    return updated


def main():
    print("=" * 60)
    print("FII/DII DAILY FLOWS — hedge_v2")
    print("=" * 60)

    print("\n🔗 Creating NSE session...")
    session = get_nse_session()

    print("📥 Fetching FII/DII data from NSE API...")
    raw = fetch_fii_dii_page(session)

    if not raw:
        print("❌ Could not fetch FII/DII data from NSE.")
        print("   Try running during IST market hours (9am–6pm).")
        return

    df = parse_fii_dii(raw)

    if df.empty:
        print("❌ No records parsed from NSE response.")
        return

    print(f"✅ Got {len(df)} days of FII/DII data")
    print(f"   Date range: {df['Date'].min()} → {df['Date'].max()}")
    print(f"\n   Sample (last 5 days):")
    print(df.tail(5).to_string(index=False))

    missing_dates = get_dates_missing_fii_daily()
    df_to_update  = df[df["Date"].isin(missing_dates)]

    if df_to_update.empty:
        print("\n✅ All available dates already have FII_Daily_Net_Cr data.")
    else:
        print(f"\n💾 Updating {len(df_to_update)} rows in macro_indicators...")
        updated = update_macro_table(df_to_update)
        print(f"✅ Updated {updated} rows (FII_Daily_Net_Cr, DII_Net_Buy_Cr)")

    print("""
ℹ️  FII data strategy:
   FII_Daily_Net_Cr  = actual daily values (this script, last ~30 days)
   FII_Monthly_Net_Cr = monthly total ÷ trading days (fii_dii_historical.py)
   features.py uses daily when available, monthly as fallback for history.
""")

    print("📋 Next: python data/rbi_macro.py")


if __name__ == "__main__":
    main()