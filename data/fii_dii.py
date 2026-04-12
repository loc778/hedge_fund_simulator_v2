# data/fii_dii.py — hedge_v2
# ═══════════════════════════════════════════════════════════
# FII/DII DAILY FLOWS
# Source: NSE public API (no auth required)
# Endpoint: https://www.nseindia.com/api/fiidiiTradeReact
#
# Returns daily net buy/sell values for FII and DII.
# Updates FII_Net_Buy_Cr and DII_Net_Buy_Cr columns in
# macro_indicators table (rows already created by macro.py).
#
# WHY THIS MATTERS:
# FII sell-off > Rs 5,000 Cr in a single session triggers the
# risk manager's defensive cash-raise protocol. This data
# is the most important daily macro signal in Indian markets.
#
# RESUME SUPPORT:
# Checks which dates already have FII/DII data and skips them.
# ═══════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from datetime import date, timedelta, datetime
from dotenv import load_dotenv
import time

from config import DATA_START, TABLES
from data.db import get_engine

load_dotenv()
engine = get_engine()

# NSE requires browser-like headers — blocks plain requests
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
    """NSE requires a valid session cookie — visit homepage first."""
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
    """
    Fetches the FII/DII trade data from NSE API.
    Returns the last ~30 days of data as a list of dicts.
    NSE's endpoint does not support date range — always returns recent data.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(NSE_FII_DII_URL, timeout=20)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (401, 403):
                # Session expired — refresh
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
    Parses NSE FII/DII JSON response into a clean DataFrame.

    NSE response structure (each item):
    {
      "date": "09-Apr-2026",
      "fiiNetTrade": "1234.56",   or "-1234.56" for net sell
      "diiNetTrade": "567.89"
    }
    Returns DataFrame with columns: Date, FII_Net_Buy_Cr, DII_Net_Buy_Cr
    """
    records = []
    for item in raw:
        try:
            date_str = item.get("date", "")
            fii_val  = item.get("fiiNetTrade", "0") or "0"
            dii_val  = item.get("diiNetTrade", "0") or "0"

            # Parse date — NSE uses "DD-Mon-YYYY" format
            parsed_date = datetime.strptime(date_str.strip(), "%d-%b-%Y").date()

            fii_net = float(str(fii_val).replace(",", ""))
            dii_net = float(str(dii_val).replace(",", ""))

            records.append({
                "Date":           parsed_date,
                "FII_Net_Buy_Cr": round(fii_net, 4),
                "DII_Net_Buy_Cr": round(dii_net, 4),
            })
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def get_dates_missing_fii() -> set:
    """Returns set of dates in macro_indicators that have NULL FII_Net_Buy_Cr."""
    try:
        result = pd.read_sql(
            f"SELECT Date FROM {TABLES['macro']} WHERE FII_Net_Buy_Cr IS NULL",
            con=engine
        )
        return set(pd.to_datetime(result["Date"]).dt.date.tolist())
    except Exception:
        return set()


def update_macro_table(df: pd.DataFrame):
    """
    Updates FII_Net_Buy_Cr and DII_Net_Buy_Cr in macro_indicators
    for dates where the values are currently NULL.
    Uses UPDATE not INSERT — rows already exist from macro.py.
    """
    updated = 0
    with engine.connect() as conn:
        for _, row in df.iterrows():
            result = conn.execute(
                f"UPDATE {TABLES['macro']} "
                f"SET FII_Net_Buy_Cr = %s, DII_Net_Buy_Cr = %s "
                f"WHERE Date = %s",
                (row["FII_Net_Buy_Cr"], row["DII_Net_Buy_Cr"],
                 row["Date"].strftime("%Y-%m-%d"))
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
        print("   NSE API is sometimes unavailable outside market hours.")
        print("   Try running during IST market hours (9am-6pm).")
        return

    df = parse_fii_dii(raw)

    if df.empty:
        print("❌ No records parsed from NSE response.")
        return

    print(f"✅ Got {len(df)} days of FII/DII data")
    print(f"   Date range: {df['Date'].min()} → {df['Date'].max()}")
    print(f"\n   Sample (last 5 days):")
    print(df.tail(5).to_string(index=False))

    # Only update dates that are missing FII data in macro_indicators
    missing_dates = get_dates_missing_fii()
    df_to_update  = df[df["Date"].isin(missing_dates)]

    if df_to_update.empty:
        print("\n✅ All available dates already have FII/DII data.")
    else:
        print(f"\n💾 Updating {len(df_to_update)} rows in macro_indicators...")
        updated = update_macro_table(df_to_update)
        print(f"✅ Updated {updated} rows")

    # Note on historical data
    print("""
⚠️  NOTE ON HISTORICAL FII/DII DATA:
   NSE's API only returns the last ~30 days.
   For historical FII/DII flows (2010-2025), manual download required:
   1. Go to: https://www.nseindia.com/reports-indices-equity-fii-dii
   2. Download monthly CSVs
   3. Run: python data/fii_dii_historical.py (to be built if needed)
   For now, pre-2025 rows will remain NULL in macro_indicators.
   features.py will forward-fill these via macro_feature_cols ffill step.
""")

    print("📋 Next: python data/rbi_macro.py")


if __name__ == "__main__":
    main()