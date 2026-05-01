# data/macro.py — hedge_v2
# ═══════════════════════════════════════════════════════════
# MACRO DATA INGESTION
# Sources:
#   yfinance  — India VIX, USDINR, Crude Oil, Gold (daily)
#   FRED API  — GDP India, CPI India, Fed Rate, US CPI, US 10Y Bond
# ═══════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
from sqlalchemy import text
import time

from config import DATA_START, MACRO_YFINANCE, MACRO_FRED, TABLES
from data.db import get_engine

load_dotenv()
engine = get_engine()
fred   = Fred(api_key=os.getenv("FRED_API_KEY"))


def get_latest_date_in_db() -> str:
    try:
        result = pd.read_sql(
            f"SELECT MAX(Date) as max_date FROM {TABLES['macro']}",
            con=engine
        )
        max_date = result["max_date"].iloc[0]
        if pd.isna(max_date):
            return DATA_START
        # Overlap 5 days to catch revised FRED values (C6 fix: these are now UPSERTed)
        return (pd.to_datetime(str(max_date)) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    except Exception:
        return DATA_START


def download_yfinance(symbol: str, start: str, retries: int = 3) -> pd.Series | None:
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol, start=start, interval="1d",
                progress=False, timeout=20
                # auto_adjust removed — causes YFTzMissingError on index/FX/commodity
                # symbols (^INDIAVIX, INR=X, CL=F, GC=F) in newer yfinance versions
            )
            if df is None or df.empty:
                return None
            # Flatten MultiIndex columns (newer yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Newer yfinance auto-adjusts by default — Close is already adjusted
            if "Close" in df.columns:
                series = df["Close"]
            elif "Adj Close" in df.columns:
                series = df["Adj Close"]
            else:
                return None
            series = series.dropna()
            return series if not series.empty else None
        except Exception as e:
            if attempt < retries:
                time.sleep(5 * attempt)
            else:
                print(f"    ❌ {symbol} failed after {retries} attempts: {e}")
                return None


def upsert_macro_rows(df: pd.DataFrame):
    import math

    yf_cols     = list(MACRO_YFINANCE.keys())
    fred_cols   = list(MACRO_FRED.keys())
    update_cols = [c for c in yf_cols + fred_cols if c in df.columns]

    if not update_cols:
        return

    set_clause   = ", ".join([f"{c} = VALUES({c})" for c in update_cols])
    col_list     = ["Date"] + update_cols
    placeholders = ", ".join([":" + c for c in col_list])
    col_names    = ", ".join(col_list)

    sql = text(f"""
        INSERT INTO {TABLES['macro']} ({col_names})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {set_clause}
    """)

    def clean_val(v):
        """Convert NaN/inf to None — MySQL cannot store Python nan literals."""
        if v is None:
            return None
        try:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
        except Exception:
            pass
        return v

    rows = df[col_list].to_dict(orient="records")

    with engine.connect() as conn:
        for row in rows:
            row["Date"] = str(row["Date"])
            clean_row   = {k: clean_val(v) for k, v in row.items()}
            conn.execute(sql, clean_row)
        conn.commit()

    print(f"  ✅ Upserted {len(rows)} rows into macro_indicators")


def main():
    import sys as _sys
    print("=" * 60)
    print("MACRO DATA INGESTION — hedge_v2")
    print("=" * 60)

    # Optional CLI override: python data/macro.py 2010-01-01
    # Use for full historical backfill when yfinance columns are NULL
    if len(_sys.argv) > 1:
        fetch_start = _sys.argv[1]
        print(f"\n⚠️  CLI override: full backfill from {fetch_start} → today\n")
    else:
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
        print("\n❌ All yfinance fetches failed — check network")
        return

    daily = pd.concat(daily_frames, axis=1, sort=True)
    daily.index = pd.to_datetime([str(d)[:10] for d in daily.index])
    daily.index.name = "Date"

    # ── Step 2: FRED data ─────────────────────────────────────────────
    print("\n📥 Fetching macro indicators from FRED...")
    fred_frames = []

    for col_name, series_id in MACRO_FRED.items():
        try:
            series = fred.get_series(series_id, observation_start=fetch_start)
            series.name = col_name
            series.index = pd.to_datetime([str(d)[:10] for d in series.index])
            series.index.name = "Date"
            fred_frames.append(series)
            print(f"  ✅ {col_name} ({series_id}) — {len(series)} observations")
        except Exception as e:
            print(f"  ❌ {col_name} ({series_id}) — {e}")

    if not fred_frames:
        print("❌ No FRED data fetched — check FRED_API_KEY in .env")
        return

    fred_df = pd.concat(fred_frames, axis=1, sort=True).round(4)
    fred_df = fred_df.ffill()
    fred_daily = fred_df.reindex(daily.index, method="ffill")

    # ── Step 3: Combine ───────────────────────────────────────────────
    combined = pd.concat([daily, fred_daily], axis=1)
    combined.reset_index(inplace=True)
    combined["Date"] = pd.to_datetime(
        combined["Date"].astype(str).str[:10]
    ).dt.date

    market_cols = [c for c in list(MACRO_YFINANCE.keys()) if c in combined.columns]
    combined.dropna(how="all", subset=market_cols, inplace=True)
    combined = combined.round(4)

    print(f"\n✅ Combined macro data: {len(combined)} rows")

    # ── Step 4: UPSERT (C6 fix — not INSERT IGNORE) ───────────────────
    print("\n💾 Upserting into macro_indicators...")
    upsert_macro_rows(combined)


if __name__ == "__main__":
    main()