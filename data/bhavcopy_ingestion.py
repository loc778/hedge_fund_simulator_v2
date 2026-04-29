# data/bhavcopy_ingestion.py
# ═══════════════════════════════════════════════════════════
# NSE BHAVCOPY INGESTION — hedge_v2
# Downloads daily bhavcopy ZIPs from NSE archives (Jan 2010 → today)
# Parses OHLCV, builds Adj_Close from corporate_actions splits table,
# saves to nifty500_ohlcv table.
# ═══════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import zipfile
import io
import pandas as pd
import numpy as np
import yfinance as yf
import time
import bisect
import pickle
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta

from config import (TICKERS, DATA_START, DATA_END, BHAVCOPY_TEMP_DIR,
                    TABLES, BATCH_SIZE, BATCH_DELAY, NSE_HOLIDAYS)
from data.db import get_engine, save_to_db

engine = get_engine()

# ── NSE Bhavcopy Request Headers ─────────────────────────────────────
# NSE blocks requests without a proper browser user-agent
HEADERS = {
    "User-Agent"      : ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/120.0.0.0 Safari/537.36"),
    "Accept"          : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language" : "en-US,en;q=0.5",
    "Accept-Encoding" : "gzip, deflate, br",
    "Connection"      : "keep-alive",
    "Referer"         : "https://www.nseindia.com/",
}

# NSE Session (required — NSE uses cookies for bot detection)
session = requests.Session()
session.headers.update(HEADERS)

# ── Ticker Set for Fast Lookup ────────────────────────────────────────
# Convert TICKERS list to set of base symbols (without .NS) for bhavcopy matching
TICKER_SET = {t.replace('.NS', '') for t in TICKERS}

# ═══════════════════════════════════════════════════════════
# SECTION 1 — DATE UTILITIES
# ═══════════════════════════════════════════════════════════

def is_trading_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    if d in NSE_HOLIDAYS:
        return False
    return True


def get_trading_days(start: str, end: str = None) -> list:
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date   = date.today() if end is None else datetime.strptime(end, "%Y-%m-%d").date()

    days = []
    current = start_date
    while current <= end_date:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)

    return days


def get_already_ingested_dates() -> set:
    try:
        result = pd.read_sql(
            f"SELECT DISTINCT Date FROM {TABLES['ohlcv']}",
            con=engine
        )
        return set(pd.to_datetime(result['Date']).dt.date.tolist())
    except Exception:
        return set()


# ═══════════════════════════════════════════════════════════
# SECTION 2 — BHAVCOPY DOWNLOAD AND PARSE
# ═══════════════════════════════════════════════════════════

def build_bhavcopy_urls(d: date) -> list:
    """
    Returns list of URLs to try for a given date, in priority order.

    NSE changed bhavcopy format on July 8, 2024 (NSE Circular 62424):
    - Old format (pre Jul 8 2024):
        /content/historical/EQUITIES/{YEAR}/{MON}/cm{DD}{MON}{YYYY}bhav.csv.zip
        Columns: SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, TOTTRDQTY
    - New UDiFF format (Jul 8 2024 onwards):
        /content/cm/BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip
        Columns: TckrSymb, SctySrs, OpnPric, HghPric, LwPric, ClsPric, TtlTradgVol
    """
    year     = d.strftime("%Y")
    month    = d.strftime("%b").upper()
    date_old = d.strftime("%d%b%Y").upper()   # 05JUL2024
    date_new = d.strftime("%Y%m%d")            # 20240708

    old_url = (f"https://nsearchives.nseindia.com/content/historical/EQUITIES"
               f"/{year}/{month}/cm{date_old}bhav.csv.zip")
    new_url = (f"https://nsearchives.nseindia.com/content/cm"
               f"/BhavCopy_NSE_CM_0_0_0_{date_new}_F_0000.csv.zip")

    udiff_cutoff = date(2024, 7, 8)
    if d >= udiff_cutoff:
        return [new_url, old_url]
    else:
        return [old_url]


def parse_bhavcopy_df(raw_df: pd.DataFrame, d: date) -> pd.DataFrame | None:
    raw_df.columns = raw_df.columns.str.strip().str.upper()

    # Column name mapping across all NSE format versions
    col_map = {
        'SYMBOL'          : 'SYMBOL',
        'TCKRSYMB'        : 'SYMBOL',
        'SERIES'          : 'SERIES',
        'SCTYSRS'         : 'SERIES',
        'OPEN'            : 'OPEN',
        'OPNPRIC'         : 'OPEN',
        'HIGH'            : 'HIGH',
        'HGHPRIC'         : 'HIGH',
        'LOW'             : 'LOW',
        'LWPRIC'          : 'LOW',
        'CLOSE'           : 'CLOSE',
        'CLSPRIC'         : 'CLOSE',
        'TOTTRDQTY'       : 'VOLUME',
        'TTL_TRD_QNTY'    : 'VOLUME',
        'TOTAL_TRADED_QTY': 'VOLUME',
        'TTLTRADGVOL'     : 'VOLUME',
    }
    raw_df.rename(columns={k: v for k, v in col_map.items()
                            if k in raw_df.columns}, inplace=True)

    # Keep only EQ series
    if 'SERIES' in raw_df.columns:
        raw_df = raw_df[raw_df['SERIES'].str.strip() == 'EQ']

    keep_cols = ['SYMBOL', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    available = [c for c in keep_cols if c in raw_df.columns]
    raw_df = raw_df[available].copy()

    # Filter to our tickers
    raw_df = raw_df[raw_df['SYMBOL'].str.strip().isin(TICKER_SET)]
    if raw_df.empty:
        return None

    raw_df['Date']   = d
    raw_df['SYMBOL'] = raw_df['SYMBOL'].str.strip() + '.NS'
    raw_df.rename(columns={'SYMBOL': 'Ticker'}, inplace=True)

    for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')

    return raw_df


def download_bhavcopy(d: date, retries: int = 3) -> pd.DataFrame | None:
    urls = build_bhavcopy_urls(d)

    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                response = session.get(url, timeout=30)

                if response.status_code == 404:
                    break   # This URL doesn't have this date — try next URL

                if response.status_code != 200:
                    if attempt < retries:
                        time.sleep(5 * attempt)
                        continue
                    break   # All retries exhausted for this URL

                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    csv_filename = z.namelist()[0]
                    with z.open(csv_filename) as f:
                        raw_df = pd.read_csv(f)

                result = parse_bhavcopy_df(raw_df, d)
                return result   # Success — return immediately

            except zipfile.BadZipFile:
                break   # Bad file for this URL — try next URL
            except Exception as e:
                if attempt < retries:
                    time.sleep(5 * attempt)
                else:
                    break   # Move to next URL

    return None   # All URLs exhausted — holiday or genuine missing date


# ═══════════════════════════════════════════════════════════
# SECTION 3 — SPLITS-BASED ADJUSTMENT FACTORS (C3 + C4 FIX)
# ═══════════════════════════════════════════════════════════
def build_adjustment_factors_from_splits() -> dict:
    """
    Builds cumulative price adjustment factors from corporate_actions table.
    Uses splits only (dividends ignored — too small to affect 21-day models).

    Method: walk backwards from today.
    For each split event on date D with ratio R:
      All prices on dates < D are multiplied by (1 / R).
    """
    print("\n📊 Building adjustment factors from corporate_actions (splits only)...")

    try:
        splits_df = pd.read_sql(
            f"""SELECT Ticker, Date, Ratio
                FROM {TABLES['corporate_actions']}
                WHERE Action_Type = 'split'
                  AND Ratio IS NOT NULL
                  AND Ratio > 0
                ORDER BY Ticker, Date ASC""",
            con=engine
        )
    except Exception as e:
        print(f"  ❌ Cannot read corporate_actions: {e}")
        return {}

    if splits_df.empty:
        print("  ⚠️  No splits found in corporate_actions. All Adj_Close = Close.")
        return {}

    splits_df['Date'] = pd.to_datetime(splits_df['Date']).dt.date

    adj_factors = {}

    for ticker, grp in splits_df.groupby('Ticker'):
        grp          = grp.sort_values('Date', ascending=True).reset_index(drop=True)
        split_dates  = grp['Date'].tolist()
        split_ratios = grp['Ratio'].tolist()
        n            = len(split_dates)

        # cum_factors[i] = product of (1/ratio) for all splits at index >= i
        # i.e. how much to multiply a price at a date just before split_dates[i]
        cum_factors = [1.0] * n
        running = 1.0
        for i in range(n - 1, -1, -1):
            running *= (1.0 / split_ratios[i])
            cum_factors[i] = running

        adj_factors[ticker] = (split_dates, cum_factors)

    print(f"  ✅ Adjustment factors built for {len(adj_factors)} tickers with splits")
    return adj_factors


def get_adj_close(ticker: str, raw_close: float, d: date,
                  adj_factors: dict) -> float:
    if ticker not in adj_factors:
        return round(raw_close, 4)

    split_dates, cum_factors = adj_factors[ticker]

    # bisect_right gives the insertion point for d in split_dates.
    # All elements at index < idx are <= d.
    # The first split AFTER d is at index idx.
    idx = bisect.bisect_right(split_dates, d)

    if idx == len(split_dates):
        # d is after all split dates — no adjustment needed
        return round(raw_close, 4)

    # Apply cumulative factor for splits at index idx onwards
    factor = cum_factors[idx]
    return round(raw_close * factor, 4)


# ═══════════════════════════════════════════════════════════
# SECTION 4 — CORPORATE ACTIONS SAVE
# ═══════════════════════════════════════════════════════════

def save_corporate_actions(tickers: list):
    print(f"\n💾 Saving corporate actions history...")
    all_actions = []
    failed      = []

    for idx, ticker in enumerate(tickers, 1):
        try:
            stock = yf.Ticker(ticker)

            # Splits
            splits = stock.splits
            if splits is not None and not splits.empty:
                for split_date, ratio in splits.items():
                    all_actions.append({
                        'Ticker'      : ticker,
                        'Date'        : split_date.date(),
                        'Action_Type' : 'split',
                        'Ratio'       : round(float(ratio), 6),
                        'Amount'      : None,
                    })

            # Dividends
            divs = stock.dividends
            if divs is not None and not divs.empty:
                for div_date, amount in divs.items():
                    all_actions.append({
                        'Ticker'      : ticker,
                        'Date'        : div_date.date(),
                        'Action_Type' : 'dividend',
                        'Ratio'       : None,
                        'Amount'      : round(float(amount), 4),
                    })

            time.sleep(0.3)

        except Exception as e:
            print(f"  ⚠️  {ticker} corporate actions failed: {e}")
            failed.append(ticker)

        # Batch save every 250 rows to avoid memory buildup
        if len(all_actions) >= 250:
            save_to_db(pd.DataFrame(all_actions), TABLES['corporate_actions'], engine)
            all_actions = []

        if idx % 50 == 0:
            print(f"  {idx}/{len(tickers)} tickers processed...")

    if all_actions:
        save_to_db(pd.DataFrame(all_actions), TABLES['corporate_actions'], engine)

    print(f"  ✅ Corporate actions saved")
    if failed:
        print(f"  ⚠️  {len(failed)} tickers failed")


# ═══════════════════════════════════════════════════════════
# SECTION 5 — VERIFICATION
# ═══════════════════════════════════════════════════════════

def verify_ingestion(all_days: list):
    print("\n" + "=" * 60)
    print("VERIFICATION — Checking ingested data quality")
    print("=" * 60)

    try:
        # Overall summary
        summary = pd.read_sql("""
            SELECT MIN(Date) as Min_Date, MAX(Date) as Max_Date,
                   COUNT(DISTINCT Ticker) as Tickers,
                   COUNT(*) as Total_Rows
            FROM nifty500_ohlcv
        """, con=engine)
        print(f"\n  Min Date      : {summary['Min_Date'].iloc[0]}")
        print(f"  Max Date      : {summary['Max_Date'].iloc[0]}")
        print(f"  Distinct Tickers: {summary['Tickers'].iloc[0]} (expected ~500)")
        print(f"  Total Rows    : {summary['Total_Rows'].iloc[0]:,}")

        # Check for expected date range coverage
        expected_days = len(all_days)
        ingested_days = len(get_already_ingested_dates())
        coverage_pct  = ingested_days / expected_days * 100 if expected_days > 0 else 0
        print(f"\n  Trading days expected : {expected_days:,}")
        print(f"  Trading days ingested : {ingested_days:,}")
        print(f"  Coverage              : {coverage_pct:.1f}%")

        # Tickers with very few days (recent listings or download failures)
        thin = pd.read_sql("""
            SELECT Ticker, COUNT(*) as days
            FROM nifty500_ohlcv
            GROUP BY Ticker
            HAVING days < 100
            ORDER BY days ASC
            LIMIT 15
        """, con=engine)
        if not thin.empty:
            print(f"\n  ⚠️  Tickers with < 100 days (likely recent listings):")
            for _, row in thin.iterrows():
                print(f"      {row['Ticker']}: {row['days']} days")

    except Exception as e:
        print(f"  ❌ Verification query failed: {e}")
        return

    issues = []
    max_date = pd.to_datetime(summary['Max_Date'].iloc[0]).date()
    today    = date.today()
    days_gap = (today - max_date).days

    if days_gap > 5:
        issues.append(f"Max date is {max_date} — {days_gap} days behind today. "
                      f"Recent data may be missing (URL format issue or holidays).")
    if int(summary['Tickers'].iloc[0]) < 450:
        issues.append(f"Only {summary['Tickers'].iloc[0]} tickers found — expected ~500. "
                      f"Some tickers may be recent IPOs (acceptable) or symbol mismatches.")

    if issues:
        print(f"\n  ⚠️  ISSUES DETECTED:")
        for i, issue in enumerate(issues, 1):
            print(f"     {i}. {issue}")
    else:
        print(f"\n  ✅ All checks passed.")

    print("\n" + "=" * 60)
    print("✅ Verification complete. Review the numbers above before running the next script.")
    print("   Next step: python data/indicators.py")


# ═══════════════════════════════════════════════════════════
# SECTION 6 — MAIN INGESTION LOOP
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("NSE BHAVCOPY INGESTION — hedge_v2")
    print(f"Universe: {len(TICKERS)} stocks")
    print(f"Period:   {DATA_START} → today")
    print("=" * 60)

    # ── Step 1: Get all trading days in range ─────────────────────────
    print("\n📅 Building trading day calendar...")
    all_days     = get_trading_days(DATA_START, DATA_END)
    ingested     = get_already_ingested_dates()
    pending_days = [d for d in all_days if d not in ingested]

    print(f"  Total trading days in range : {len(all_days):,}")
    print(f"  Already in database         : {len(ingested):,}")
    print(f"  Remaining to download       : {len(pending_days):,}")

    if not pending_days:
        print("\n✅ All dates already ingested.")
        verify_ingestion(all_days)
        return

    # ── Step 2: Save corporate actions (once) ─────────────────────────
    corp_actions_done = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "corp_actions_saved.flag"
    )
    os.makedirs(os.path.dirname(corp_actions_done), exist_ok=True)

    if not os.path.exists(corp_actions_done):
        save_corporate_actions(TICKERS)
        open(corp_actions_done, 'w').close()
    else:
        print("\n📂 Corporate actions already saved (flag exists).")

    # ── Step 3: Build adjustment factors from splits ──────────────────
    adj_factors = build_adjustment_factors_from_splits()

    # ── Step 4: Download bhavcopy day by day ─────────────────────────
    print(f"\n📥 Downloading {len(pending_days):,} bhavcopy files...\n")

    batch_buffer = []
    SAVE_EVERY   = 50

    success_count = 0
    skip_count    = 0
    fail_count    = 0

    # Refresh NSE session cookie upfront
    try:
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(2)
    except Exception:
        pass

    for idx, d in enumerate(pending_days, 1):
        if idx % 100 == 0 or idx == 1:
            pct = idx / len(pending_days) * 100
            print(f"  Progress: {idx:,}/{len(pending_days):,} ({pct:.1f}%) "
                  f"| ✅ {success_count} | ⏭️  {skip_count} | ❌ {fail_count}")

        # Refresh NSE session cookie every 20 requests
        # NSE cookies expire quickly — without this, requests return empty/blocked
        if idx % 20 == 0:
            try:
                session.get("https://www.nseindia.com", timeout=15)
                time.sleep(1)
            except Exception:
                pass

        raw_df = download_bhavcopy(d)

        if raw_df is None or raw_df.empty:
            skip_count += 1
            continue

        rows = []
        for _, row in raw_df.iterrows():
            ticker    = row['Ticker']
            raw_close = float(row['CLOSE']) if pd.notna(row['CLOSE']) else None
            if raw_close is None:
                continue

            adj_close     = get_adj_close(ticker, raw_close, d, adj_factors)
            high          = float(row['HIGH']) if pd.notna(row.get('HIGH')) else raw_close
            low           = float(row['LOW'])  if pd.notna(row.get('LOW'))  else raw_close
            typical_price = round((high + low + raw_close) / 3, 4)

            rows.append({
                'Date'          : d,
                'Ticker'        : ticker,
                'Open'          : round(float(row['OPEN']), 4) if pd.notna(row.get('OPEN')) else None,
                'High'          : round(high, 4),
                'Low'           : round(low, 4),
                'Close'         : round(raw_close, 4),
                'Adj_Close'     : adj_close,
                'Volume'        : int(row['VOLUME']) if pd.notna(row.get('VOLUME')) else None,
                'Typical_Price' : typical_price,
                'VWAP_Daily'    : typical_price,
            })

        if rows:
            batch_buffer.extend(rows)
            success_count += 1

        if len(batch_buffer) >= SAVE_EVERY * len(TICKERS):
            batch_df = pd.DataFrame(batch_buffer)
            save_to_db(batch_df, TABLES['ohlcv'], engine)
            batch_buffer = []

        time.sleep(0.5)

        if idx % 200 == 0:
            print(f"  ⏸️  Pausing 15s to avoid NSE rate limiting...")
            time.sleep(15)

    # Save any remaining rows after loop ends
    if batch_buffer:
        batch_df = pd.DataFrame(batch_buffer)
        save_to_db(batch_df, TABLES['ohlcv'], engine)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"""
{'='*60}
BHAVCOPY INGESTION COMPLETE
{'='*60}
  Days downloaded successfully  : {success_count:,}
  Days skipped (holiday/missing): {skip_count:,}
  Days failed                   : {fail_count:,}
  Total trading days in range   : {len(all_days):,}
{'='*60}""")

    # ── Verification ──────────────────────────────────────────────────
    verify_ingestion(all_days)


if __name__ == "__main__":
    main()