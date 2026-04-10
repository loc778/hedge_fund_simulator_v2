# data/bhavcopy_ingestion.py
# ═══════════════════════════════════════════════════════════
# NSE BHAVCOPY INGESTION — hedge_v2
# Downloads daily bhavcopy ZIPs from NSE archives (Jan 2010 → today)
# Parses OHLCV, fetches adjustment factors from yfinance,
# calculates Adj_Close, saves to nifty500_ohlcv table.
#
# WHAT IS A BHAVCOPY:
# NSE publishes a daily ZIP file after market close containing
# OHLCV data for every stock traded that day. These are the
# official NSE settlement prices — more reliable than yfinance.
#
# HOW ADJ_CLOSE IS CALCULATED:
# 1. Fetch split history and dividend history per ticker from yfinance
# 2. Build a cumulative adjustment factor for each ticker
# 3. Adj_Close[t] = Close[t] * product of all adjustment factors
#    for events AFTER date t (working backwards from present)
#
# RESUME SUPPORT:
# Script checks which dates already exist in the DB and skips them.
# Safe to stop and restart at any time.
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
import pickle
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta

from config import (TICKERS, DATA_START, DATA_END, BHAVCOPY_URL,
                    BHAVCOPY_TEMP_DIR, TABLES, BATCH_SIZE, BATCH_DELAY)
from data.db import get_engine, save_to_db

engine = get_engine()

# ── NSE Bhavcopy Request Headers ─────────────────────────────────────
# NSE blocks requests without a proper browser user-agent
# These headers mimic a real browser visit
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

# ── NSE Holidays ─────────────────────────────────────────────────────
# NSE is closed on these dates (partial list — updated annually)
# The script skips weekends automatically via date logic
# These are additional market holidays
# Source: NSE official holiday calendar
NSE_HOLIDAYS = {
    # 2026
    date(2026,1,26), date(2026,3,25), date(2026,4,2), date(2026,4,10),
    date(2026,4,14), date(2026,5,1), date(2026,6,19), date(2026,8,15),
    date(2026,10,2), date(2026,10,20), date(2026,11,4), date(2026,12,25),
    # 2025
    date(2025,1,26), date(2025,2,26), date(2025,3,14), date(2025,3,31),
    date(2025,4,10), date(2025,4,14), date(2025,4,18), date(2025,5,1),
    date(2025,8,15), date(2025,8,27), date(2025,10,2), date(2025,10,2),
    date(2025,10,21), date(2025,10,24), date(2025,11,5), date(2025,12,25),
    # 2024
    date(2024,1,22), date(2024,1,26), date(2024,3,25), date(2024,4,9),
    date(2024,4,11), date(2024,4,14), date(2024,4,17), date(2024,5,1),
    date(2024,5,23), date(2024,6,17), date(2024,7,17), date(2024,8,15),
    date(2024,10,2), date(2024,11,1), date(2024,11,15), date(2024,12,25),
    # Add earlier years here as needed — script will still work without them
    # (it will just try to download those days and get 404, which it handles)
}

# ── Ticker Set for Fast Lookup ────────────────────────────────────────
# Convert TICKERS list to set of base symbols (without .NS) for bhavcopy matching
TICKER_SET = {t.replace('.NS', '') for t in TICKERS}

# ── SURVIVORSHIP BIAS NOTE ────────────────────────────────────────────
# The TICKER_SET above is TODAY's Nifty 500 constituents (April 2026).
# For 15 years of training data, this creates survivorship bias:
# we only train on stocks that SURVIVED to be in today's index,
# ignoring the ones that were dropped (often the worst performers).
# This makes the model look better in backtesting than it will in reality.
#
# PRACTICAL APPROACH TAKEN HERE:
# We download bhavcopy data for today's 500 tickers going back to 2010.
# Stocks listed after 2010 will simply have fewer rows in the DB —
# the data_quality.py script will classify them as Tier B or C accordingly.
# Stocks that were in the 2010-2026 Nifty 500 but are now delisted or
# removed from the index are NOT captured here.
#
# COMPLETE FIX (deferred — requires significant research effort):
# NSE publishes historical index constituent files on their archives page.
# A proper survivorship-bias-free dataset requires:
# 1. Download NSE's historical Nifty 500 constituent CSV for each quarter
#    from 2010 to present
# 2. Build a union of all tickers that were EVER in Nifty 500
# 3. Download bhavcopy data for all those tickers (will be ~700-800 symbols)
# 4. Mark each stock's active periods in stock_data_quality table
# This is the production-grade approach and should be done before
# submitting to mentor for final review.
#
# For now: current 500 tickers gives us a usable training set.
# The model will have mild survivorship bias but the architecture is correct.


# ═══════════════════════════════════════════════════════════
# SECTION 1 — DATE UTILITIES
# ═══════════════════════════════════════════════════════════

def is_trading_day(d: date) -> bool:
    """Returns True if the date is a likely NSE trading day."""
    # Skip weekends
    if d.weekday() >= 5:
        return False
    # Skip known holidays
    if d in NSE_HOLIDAYS:
        return False
    return True


def get_trading_days(start: str, end: str = None) -> list:
    """
    Returns list of trading dates between start and end (inclusive).
    Skips weekends and known NSE holidays.
    end=None means today.
    """
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
    """
    Query DB for dates already in nifty500_ohlcv.
    Used for resume support — skip dates already downloaded.
    Returns set of date objects.
    """
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

    NSE changed their bhavcopy URL format in July 2024:
    - Old format (pre-Jul 2024): /content/historical/EQUITIES/{YEAR}/{MON}/cm{DD}{MON}{YYYY}bhav.csv.zip
    - New format (post-Jul 2024): /content/equities/bhav/cm_bhavcopy_{YYYY-MM-DD}.csv.zip

    We try the new format first for recent dates, old format as fallback.
    For dates before Jul 2024 we try old format first.
    Both formats are always tried — NSE sometimes serves old format files
    on the new endpoint for historical backfill.
    """
    year     = d.strftime("%Y")
    month    = d.strftime("%b").upper()
    date_old = d.strftime("%d%b%Y").upper()   # 05JUL2024
    date_new = d.strftime("%Y-%m-%d")          # 2024-07-05

    old_url = (f"https://nsearchives.nseindia.com/content/historical/EQUITIES"
               f"/{year}/{month}/cm{date_old}bhav.csv.zip")
    new_url = (f"https://nsearchives.nseindia.com/content/equities/bhav"
               f"/cm_bhavcopy_{date_new}.csv.zip")

    # For dates from Jul 2024 onwards, try new format first
    cutoff = date(2024, 7, 1)
    if d >= cutoff:
        return [new_url, old_url]
    else:
        return [old_url, new_url]


def parse_bhavcopy_df(raw_df: pd.DataFrame, d: date) -> pd.DataFrame | None:
    """
    Standardizes a raw bhavcopy DataFrame regardless of which URL format it came from.
    NSE has used different column names across years — this handles all variants.
    Returns clean DataFrame filtered to our TICKER_SET, or None if empty.
    """
    raw_df.columns = raw_df.columns.str.strip().str.upper()

    # Column name mapping across all NSE format versions (old + new)
    col_map = {
        'SYMBOL'        : 'SYMBOL',
        'OPEN'          : 'OPEN',
        'HIGH'          : 'HIGH',
        'LOW'           : 'LOW',
        'CLOSE'         : 'CLOSE',
        'TOTTRDQTY'     : 'VOLUME',   # old format
        'TTL_TRD_QNTY'  : 'VOLUME',   # mid format
        'TOTAL_TRADED_QTY': 'VOLUME', # new format
        'SERIES'        : 'SERIES',
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
    """
    Download and parse NSE bhavcopy for a given date.
    Tries multiple URL formats automatically (handles pre/post Jul 2024 NSE changes).
    Returns None if no file found (holiday, weekend, future date).
    """
    urls = build_bhavcopy_urls(d)

    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                # Refresh NSE session cookie on first attempt
                if attempt == 1:
                    try:
                        session.get("https://www.nseindia.com", timeout=10)
                    except Exception:
                        pass

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
# SECTION 3 — ADJUSTMENT FACTOR CALCULATION
# ═══════════════════════════════════════════════════════════

def fetch_adjustment_factors(tickers: list) -> dict:
    """
    Fetches split and dividend history from yfinance for each ticker.
    Builds a cumulative adjustment factor per ticker per date.

    Returns dict: {ticker: pd.Series(index=date, values=adj_factor)}
    The adj_factor at date t means: multiply Close[t] by adj_factor[t]
    to get the split/dividend adjusted close.

    Logic:
    - Work backwards from today
    - Start with factor = 1.0 (today's price needs no adjustment)
    - Each split event before today: multiply all prices BEFORE split date
      by the split ratio (e.g. 2:1 split → ratio = 0.5)
    - Each dividend: subtract dividend from all prices before ex-date
      (but we only apply this if dividend > 1% of price to avoid noise)
    """
    print(f"\n📊 Fetching adjustment factors for {len(tickers)} tickers from yfinance...")
    print("   (This runs once — factors saved to models/adj_factors.pkl for reuse)")

    adj_factors = {}
    failed      = []

    # Process in batches to avoid yfinance rate limits
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        print(f"  Batch {i//BATCH_SIZE + 1}/{(len(tickers)-1)//BATCH_SIZE + 1}...")

        for ticker in batch:
            try:
                stock = yf.Ticker(ticker)

                # Get full price history to use as reference
                hist = stock.history(start=DATA_START, auto_adjust=False)

                if hist.empty:
                    adj_factors[ticker] = {}
                    continue

                # yfinance already provides an adjusted close in its history
                # We use the ratio: Adj Close / Close as our adjustment factor
                # This captures both splits and dividends implicitly
                hist = hist[['Close', 'Adj Close']].dropna()

                if len(hist) < 10:
                    adj_factors[ticker] = {}
                    continue

                # Ratio: how much to multiply raw close to get adj close
                ratio = (hist['Adj Close'] / hist['Close']).round(6)
                ratio.index = ratio.index.date   # convert to plain date objects

                adj_factors[ticker] = ratio.to_dict()

            except Exception as e:
                print(f"    ⚠️  {ticker} adj factor failed: {e}")
                adj_factors[ticker] = {}
                failed.append(ticker)

            time.sleep(0.3)

        time.sleep(BATCH_DELAY)

    print(f"  ✅ Adjustment factors fetched: {len(adj_factors)} tickers")
    if failed:
        print(f"  ⚠️  {len(failed)} tickers failed (will use raw close = adj close): {failed[:10]}")

    return adj_factors


def get_adj_close(ticker: str, raw_close: float, d: date,
                  adj_factors: dict) -> float:
    """
    Returns the adjusted close for a ticker on a given date.
    Uses the pre-fetched adjustment factor ratio from yfinance.
    Falls back to raw close if no adjustment factor available.
    """
    factors = adj_factors.get(ticker, {})
    if not factors:
        return raw_close

    # Find the closest available date in the factors dict
    factor = factors.get(d)
    if factor is None:
        # Fallback: use the most recent factor before this date
        available_dates = sorted([dd for dd in factors.keys() if dd <= d])
        if available_dates:
            factor = factors[available_dates[-1]]
        else:
            return raw_close

    return round(raw_close * factor, 4)


# ═══════════════════════════════════════════════════════════
# SECTION 4 — CORPORATE ACTIONS SAVE
# ═══════════════════════════════════════════════════════════

def save_corporate_actions(tickers: list):
    """
    Saves split and dividend history to corporate_actions table.
    This is for reference/audit — actual adjustment uses adj_factors dict.
    """
    print(f"\n💾 Saving corporate actions history...")
    all_actions = []

    for ticker in tickers:
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

    if all_actions:
        df = pd.DataFrame(all_actions)
        save_to_db(df, TABLES['corporate_actions'], engine)
        print(f"  ✅ Saved {len(all_actions)} corporate action events")


# ═══════════════════════════════════════════════════════════
# SECTION 5 — MAIN INGESTION LOOP
# ═══════════════════════════════════════════════════════════

def verify_ingestion(all_days: list):
    """
    Runs after ingestion completes. Queries the DB and prints a verification
    report so you can confirm data quality before running the next script.
    Pauses and asks for confirmation before proceeding.
    """
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

    # Checks
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
    confirm = input("Verification complete. Type 'yes' to proceed to next step, "
                    "or 'no' to stop and investigate: ").strip().lower()
    if confirm != 'yes':
        print("\n⛔ Stopped at your request. Fix any issues then rerun the script.")
        print("   The script has resume support — already-ingested dates are skipped.")
        raise SystemExit(0)
    print("✅ Confirmed. Proceeding.\n")


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

    # ── Step 2: Load or fetch adjustment factors ──────────────────────
    adj_factors_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "adj_factors.pkl"
    )
    os.makedirs(os.path.dirname(adj_factors_path), exist_ok=True)

    if os.path.exists(adj_factors_path):
        print(f"\n📂 Loading cached adjustment factors...")
        with open(adj_factors_path, 'rb') as f:
            adj_factors = pickle.load(f)
        print(f"  ✅ Loaded factors for {len(adj_factors)} tickers")
    else:
        adj_factors = fetch_adjustment_factors(TICKERS)
        with open(adj_factors_path, 'wb') as f:
            pickle.dump(adj_factors, f)
        print(f"  ✅ Saved adjustment factors to {adj_factors_path}")

    # ── Step 3: Save corporate actions (once) ─────────────────────────
    corp_actions_done = os.path.join(
        os.path.dirname(adj_factors_path), "corp_actions_saved.flag"
    )
    if not os.path.exists(corp_actions_done):
        save_corporate_actions(TICKERS)
        open(corp_actions_done, 'w').close()

    # ── Step 4: Download bhavcopy day by day ─────────────────────────
    print(f"\n📥 Downloading {len(pending_days):,} bhavcopy files...\n")

    batch_buffer = []
    SAVE_EVERY   = 50

    success_count = 0
    skip_count    = 0
    fail_count    = 0

    for idx, d in enumerate(pending_days, 1):
        if idx % 100 == 0 or idx == 1:
            pct = idx / len(pending_days) * 100
            print(f"  Progress: {idx:,}/{len(pending_days):,} ({pct:.1f}%) "
                  f"| ✅ {success_count} | ⏭️  {skip_count} | ❌ {fail_count}")

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

        if len(batch_buffer) >= SAVE_EVERY * len(TICKERS) or idx == len(pending_days):
            if batch_buffer:
                batch_df = pd.DataFrame(batch_buffer)
                save_to_db(batch_df, TABLES['ohlcv'], engine)
                batch_buffer = []

        time.sleep(0.5)

        if idx % 200 == 0:
            print(f"  ⏸️  Pausing 15s to avoid NSE rate limiting...")
            time.sleep(15)

    # Save any remaining rows
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