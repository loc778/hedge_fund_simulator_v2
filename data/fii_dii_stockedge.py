"""
fii_dii_stockedge.py
---------------------
TWO MODES — both run on every execution:

FII_Source_Flag tells features.py which resolution each row has:
  'monthly' = only monthly total available (historical)
  'daily'   = actual daily value available (recent ~50 days)

"""

import os
import sys
import time
import logging
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import requests
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/fii_dii_stockedge.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

API_URL = "https://api.stockedge.com/Api/FIIDashboardApi/GetFIIDIIProvisional"

BROWSER_HEADERS = {
    "Accept":             "application/json, text/plain, */*",
    "Accept-Encoding":    "gzip, deflate, br, zstd",
    "Accept-Language":    "en-US,en;q=0.7",
    "Connection":         "keep-alive",
    "Origin":             "https://web.stockedge.com",
    "Referer":            "https://web.stockedge.com/",
    "Sec-Ch-Ua":          '"Brave";v="147", "Not.A.Brand";v="8", "Chromium";v="147"',
    "Sec-Ch-Ua-Mobile":   "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest":     "empty",
    "Sec-Fetch-Mode":     "cors",
    "Sec-Fetch-Site":     "same-site",
    "Sec-Gpc":            "1",
    "User-Agent":         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
}

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,  "May": 5,  "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# ── Session ───────────────────────────────────────────────────────────────────

def build_session() -> requests.Session:
    """Establishes a browser-like session to avoid SSL reset from StockEdge."""
    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)
    log.info("Establishing browser session with StockEdge...")
    try:
        r = session.get("https://web.stockedge.com/", timeout=30)
        r.raise_for_status()
        log.info(f"  Session ready. Status: {r.status_code}")
    except Exception as e:
        log.warning(f"  Homepage prefetch failed ({e}) — proceeding anyway")
    time.sleep(1)
    return session

# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch(session: requests.Session, fii_dii_type: str, time_span: str) -> list:
    """Raw fetch — returns list of dicts from StockEdge API."""
    params = {"FiiDiiType": fii_dii_type, "TimeSpan": time_span, "lang": "en"}
    resp = session.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

# ── Monthly parsing ───────────────────────────────────────────────────────────

def parse_monthly(raw: list) -> pd.DataFrame:
    """
    Parses TimeSpan=M response.
    DateText format: "Mar 2026" — year is explicit.
    Returns DataFrame: year, month, net_value
    """
    rows = []
    for item in raw:
        if item.get("NetValue") is None:
            continue
        parts = item["DateText"].strip().split()   # ["Mar", "2026"]
        rows.append({
            "year":      int(parts[1]),
            "month":     MONTH_MAP[parts[0]],
            "net_value": float(item["NetValue"]),
        })
    return pd.DataFrame(rows)

# ── Daily parsing ─────────────────────────────────────────────────────────────

def parse_daily(raw: list) -> pd.DataFrame:
    """
    Parses TimeSpan=D response.
    DateText format: "Apr 16" — NO year. Year inferred by walking backwards
    from today: when the month rolls back past the current month, decrement year.
    """
    rows = []
    today      = date.today()
    cur_year   = today.year
    prev_month = today.month  # tracks last seen month to detect year rollover

    for item in raw:
        if item.get("NetValue") is None:
            continue

        parts = item["DateText"].strip().split()   # ["Apr", "16"]
        month = MONTH_MAP[parts[0]]
        day   = int(parts[1])

        # If month jumped forward vs previous month, we crossed a year boundary
        if month > prev_month:
            cur_year -= 1

        prev_month = month

        rows.append({
            "date":      date(cur_year, month, day),
            "net_value": float(item["NetValue"]),
        })

    return pd.DataFrame(rows)

# ── DB helpers ────────────────────────────────────────────────────────────────

def get_trading_days(engine, year: int, month: int) -> list:
    sql = text("""
        SELECT DISTINCT Date FROM nifty500_ohlcv
        WHERE YEAR(Date) = :yr AND MONTH(Date) = :mo
        ORDER BY Date
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"yr": year, "mo": month}).fetchall()
    return [r[0] for r in rows]

# ── Write monthly ─────────────────────────────────────────────────────────────

def get_latest_monthly(engine):
    """Returns (year, month) of the latest month already written, or (0, 0) if none."""
    sql = text("""
        SELECT YEAR(Date) AS yr, MONTH(Date) AS mo
        FROM macro_indicators
        WHERE FII_Monthly_Net_Cr IS NOT NULL
        ORDER BY Date DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql).fetchone()
    if row:
        return row.yr, row.mo
    return 0, 0


def write_monthly(engine, fii_df: pd.DataFrame, dii_df: pd.DataFrame):
    """
    Merges FII + DII monthly frames, expands to trading-day level,
    upserts FII_Monthly_Net_Cr + DII_Monthly_Net_Cr into macro_indicators.
    Skips months already present — only writes new months.
    """
    latest_yr, latest_mo = get_latest_monthly(engine)
    if latest_yr:
        log.info(f"Monthly — latest in DB: {latest_yr}-{latest_mo:02d}. Skipping older months.")

    merged = pd.merge(fii_df, dii_df, on=["year", "month"],
                      suffixes=("_fii", "_dii"))

    upsert_sql = text("""
        INSERT INTO macro_indicators
            (Date, FII_Monthly_Net_Cr, DII_Monthly_Net_Cr, FII_Source_Flag)
        VALUES
            (:date, :fii_net, :dii_net, 'monthly')
        ON DUPLICATE KEY UPDATE
            FII_Monthly_Net_Cr = VALUES(FII_Monthly_Net_Cr),
            DII_Monthly_Net_Cr = VALUES(DII_Monthly_Net_Cr),
            FII_Source_Flag    = IF(FII_Daily_Net_Cr IS NOT NULL, 'daily', 'monthly')
    """)

    total   = 0
    skipped = 0

    with engine.begin() as conn:
        for _, row in merged.iterrows():
            yr, mo = int(row["year"]), int(row["month"])

            # Skip months already in DB — only process newer months
            if latest_yr and (yr < latest_yr or (yr == latest_yr and mo < latest_mo)):
                skipped += 1
                continue

            trading_days = get_trading_days(engine, yr, mo)
            if not trading_days:
                skipped += 1
                continue
            for d in trading_days:
                conn.execute(upsert_sql, {
                    "date":    d,
                    "fii_net": row["net_value_fii"],
                    "dii_net": row["net_value_dii"],
                })
            total += len(trading_days)

    log.info(f"Monthly — trading days written : {total}")
    log.info(f"Monthly — months skipped       : {skipped}")

# ── Write daily ───────────────────────────────────────────────────────────────

def get_latest_daily(engine):
    """Returns latest date with FII_Daily_Net_Cr already written, or None."""
    sql = text("""
        SELECT MAX(Date) AS latest
        FROM macro_indicators
        WHERE FII_Daily_Net_Cr IS NOT NULL
    """)
    with engine.connect() as conn:
        row = conn.execute(sql).fetchone()
    return row.latest if row and row.latest else None


def write_daily(engine, fii_df: pd.DataFrame, dii_df: pd.DataFrame):
    """
    Merges FII + DII daily frames, upserts FII_Daily_Net_Cr + DII_Daily_Net_Cr
    into macro_indicators. Skips dates already present.
    Daily values take precedence over monthly for the same date.
    """
    latest = get_latest_daily(engine)
    if latest:
        log.info(f"Daily  — latest in DB: {latest}. Skipping older dates.")

    merged = pd.merge(fii_df, dii_df, on="date", suffixes=("_fii", "_dii"))

    upsert_sql = text("""
        INSERT INTO macro_indicators
            (Date, FII_Daily_Net_Cr, DII_Daily_Net_Cr, FII_Source_Flag)
        VALUES
            (:date, :fii_net, :dii_net, 'daily')
        ON DUPLICATE KEY UPDATE
            FII_Daily_Net_Cr = VALUES(FII_Daily_Net_Cr),
            DII_Daily_Net_Cr = VALUES(DII_Daily_Net_Cr),
            FII_Source_Flag  = 'daily'
    """)

    total   = 0
    skipped = 0

    with engine.begin() as conn:
        for _, row in merged.iterrows():
            if latest and row["date"] <= latest:
                skipped += 1
                continue
            conn.execute(upsert_sql, {
                "date":    row["date"],
                "fii_net": row["net_value_fii"],
                "dii_net": row["net_value_dii"],
            })
            total += 1

    log.info(f"Daily  — trading days written : {total}")
    log.info(f"Daily  — dates skipped        : {skipped}")

# ── Verify ────────────────────────────────────────────────────────────────────

def verify(engine):
    sql = text("""
        SELECT
            SUM(CASE WHEN FII_Monthly_Net_Cr IS NOT NULL THEN 1 ELSE 0 END) AS monthly_filled,
            SUM(CASE WHEN FII_Daily_Net_Cr   IS NOT NULL THEN 1 ELSE 0 END) AS daily_filled,
            SUM(CASE WHEN DII_Monthly_Net_Cr IS NOT NULL THEN 1 ELSE 0 END) AS dii_monthly_filled,
            SUM(CASE WHEN DII_Daily_Net_Cr   IS NOT NULL THEN 1 ELSE 0 END) AS dii_daily_filled,
            MIN(CASE WHEN FII_Monthly_Net_Cr IS NOT NULL THEN Date END)      AS monthly_earliest,
            MAX(CASE WHEN FII_Daily_Net_Cr   IS NOT NULL THEN Date END)      AS daily_latest
        FROM macro_indicators
    """)
    with engine.connect() as conn:
        r = conn.execute(sql).fetchone()

    log.info("── Verification ──────────────────────────────────────────")
    log.info(f"  FII_Monthly_Net_Cr filled : {r.monthly_filled} rows (from {r.monthly_earliest})")
    log.info(f"  DII_Monthly_Net_Cr filled : {r.dii_monthly_filled} rows")
    log.info(f"  FII_Daily_Net_Cr   filled : {r.daily_filled} rows (latest: {r.daily_latest})")
    log.info(f"  DII_Daily_Net_Cr   filled : {r.dii_daily_filled} rows")

    # Spot-check: Mar 2020 monthly FII
    spot = text("""
        SELECT Date, FII_Monthly_Net_Cr, DII_Monthly_Net_Cr, FII_Source_Flag
        FROM macro_indicators
        WHERE YEAR(Date)=2020 AND MONTH(Date)=3
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(spot).fetchone()
    if row:
        ok = row.FII_Monthly_Net_Cr is not None and row.FII_Monthly_Net_Cr < -60000
        log.info(f"  Spot Mar 2020 FII        : {row.FII_Monthly_Net_Cr} "
                 f"({'PASS' if ok else 'FAIL'})")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine  = get_engine()

    session = build_session()

    # ── Monthly ──
    log.info("=== MODE 1: MONTHLY ===")
    fii_monthly_raw = fetch(session, "fii", "M")
    time.sleep(1)
    dii_monthly_raw = fetch(session, "dii", "M")
    time.sleep(1)

    fii_monthly = parse_monthly(fii_monthly_raw)
    dii_monthly = parse_monthly(dii_monthly_raw)
    log.info(f"  FII monthly: {len(fii_monthly)} months | "
             f"DII monthly: {len(dii_monthly)} months")

    write_monthly(engine, fii_monthly, dii_monthly)

    # ── Daily ──
    log.info("=== MODE 2: DAILY ===")
    fii_daily_raw = fetch(session, "fii", "D")
    time.sleep(1)
    dii_daily_raw = fetch(session, "dii", "D")

    fii_daily = parse_daily(fii_daily_raw)
    dii_daily = parse_daily(dii_daily_raw)
    log.info(f"  FII daily: {len(fii_daily)} days | "
             f"DII daily: {len(dii_daily)} days")

    write_daily(engine, fii_daily, dii_daily)

    verify(engine)