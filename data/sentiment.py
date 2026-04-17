# data/sentiment.py
# ═══════════════════════════════════════════════════════════════════════
# SENTIMENT INGESTION — hedge_v2 (Layer 3B)
#
# Two data sources:
#   1. NSE Corporate Announcements — structured events (earnings, buyback,
#      pledge, rating, etc.), rule-scored at ingest. Historical depth from
#      SENTIMENT_START_DATE (config.py).
#   2. Moneycontrol RSS — unstructured headlines. Current snapshot only;
#      no historical backfill possible (RSS = today-forward). Raw text
#      stored; FinBERT scoring deferred to separate Colab notebook.
#
# Writes to two tables:
#   - nifty500_sentiment_raw : one row per event/headline
#   - nifty500_sentiment     : daily aggregate per (Ticker, Date),
#                              rebuilt from _raw via --mode aggregate
#
# THREE MODES (CLI):
#   --mode backfill   → NSE announcements SENTIMENT_START_DATE → today
#                       (50-day window batches) + current Moneycontrol RSS.
#                       Then runs aggregate automatically.
#   --mode daily      → last SENTIMENT['daily_lookback_days'] days, both
#                       sources. Then runs aggregate automatically.
#   --mode aggregate  → rebuilds nifty500_sentiment from nifty500_sentiment_raw.
#                       Safe to re-run anytime.
#
# NSE SESSION PATTERN (reused from bhavcopy_ingestion.py):
#   - Shared requests.Session() with browser User-Agent
#   - Homepage cookie refresh before first request and every 20 API calls
#   - Exponential backoff on 401/429/5xx
#
# FUZZY MATCHING (Moneycontrol only):
#   - Ticker → Company-name map built from NIFTY500_SECTORS_CSV
#   - rapidfuzz.process.extractOne with threshold from config
#   - One headline can match multiple tickers (stored as separate rows)
#
# RULE SCORING (NSE only):
#   - Case-insensitive keyword match against subject + description
#   - Categories ordered in config; first hit wins
#   - Score range [-1.0, +1.0]
# ═══════════════════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import re
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import feedparser
from rapidfuzz import process, fuzz
from sqlalchemy import text

from config import (
    TICKERS, TABLES, SENTIMENT, SENTIMENT_START_DATE,
    NSE_ANNOUNCEMENT_URL, MONEYCONTROL_RSS_FEEDS, NSE_ANNOUNCEMENT_RULES,
    NIFTY500_SECTORS_CSV, NSE_HOLIDAYS,
)
from data.db import get_engine, save_to_db

engine = get_engine()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — NSE SESSION (reused pattern from bhavcopy_ingestion.py)
# ═══════════════════════════════════════════════════════════════════════

HEADERS = {
    "User-Agent"      : ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/120.0.0.0 Safari/537.36"),
    "Accept"          : "application/json, text/plain, */*",
    "Accept-Language" : "en-US,en;q=0.9",
    "Accept-Encoding" : "gzip, deflate, br",
    "Connection"      : "keep-alive",
    "Referer"         : "https://www.nseindia.com/companies-listing/"
                        "corporate-filings-announcements",
}

nse_session = requests.Session()
nse_session.headers.update(HEADERS)


def refresh_nse_cookies():
    """Refresh NSE session cookies. Call before first request and every 20."""
    try:
        nse_session.get("https://www.nseindia.com", timeout=15)
        time.sleep(2)
        nse_session.get(
            "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
            timeout=15
        )
        time.sleep(1)
    except Exception as e:
        print(f"  ⚠️  Cookie refresh failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — TICKER → COMPANY NAME MAP (for fuzzy matching)
# ═══════════════════════════════════════════════════════════════════════

def build_ticker_name_map():
    """
    Build ticker → list of name aliases from nifty500_sectors.csv.
    Returns: { 'RELIANCE.NS': ['Reliance Industries', 'Reliance'], ... }
    """
    try:
        df = pd.read_csv(NIFTY500_SECTORS_CSV)
    except FileNotFoundError:
        print(f"⚠️  {NIFTY500_SECTORS_CSV} not found. Fuzzy matching disabled.")
        return {}

    # Common suffixes to strip for additional alias
    suffixes = [
        r"\s+Limited$", r"\s+Ltd\.?$", r"\s+Corporation$", r"\s+Corp\.?$",
        r"\s+Industries$", r"\s+Enterprises$", r"\s+Company$", r"\s+Co\.?$",
    ]
    suffix_re = re.compile("|".join(suffixes), flags=re.IGNORECASE)

    ticker_map = {}
    for _, row in df.iterrows():
        symbol = str(row.get("Symbol", "")).strip()
        full_name = str(row.get("Company Name", "")).strip()
        if not symbol or not full_name:
            continue

        ticker = f"{symbol}.NS"
        aliases = {full_name}
        short = suffix_re.sub("", full_name).strip()
        if short and short != full_name:
            aliases.add(short)
        # Add ticker symbol itself as alias (often appears in headlines)
        aliases.add(symbol)

        ticker_map[ticker] = list(aliases)

    return ticker_map


TICKER_NAME_MAP = build_ticker_name_map()

# Flat list for rapidfuzz: (alias, ticker)
ALIAS_FLAT = [(alias, ticker)
              for ticker, aliases in TICKER_NAME_MAP.items()
              for alias in aliases]
ALIAS_STRINGS = [a for a, _ in ALIAS_FLAT]
ALIAS_TO_TICKER = {a: t for a, t in ALIAS_FLAT}


def match_tickers_in_text(text_blob: str, threshold: int = None) -> list:
    """
    Find all tickers whose company name/alias appears in text_blob.
    Returns list of unique tickers. One headline can match multiple tickers.
    """
    if not text_blob or not ALIAS_STRINGS:
        return []

    threshold = threshold or SENTIMENT.get("fuzzy_match_threshold", 85)
    matched = set()

    # Token-based partial matching — faster than full-string fuzzy on long text
    # Extract candidate phrases (words + 2-3 word n-grams)
    words = re.findall(r"[A-Za-z&]+(?:\s+[A-Za-z&]+){0,2}", text_blob)
    candidates = set(words)

    for cand in candidates:
        if len(cand) < 4:
            continue
        result = process.extractOne(
            cand, ALIAS_STRINGS,
            scorer=fuzz.ratio,
            score_cutoff=threshold,
        )
        if result:
            matched.add(ALIAS_TO_TICKER[result[0]])

    return list(matched)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — NSE CORPORATE ANNOUNCEMENTS
# ═══════════════════════════════════════════════════════════════════════

# Map NSE symbol (without .NS) → full ticker for lookup
NSE_SYMBOL_TO_TICKER = {t.replace(".NS", ""): t for t in TICKERS}


def score_announcement(subject: str, description: str) -> tuple:
    """
    Apply rule-based scoring. Returns (category, score) or (None, None).
    Case-insensitive keyword match. First matching category wins.
    """
    blob = f"{subject or ''} {description or ''}".lower()
    for category, spec in NSE_ANNOUNCEMENT_RULES.items():
        for keyword in spec["keywords"]:
            if keyword.lower() in blob:
                return category, float(spec["score"])
    return None, None


def fetch_nse_announcements(from_date: date, to_date: date,
                            max_retries: int = 3) -> list:
    """
    Fetch NSE corporate announcements for a date window.
    Returns list of dicts (raw JSON items).
    """
    url = NSE_ANNOUNCEMENT_URL.format(
        from_date=from_date.strftime("%d-%m-%Y"),
        to_date=to_date.strftime("%d-%m-%Y"),
    )

    for attempt in range(max_retries):
        try:
            resp = nse_session.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                # NSE returns either a list directly or a dict with data key
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return data.get("data", data.get("rows", []))
                return []
            elif resp.status_code in (401, 403):
                refresh_nse_cookies()
                time.sleep(2 ** attempt)
            elif resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt * 2)
            else:
                print(f"  ⚠️  NSE API HTTP {resp.status_code} for "
                      f"{from_date}→{to_date}")
                return []
        except requests.RequestException as e:
            print(f"  ⚠️  NSE request failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)

    return []


def parse_announcement(item: dict) -> dict | None:
    """
    Parse one NSE announcement JSON item into a standardized row.
    Returns None if item doesn't map to a universe ticker.
    """
    # NSE field names vary across payload versions — try the common ones
    symbol = (item.get("symbol") or item.get("Symbol") or
              item.get("sym") or "").strip().upper()
    ticker = NSE_SYMBOL_TO_TICKER.get(symbol)
    if not ticker:
        return None

    subject = (item.get("desc") or item.get("subject") or
               item.get("Subject") or "").strip()
    description = (item.get("attchmntText") or item.get("smIndustry") or
                   item.get("descrip") or "").strip()
    url = item.get("attchmntFile") or item.get("attachmentUrl") or ""

    # Date parsing — NSE uses multiple formats
    raw_date = (item.get("an_dt") or item.get("date") or
                item.get("dt") or item.get("announcementDate"))
    if not raw_date:
        return None

    parsed_date = None
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d", "%d/%m/%Y"):
        try:
            parsed_date = datetime.strptime(raw_date.strip(), fmt).date()
            break
        except (ValueError, AttributeError):
            continue
    if not parsed_date:
        return None

    category, score = score_announcement(subject, description)

    return {
        "Date"          : parsed_date,
        "Ticker"        : ticker,
        "Source"        : "nse_announcement",
        "Event_Category": category,
        "Headline"      : subject[:1000] if subject else None,
        "Raw_Text"      : description[:5000] if description else None,
        "Rule_Score"    : score,
        "FinBERT_Score" : None,
        "FinBERT_Label" : None,
        "URL"           : url[:500] if url else None,
    }


def ingest_nse_announcements(start: date, end: date):
    """
    Pull NSE announcements in 50-day windows between start and end.
    Writes to nifty500_sentiment_raw via INSERT IGNORE.
    """
    window_days = SENTIMENT.get("backfill_window_days", 50)
    print(f"\n📡 NSE Announcements: {start} → {end} "
          f"(window={window_days}d)")
    refresh_nse_cookies()

    total_fetched  = 0
    total_matched  = 0
    total_scored   = 0
    total_written  = 0
    request_count  = 0

    current = start
    while current <= end:
        window_end = min(current + timedelta(days=window_days - 1), end)

        if request_count > 0 and request_count % 20 == 0:
            refresh_nse_cookies()

        items = fetch_nse_announcements(current, window_end)
        request_count += 1
        total_fetched += len(items)

        rows = []
        for item in items:
            parsed = parse_announcement(item)
            if parsed is None:
                continue
            total_matched += 1
            if parsed["Rule_Score"] is not None:
                total_scored += 1
            rows.append(parsed)

        if rows:
            df = pd.DataFrame(rows)
            # Deduplicate within batch before DB write
            df = df.drop_duplicates(
                subset=["Ticker", "Date", "Source", "Headline"]
            )
            try:
                save_to_db(df, TABLES["sentiment_raw"], engine)
                total_written += len(df)
            except Exception as e:
                print(f"  ⚠️  DB write failed for window "
                      f"{current}→{window_end}: {e}")

        print(f"  {current} → {window_end}: "
              f"fetched={len(items)} matched={len(rows)}")

        current = window_end + timedelta(days=1)
        time.sleep(1.5)  # rate-limit politeness

    print(f"\n✅ NSE done: fetched={total_fetched} "
          f"matched={total_matched} rule_scored={total_scored} "
          f"written={total_written}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — MONEYCONTROL RSS
# ═══════════════════════════════════════════════════════════════════════

def ingest_moneycontrol_rss():
    """
    Fetch current Moneycontrol RSS feeds, fuzzy-match tickers, write raw rows.
    RSS = today-forward only. Run daily to accumulate coverage.
    """
    print(f"\n📡 Moneycontrol RSS ({len(MONEYCONTROL_RSS_FEEDS)} feeds)")

    total_items   = 0
    total_matched = 0
    total_written = 0
    all_rows      = []

    for feed_name, url in MONEYCONTROL_RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"  ⚠️  {feed_name} feed failed: {e}")
            continue

        items = feed.entries or []
        total_items += len(items)

        for entry in items:
            headline = (entry.get("title") or "").strip()
            summary  = (entry.get("summary") or
                        entry.get("description") or "").strip()
            link     = (entry.get("link") or "").strip()

            # Parse published date; fall back to today
            pub_date = None
            for attr in ("published_parsed", "updated_parsed"):
                t = entry.get(attr)
                if t:
                    try:
                        pub_date = date(t.tm_year, t.tm_mon, t.tm_mday)
                        break
                    except Exception:
                        continue
            if pub_date is None:
                pub_date = date.today()

            text_blob = f"{headline} {summary}"
            matched_tickers = match_tickers_in_text(text_blob)

            if not matched_tickers:
                continue

            total_matched += 1
            for ticker in matched_tickers:
                all_rows.append({
                    "Date"          : pub_date,
                    "Ticker"        : ticker,
                    "Source"        : "moneycontrol",
                    "Event_Category": None,
                    "Headline"      : headline[:1000] if headline else None,
                    "Raw_Text"      : summary[:5000] if summary else None,
                    "Rule_Score"    : None,
                    "FinBERT_Score" : None,
                    "FinBERT_Label" : None,
                    "URL"           : link[:500] if link else None,
                })

        print(f"  {feed_name}: {len(items)} items")
        time.sleep(1)

    if all_rows:
        df = pd.DataFrame(all_rows).drop_duplicates(
            subset=["Ticker", "Date", "Source", "Headline"]
        )
        try:
            save_to_db(df, TABLES["sentiment_raw"], engine)
            total_written = len(df)
        except Exception as e:
            print(f"  ⚠️  DB write failed: {e}")

    print(f"\n✅ Moneycontrol done: items={total_items} "
          f"matched_articles={total_matched} "
          f"rows_written={total_written}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — AGGREGATION (rebuild nifty500_sentiment from _raw)
# ═══════════════════════════════════════════════════════════════════════

def aggregate_daily_sentiment():
    """
    Rebuild nifty500_sentiment from nifty500_sentiment_raw.
    Per (Ticker, Date):
      - Announcement_Score    = mean(Rule_Score) across NSE events
      - News_Sentiment_Score  = mean(FinBERT_Score) across RSS (if present)
      - Sentiment_Score       = weighted combination (see below)
      - Positive/Negative/Neutral = FinBERT label distribution (RSS only)
      - Counts + flags
    Uses TRUNCATE + INSERT on the AGGREGATE table only. Raw table is NEVER
    touched — this is safe per the "never truncate sentiment" rule, which
    applies to the raw layer (historical source of truth).
    """
    print("\n📊 Aggregating daily sentiment from nifty500_sentiment_raw...")

    raw_table = TABLES["sentiment_raw"]
    agg_table = TABLES["sentiment"]

    with engine.connect() as conn:
        count = conn.execute(text(
            f"SELECT COUNT(*) FROM {raw_table}"
        )).scalar()
        print(f"  Raw rows: {count:,}")

        if count == 0:
            print("  No raw rows to aggregate. Skipping.")
            return

    query = f"""
        SELECT Ticker, Date, Source, Rule_Score,
               FinBERT_Score, FinBERT_Label
        FROM {raw_table}
    """
    df = pd.read_sql(query, engine)

    # --- NSE announcement aggregate ---
    nse = df[df["Source"] == "nse_announcement"].copy()
    nse_agg = (nse.groupby(["Ticker", "Date"])
                  .agg(Announcement_Score=("Rule_Score", "mean"),
                       Events_Count=("Rule_Score", "count"))
                  .reset_index())

    # --- RSS / FinBERT aggregate ---
    rss = df[df["Source"] == "moneycontrol"].copy()
    rss_agg = (rss.groupby(["Ticker", "Date"])
                  .agg(News_Sentiment_Score=("FinBERT_Score", "mean"),
                       Headlines_Count=("Source", "count"))
                  .reset_index())

    # FinBERT label distribution (only where scored)
    scored = rss[rss["FinBERT_Label"].notna()].copy()
    if not scored.empty:
        label_counts = (scored.groupby(["Ticker", "Date", "FinBERT_Label"])
                              .size()
                              .unstack(fill_value=0)
                              .reset_index())
        for col in ("positive", "negative", "neutral"):
            if col not in label_counts.columns:
                label_counts[col] = 0
        label_counts["_total"] = (label_counts["positive"] +
                                  label_counts["negative"] +
                                  label_counts["neutral"])
        label_counts["Positive_Score"] = (label_counts["positive"] /
                                          label_counts["_total"].replace(0, 1))
        label_counts["Negative_Score"] = (label_counts["negative"] /
                                          label_counts["_total"].replace(0, 1))
        label_counts["Neutral_Score"]  = (label_counts["neutral"] /
                                          label_counts["_total"].replace(0, 1))
        label_counts = label_counts[["Ticker", "Date",
                                     "Positive_Score",
                                     "Negative_Score",
                                     "Neutral_Score"]]
    else:
        label_counts = pd.DataFrame(columns=[
            "Ticker", "Date", "Positive_Score", "Negative_Score", "Neutral_Score"
        ])

    # --- Merge NSE + RSS aggregates ---
    agg = pd.merge(nse_agg, rss_agg, on=["Ticker", "Date"], how="outer")
    agg = pd.merge(agg, label_counts, on=["Ticker", "Date"], how="left")

    agg["Events_Count"]    = agg["Events_Count"].fillna(0).astype(int)
    agg["Headlines_Count"] = agg["Headlines_Count"].fillna(0).astype(int)
    agg["Has_Announcement"] = (agg["Events_Count"] > 0).astype(int)
    agg["Has_News"]         = (agg["Headlines_Count"] > 0).astype(int)

    # Unified Sentiment_Score:
    #   Avg of Announcement_Score and News_Sentiment_Score when both present.
    #   Otherwise whichever is available. NULL if neither.
    def combine(row):
        a, n = row["Announcement_Score"], row["News_Sentiment_Score"]
        if pd.notna(a) and pd.notna(n):
            return (a + n) / 2.0
        if pd.notna(a):
            return a
        if pd.notna(n):
            return n
        return None
    agg["Sentiment_Score"] = agg.apply(combine, axis=1)

    agg = agg[[
        "Date", "Ticker",
        "Announcement_Score", "News_Sentiment_Score", "Sentiment_Score",
        "Positive_Score", "Negative_Score", "Neutral_Score",
        "Events_Count", "Headlines_Count",
        "Has_Announcement", "Has_News",
    ]]

    # --- Write: TRUNCATE aggregate then INSERT (raw table untouched) ---
    with engine.connect() as conn:
        conn.execute(text(f"TRUNCATE TABLE {agg_table}"))
        conn.commit()

    save_to_db(agg, agg_table, engine)
    print(f"✅ Aggregated rows written: {len(agg):,}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN / CLI
# ═══════════════════════════════════════════════════════════════════════

def run_backfill():
    start = datetime.strptime(SENTIMENT_START_DATE, "%Y-%m-%d").date()
    end   = date.today()

    print("═" * 65)
    print(f"SENTIMENT BACKFILL — {start} → {end}")
    print("═" * 65)

    ingest_nse_announcements(start, end)
    ingest_moneycontrol_rss()
    aggregate_daily_sentiment()

    print("\n" + "═" * 65)
    print("BACKFILL COMPLETE")
    print("═" * 65)


def run_daily():
    lookback = SENTIMENT.get("daily_lookback_days", 2)
    end   = date.today()
    start = end - timedelta(days=lookback)

    print("═" * 65)
    print(f"SENTIMENT DAILY — last {lookback} days ({start} → {end})")
    print("═" * 65)

    ingest_nse_announcements(start, end)
    ingest_moneycontrol_rss()
    aggregate_daily_sentiment()

    print("\n" + "═" * 65)
    print("DAILY COMPLETE")
    print("═" * 65)


def run_aggregate_only():
    print("═" * 65)
    print("SENTIMENT AGGREGATE — rebuild daily table from raw")
    print("═" * 65)
    aggregate_daily_sentiment()
    print("═" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="hedge_v2 sentiment ingestion (Layer 3B)"
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "daily", "aggregate"],
        required=True,
        help="backfill: full historical pull; "
             "daily: last N days; "
             "aggregate: rebuild daily table from raw",
    )
    args = parser.parse_args()

    if not TICKERS:
        print("❌ TICKERS empty. Check nifty500_tickers.csv path in config.py.")
        sys.exit(1)

    if args.mode == "backfill":
        run_backfill()
    elif args.mode == "daily":
        run_daily()
    elif args.mode == "aggregate":
        run_aggregate_only()