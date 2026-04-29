# config.py
# ═══════════════════════════════════════════════════════════

import os
import pandas as pd
from datetime import date

# ── Project Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the Nifty 500 CSV downloaded from NSE
# Re-download every quarter when SEBI rebalances and replace this file
NIFTY500_CSV = os.path.join(PROJECT_ROOT, "files", "nifty500_tickers.csv")

# Path to Nifty 500 sector classification CSV
# Columns: Company Name, Industry, Symbol, Series, ISIN Code
NIFTY500_SECTORS_CSV = os.path.join(PROJECT_ROOT, "files", "nifty500_sectors.csv")

# Path to RBI repo rate history CSV
# Columns: effective_date, repo_rate
# Update this file manually after each MPC meeting
RBI_REPO_CSV = os.path.join(PROJECT_ROOT, "files", "rbi", "repo_history.csv")

#path to the FO_LIST CSV file
FO_LIST_CSV = os.path.join(PROJECT_ROOT, "files", "fo_list.csv")


# ── Load Tickers from CSV ─────────────────────────────────────────────
def load_tickers(csv_path: str) -> list:
    """
    Load Nifty 500 tickers from NSE watchlist CSV.
    Strips whitespace, removes the index row, appends .NS suffix.
    """
    df = pd.read_csv(csv_path)
    symbol_col = [c for c in df.columns if 'SYMBOL' in c][0]
    symbols = df[symbol_col].str.strip().tolist()
    symbols = [s for s in symbols if s and s != 'NIFTY 500']
    return [f"{s}.NS" for s in symbols]


try:
    TICKERS = load_tickers(NIFTY500_CSV)
except FileNotFoundError:
    TICKERS = []
    print(f"WARNING: {NIFTY500_CSV} not found. TICKERS is empty.")


# ── Load Sector Map from CSV ──────────────────────────────────────────
# Replaces the hardcoded SECTOR_MAP_FALLBACK dict.
# Source: files/nifty500_sectors.csv (NSE official sector classification)
# Covers all 500 tickers — no "Others" fallback needed for known universe.
def load_sector_map(csv_path: str) -> dict:
    """
    Loads ticker → sector mapping from nifty500_sectors.csv.
    Returns dict: { 'RELIANCE.NS': 'Energy', 'HDFCBANK.NS': 'Financial Services', ... }
    Falls back to 'Others' for any ticker not found in the file.
    """
    try:
        df = pd.read_csv(csv_path)
        sector_map = {}
        for _, row in df.iterrows():
            symbol = str(row.get("Symbol", "")).strip()
            industry = str(row.get("Industry", "Others")).strip()
            if symbol:
                sector_map[f"{symbol}.NS"] = industry
        return sector_map
    except FileNotFoundError:
        print(f"WARNING: {csv_path} not found. SECTOR_MAP is empty.")
        return {}


try:
    SECTOR_MAP = load_sector_map(NIFTY500_SECTORS_CSV)
except Exception:
    SECTOR_MAP = {}


def get_sector(ticker: str) -> str:
    """Returns sector for a ticker. Defaults to 'Others' if not found."""
    return SECTOR_MAP.get(ticker, "Others")


# ── NSE Holidays ──────────────────────────────────────────────────────
# NSE is closed on these dates. Updated annually.
# bhavcopy_ingestion.py and any script that iterates trading days uses this.
# Source: NSE official holiday calendar (https://www.nseindia.com)
NSE_HOLIDAYS = {
    # 2026
    date(2026, 1, 26), date(2026, 3, 25), date(2026, 4, 2),  date(2026, 4, 10),
    date(2026, 4, 14), date(2026, 5, 1),  date(2026, 6, 19), date(2026, 8, 15),
    date(2026, 10, 2), date(2026, 10, 20),date(2026, 11, 4), date(2026, 12, 25),
    # 2025
    date(2025, 1, 26), date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31),
    date(2025, 4, 10), date(2025, 4, 14), date(2025, 4, 18), date(2025, 5, 1),
    date(2025, 8, 15), date(2025, 8, 27), date(2025, 10, 2), date(2025, 10, 21),
    date(2025, 10, 24),date(2025, 11, 5), date(2025, 12, 25),
    # 2024
    date(2024, 1, 22), date(2024, 1, 26), date(2024, 3, 25), date(2024, 4, 9),
    date(2024, 4, 11), date(2024, 4, 14), date(2024, 4, 17), date(2024, 5, 1),
    date(2024, 5, 23), date(2024, 6, 17), date(2024, 7, 17), date(2024, 8, 15),
    date(2024, 10, 2), date(2024, 11, 1), date(2024, 11, 15),date(2024, 12, 25),
    # 2023
    date(2023, 1, 26), date(2023, 3, 7),  date(2023, 3, 30), date(2023, 4, 4),
    date(2023, 4, 7),  date(2023, 4, 14), date(2023, 5, 1),  date(2023, 6, 29),
    date(2023, 8, 15), date(2023, 9, 19), date(2023, 10, 2), date(2023, 10, 24),
    date(2023, 11, 14),date(2023, 11, 27),date(2023, 12, 25),
    # 2022
    date(2022, 1, 26), date(2022, 3, 1),  date(2022, 3, 18), date(2022, 4, 14),
    date(2022, 4, 15), date(2022, 5, 3),  date(2022, 8, 9),  date(2022, 8, 15),
    date(2022, 10, 2), date(2022, 10, 5), date(2022, 10, 24),date(2022, 10, 26),
    date(2022, 11, 8), date(2022, 12, 26),
    # 2021
    date(2021, 1, 26), date(2021, 3, 11), date(2021, 3, 29), date(2021, 4, 2),
    date(2021, 4, 14), date(2021, 4, 21), date(2021, 5, 13), date(2021, 7, 21),
    date(2021, 8, 19), date(2021, 9, 10), date(2021, 10, 15),date(2021, 11, 4),
    date(2021, 11, 5), date(2021, 11, 19),date(2021, 12, 25),
    # 2020
    date(2020, 2, 21), date(2020, 3, 10), date(2020, 4, 2),  date(2020, 4, 6),
    date(2020, 4, 10), date(2020, 4, 14), date(2020, 5, 1),  date(2020, 5, 25),
    date(2020, 8, 3),  date(2020, 8, 11), date(2020, 8, 31), date(2020, 9, 2),
    date(2020, 10, 2), date(2020, 11, 16),date(2020, 11, 30),date(2020, 12, 25),
    # 2019
    date(2019, 3, 4),  date(2019, 3, 21), date(2019, 4, 17), date(2019, 4, 19),
    date(2019, 4, 29), date(2019, 5, 18), date(2019, 6, 5),  date(2019, 8, 12),
    date(2019, 8, 15), date(2019, 9, 2),  date(2019, 10, 2), date(2019, 10, 8),
    date(2019, 10, 28),date(2019, 11, 12),date(2019, 12, 25),
    # 2018
    date(2018, 1, 26), date(2018, 2, 13), date(2018, 3, 2),  date(2018, 3, 29),
    date(2018, 3, 30), date(2018, 5, 1),  date(2018, 8, 15), date(2018, 8, 22),
    date(2018, 9, 13), date(2018, 9, 20), date(2018, 10, 2), date(2018, 11, 7),
    date(2018, 11, 8), date(2018, 11, 23),date(2018, 12, 25),
    # 2017
    date(2017, 1, 26), date(2017, 2, 24), date(2017, 3, 13), date(2017, 4, 4),
    date(2017, 4, 14), date(2017, 5, 1),  date(2017, 6, 26), date(2017, 8, 15),
    date(2017, 8, 25), date(2017, 10, 2), date(2017, 10, 19),date(2017, 10, 20),
    date(2017, 12, 25),
    # 2016
    date(2016, 1, 26), date(2016, 3, 7),  date(2016, 3, 24), date(2016, 3, 25),
    date(2016, 4, 14), date(2016, 4, 19), date(2016, 7, 6),  date(2016, 8, 15),
    date(2016, 9, 5),  date(2016, 10, 11),date(2016, 10, 12),date(2016, 11, 14),
    date(2016, 12, 26),
    # 2015
    date(2015, 1, 26), date(2015, 2, 17), date(2015, 3, 6),  date(2015, 4, 2),
    date(2015, 4, 3),  date(2015, 4, 14), date(2015, 5, 1),  date(2015, 7, 17),
    date(2015, 8, 28), date(2015, 9, 17), date(2015, 10, 2), date(2015, 10, 22),
    date(2015, 11, 12),date(2015, 12, 25),
    # 2014
    date(2014, 1, 14), date(2014, 1, 26), date(2014, 2, 27), date(2014, 3, 17),
    date(2014, 4, 8),  date(2014, 4, 14), date(2014, 4, 15), date(2014, 4, 18),
    date(2014, 5, 1),  date(2014, 7, 29), date(2014, 8, 11), date(2014, 8, 15),
    date(2014, 10, 2), date(2014, 10, 3), date(2014, 10, 23),date(2014, 10, 24),
    date(2014, 11, 4), date(2014, 12, 25),
    # 2013
    date(2013, 1, 26), date(2013, 3, 27), date(2013, 3, 29), date(2013, 4, 11),
    date(2013, 4, 19), date(2013, 4, 24), date(2013, 5, 9),  date(2013, 6, 17),
    date(2013, 8, 9),  date(2013, 8, 15), date(2013, 9, 9),  date(2013, 10, 2),
    date(2013, 10, 14),date(2013, 10, 15),date(2013, 11, 5), date(2013, 12, 25),
    # 2012
    date(2012, 1, 26), date(2012, 2, 20), date(2012, 3, 8),  date(2012, 3, 23),
    date(2012, 4, 5),  date(2012, 4, 6),  date(2012, 4, 14), date(2012, 5, 1),
    date(2012, 8, 15), date(2012, 9, 19), date(2012, 10, 2), date(2012, 11, 2),
    date(2012, 11, 14),date(2012, 11, 28),date(2012, 12, 25),
    # 2011
    date(2011, 1, 26), date(2011, 3, 2),  date(2011, 4, 4),  date(2011, 4, 12),
    date(2011, 4, 14), date(2011, 4, 22), date(2011, 5, 13), date(2011, 8, 15),
    date(2011, 8, 31), date(2011, 9, 1),  date(2011, 10, 6), date(2011, 10, 26),
    date(2011, 10, 27),date(2011, 11, 10),date(2011, 12, 26),
    # 2010
    date(2010, 1, 26), date(2010, 3, 1),  date(2010, 3, 16), date(2010, 4, 1),
    date(2010, 4, 2),  date(2010, 4, 14), date(2010, 5, 13), date(2010, 9, 10),
    date(2010, 9, 30), date(2010, 11, 5), date(2010, 11, 17),date(2010, 11, 22),
    date(2010, 12, 17),
}


# ── Banking / NBFC Tickers ────────────────────────────────────────────
# Used by screener_fundamentals.py to apply bank-specific P&L row names.
# Only pure deposit-taking banks and pure lending NBFCs.
# Insurance, fintech, holding companies use standard Sales+ layout.
BANKING_TICKERS = {
    # Scheduled commercial banks
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",       "KOTAKBANK.NS",
    "AXISBANK.NS", "BANDHANBNK.NS","FEDERALBNK.NS", "IDFCFIRSTB.NS",
    "PNB.NS",      "BANKBARODA.NS","CANBK.NS",       "UNIONBANK.NS",
    "YESBANK.NS",  "INDUSINDBK.NS","RBLBANK.NS",     "DCBBANK.NS",
    "KTKBANK.NS",  "KARURVYSYA.NS","CUB.NS",         "AUBANK.NS",
    "EQUITASBNK.NS","UJJIVANSFB.NS","ESAFSFB.NS",
    # Pure lending NBFCs
    "BAJFINANCE.NS","MUTHOOTFIN.NS","CHOLAFIN.NS",   "M&MFIN.NS",
    "MANAPPURAM.NS","RECLTD.NS",    "PFC.NS",         "IRFC.NS",
    "SHRIRAMFIN.NS","LICHSGFIN.NS", "ABCAPITAL.NS",  "AAVAS.NS",
    "HOMEFIRST.NS", "CANFINHOME.NS","APTUS.NS",       "CREDITACC.NS",
}


# ── Data Range ────────────────────────────────────────────────────────
DATA_START = "2010-01-01"
DATA_END   = None   # None = today

# ── Batch Settings ────────────────────────────────────────────────────
BATCH_SIZE  = 10
BATCH_DELAY = 2
STOCK_DELAY = 1

# ── Database ──────────────────────────────────────────────────────────
DB_NAME = "hedge_v2_db"

# ── Table Names ───────────────────────────────────────────────────────
# NOTE: "sector_median" key renamed from "sector_fundamentals_med" (Apr 2026)
#       for consistency. features.py uses TABLES["sector_median"].
TABLES = {
    "ohlcv"                    : "nifty500_ohlcv",
    "indicators"               : "nifty500_indicators",
    "fundamentals"             : "nifty500_fundamentals",
    "macro"                    : "macro_indicators",
    "sentiment"                : "nifty500_sentiment",
    "sentiment_raw"            : "nifty500_sentiment_raw",
    "features"                 : "features_master",
    "data_quality"             : "stock_data_quality",
    "portfolio_positions"      : "portfolio_positions",
    "sector_median"            : "sector_fundamentals_median",
    "corporate_actions"        : "corporate_actions",
    "market_regimes"           : "market_regimes",
}

# ── Model Version ─────────────────────────────────────────────────────
MODEL_VERSION = "v2_20260418"
MODEL_DIR     = os.path.join(PROJECT_ROOT, "models")

# ── Bhavcopy Config ───────────────────────────────────────────────────
BHAVCOPY_TEMP_DIR = os.path.join(PROJECT_ROOT, "data", "bhavcopy_temp")
DATA_SOURCE       = "bhavcopy"

# ── Data Quality Tier Thresholds ──────────────────────────────────────
# Max observed coverage_ratio across 500 bhavcopy-ingested stocks = 0.9454.
# Tier A threshold set to 0.93 (not 0.95) to account for NSE holiday gaps.
DATA_QUALITY = {
    "tier_a": {"min_years": 8,  "min_coverage": 0.93, "min_adv_cr": 5},
    "tier_b": {"min_years": 4,  "min_coverage": 0.75, "min_adv_cr": 1},
    "tier_x_min_days": 252,
}

# ── Tier X Exclusion List ─────────────────────────────────────────────
# Recent IPOs with < 252 trading days as of Apr 2026.
# data_quality.py maintains this going forward.
# NOTE: features.py does NOT use this list — it queries stock_data_quality
# table directly (single source of truth). This list remains for
# screener_fundamentals.py which runs before data_quality.py.
TIER_X_EXCLUDED = [
    "LTM.NS",        "ICICIAMC.NS",  "CANHLIFE.NS",  "JAINREC.NS",
    "LGEINDIA.NS",   "TATACAP.NS",   "JSWCEMENT.NS", "MEESHO.NS",
    "PIRAMALFIN.NS", "SWANCORP.NS",  "CEMPRO.NS",    "URBANCO.NS",
    "PINELABS.NS",   "CPPLUS.NS",    "TMCV.NS",      "ANTHEM.NS",
    "ENRIN.NS",      "TENNIND.NS",   "EMMVEE.NS",    "PWL.NS",
    "TRAVELFOOD.NS", "TMPV.NS",      "ABLBL.NS",     "GROWW.NS",
    "HDBFS.NS",      "LENSKART.NS",  "ETERNAL.NS",   "ACUTAAS.NS",
    "AEGISVOPAK.NS", "THELEELA.NS",  "BELRISE.NS",   "COHANCE.NS",
    "ATHERENERG.NS",
]

# ── Portfolio Constraints ─────────────────────────────────────────────
PORTFOLIO = {
    "min_stocks"              : 30,
    "max_stocks"              : 55,
    "max_stocks_hard_limit"   : 60,
    "max_position_pct"        : 0.05,
    "min_position_pct"        : 0.001,
    "max_sector_long_pct"     : 0.25,
    "max_sector_short_pct"    : 0.15,
    "max_long_exposure"       : 1.20,
    "target_long_exposure"    : 1.15,
    "max_short_exposure"      : 0.20,
    "target_short_exposure"   : 0.15,
    "nifty100_anchor_pct"     : 0.65,
    "alpha_midcap_pct"        : 0.35,
    "max_midcap_exposure"     : 0.35,
    "min_cash_pct"            : 0.08,
    "max_cash_pct"            : 0.12,
    "monthly_drawdown_trigger": 0.05,
    "peak_trough_hard_stop"   : 0.10,
    "long_stop_loss"          : 0.15,
    "short_stop_loss"         : 0.10,
    "brokerage_bps"           : 3,
    "stamp_duty_bps"          : 1.5,
    "stt_bps"                 : 10,
    "slippage_bps"            : 5,
}

DEPLOY_PCT = {
    "Bull"     : 0.92,
    "Sideways" : 0.90,
    "Bear"     : 0.80,
    "Elevated" : 0.75,
}

# ── Macro Data Config ─────────────────────────────────────────────────
MACRO_YFINANCE = {
    "India_VIX"    : "^INDIAVIX",
    "USDINR"       : "INR=X",
    "Crude_Oil"    : "CL=F",
    "Gold"         : "GC=F",
    "Nifty50_Close"  : "^NSEI",
    "Nifty500_Close" : "^CRSLDX",
}

MACRO_FRED = {
    "GDP_India"      : "NGDPRNSAXDCINQ",
    "CPI_India"      : "INDCPIALLMINMEI",
    "Fed_Funds_Rate" : "FEDFUNDS",
    "US_CPI"         : "CPIAUCSL",
    "US_10Y_Bond"    : "DGS10",
}

MACRO_RBI = {
    "Repo_Rate"      : "rbi_repo_rate",
    "IIP_Growth"     : "rbi_iip",
    "Forex_Reserves" : "rbi_forex_reserves",
}

# ── Fundamentals Source ───────────────────────────────────────────────
FUNDAMENTALS_SOURCE = "screener"

 # ── Sentiment Config ──────────────────────────────────────────────────
    # Data source strategy (see project_context_architecture.txt Layer 3B):
    #   1. NSE Corporate Announcements (structured, 2022-present, rule-scored)
    #      Full historical depth since SENTIMENT_START_DATE.
    #   2. Moneycontrol RSS (unstructured, today-forward only)
    #      RSS feeds are current snapshot; no historical backfill possible.
    # FinBERT scoring runs in separate Colab notebook (reads Raw_Text,
    # writes FinBERT_Score/FinBERT_Label back to nifty500_sentiment_raw).

SENTIMENT_START_DATE = "2022-01-01"
 
SENTIMENT = {
        "sources"                 : ["nse_announcement", "moneycontrol"],
        "model"                   : "ProsusAI/finbert",
        "daily_lookback_days"     : 2,
        "backfill_window_days"    : 50,
        "fuzzy_match_threshold"   : 85,
    }
 
    # NSE corporate announcements API (public, no auth, rate-limited)
NSE_ANNOUNCEMENT_URL = (
        "https://www.nseindia.com/api/corporate-announcements"
        "?index=equities&from_date={from_date}&to_date={to_date}"
    )
 
    # Moneycontrol RSS feeds — current snapshot only (~50-100 items per feed)
MONEYCONTROL_RSS_FEEDS = {
        "business"  : "https://www.moneycontrol.com/rss/business.xml",
        "markets"   : "https://www.moneycontrol.com/rss/marketsnews.xml",
        "economy"   : "https://www.moneycontrol.com/rss/economy.xml",
        "results"   : "https://www.moneycontrol.com/rss/results.xml",
        "latest"    : "https://www.moneycontrol.com/rss/latestnews.xml",
    }
 
    # Rule-based scoring for NSE announcements.
    # Matches subject/description against keyword lists (case-insensitive).
    # First matching category wins. Score range [-1.0, +1.0].
NSE_ANNOUNCEMENT_RULES = {
        "earnings_result": {
            "keywords": ["financial result", "quarterly result", "audited result",
                         "unaudited result", "board meeting outcome"],
            "score": 0.2,
        },
        "buyback": {
            "keywords": ["buy-back", "buyback", "buy back of equity"],
            "score": 0.6,
        },
        "dividend": {
            "keywords": ["dividend declared", "interim dividend", "final dividend",
                         "dividend declaration"],
            "score": 0.3,
        },
        "pledge": {
            "keywords": ["pledge", "encumbrance", "invocation of pledge"],
            "score": -1.0,
        },
        "credit_rating": {
            "keywords": ["credit rating", "rating action", "rating revision",
                         "rating upgrade", "rating downgrade"],
            "score": 0.0,
        },
        "order_win": {
            "keywords": ["order win", "new order", "contract awarded",
                         "letter of intent", "loi received", "work order"],
            "score": 0.5,
        },
        "management_change": {
            "keywords": ["resignation", "appointment", "cessation",
                         "managing director", "chief executive", "cfo"],
            "score": 0.0,
        },
        "regulatory": {
            "keywords": ["sebi", "penalty", "show cause", "adjudication",
                         "enforcement", "violation"],
            "score": -0.7,
        },
    }

# ── Liquidity Filter ──────────────────────────────────────────────────
LIQUIDITY_FILTER = {
    "long_book_min_adv_cr"  : 5.0,
    "short_book_min_adv_cr" : 10.0,
    "lookback_days"         : 90,
}


# ── Features Pipeline Config ──────────────────────────────────────────
# Tunable parameters for data/features.py. All values locked via
# Apr 2026 architecture discussion — change here to re-tune without
# editing features.py directly.
FEATURES = {
    # Fundamental availability lag:
    # Annual reports become "public" this many calendar days after fiscal
    # year-end. Controls look-ahead prevention. 60 days = standard Indian
    # reporting lag (results typically released within 45-60 days of FY-end).
    "fundamentals_availability_lag_days": 60,

    # Sector median fallback — minimum tickers with actual data required
    # to compute a reliable sector median for a given (Sector, Period).
    # If fewer than this many tickers have actual data → NULL (no fallback).
    # Rationale: median of <3 observations is statistically unstable.
    "sector_median_min_tickers": 3,

    # Maximum days to forward-fill annual fundamental data into daily rows.
    # Beyond this, data is considered stale → trigger sector-median fallback.
    # 400 days = annual reporting cycle + 35-day buffer for late filings.
    "fundamentals_max_forward_fill_days": 400,

    # Price gap detection threshold (calendar days between consecutive OHLCV
    # rows for a ticker). Rows where prior gap exceeds this → Price_Gap_Flag=1.
    # NSE max legitimate gap = long weekend + holidays = ~5 days.
    "price_gap_flag_threshold_days": 5,

    # Target forecast horizon (trading days, not calendar days).
    # All target variables use this window: Target_Return_21d, Target_Rank_21d.
    # Must match the horizon used in Colab ML training (v2 Section 5.1).
    "target_horizon_days": 21,

    # LSTM Head 2 realized volatility target window (trading days).
    # Target_Vol_5d = std(log_returns[t+1..t+N]) × sqrt(252), ddof=1.
    "target_vol_window_days": 5,

    # Volatility annualization factor (trading days per year).
    # Indian equity markets: ~250 trading days/year.
    "trading_days_per_year": 252,


    "export_dir": "exports",
}