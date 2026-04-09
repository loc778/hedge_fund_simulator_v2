# config.py
# ═══════════════════════════════════════════════════════════
# CENTRAL CONFIGURATION — hedge_v2 (scaled production version)
# Change values here to scale up/modify behaviour
# ═══════════════════════════════════════════════════════════

import os
import pandas as pd

# ── Project Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the Nifty 500 CSV downloaded from NSE
# Re-download every quarter when SEBI rebalances and replace this file
NIFTY500_CSV = os.path.join(PROJECT_ROOT, "nifty500_tickers.csv")

# ── Load Tickers from CSV ─────────────────────────────────────────────
# Reads the NSE watchlist CSV and appends .NS suffix for yfinance
# Column name in NSE CSV is 'SYMBOL \n' (with newline — NSE quirk)
def load_tickers(csv_path: str) -> list:
    """
    Load Nifty 500 tickers from NSE watchlist CSV.
    Strips whitespace, removes the index row, appends .NS suffix.
    Re-run any time the CSV is replaced with a new quarterly list.
    """
    df = pd.read_csv(csv_path)
    # Column has trailing newline in name — strip it
    symbol_col = [c for c in df.columns if 'SYMBOL' in c][0]
    symbols = df[symbol_col].str.strip().tolist()
    # Remove the index name row (first row is 'NIFTY 500')
    symbols = [s for s in symbols if s and s != 'NIFTY 500']
    # Append .NS for yfinance / NSE identification
    return [f"{s}.NS" for s in symbols]

TICKERS = load_tickers(NIFTY500_CSV)

# ── Data Range ────────────────────────────────────────────────────────
# Bhavcopy ingestion uses explicit start/end dates (not yfinance period string)
DATA_START = "2010-01-01"
DATA_END   = None   # None = today (always fetch up to latest available date)

# ── Batch Settings ────────────────────────────────────────────────────
# Used for yfinance calls (adjustment factors, fallback fetches)
BATCH_SIZE  = 10
BATCH_DELAY = 2
STOCK_DELAY = 1

# ── Database ──────────────────────────────────────────────────────────
DB_NAME = "hedge_v2_db"

# ── Table Names ───────────────────────────────────────────────────────
TABLES = {
    "ohlcv"                    : "nifty500_ohlcv",
    "indicators"               : "nifty500_indicators",
    "fundamentals"             : "nifty500_fundamentals",
    "macro"                    : "macro_indicators",
    "sentiment"                : "nifty500_sentiment",
    "features"                 : "features_master",
    "data_quality"             : "stock_data_quality",
    "portfolio_positions"      : "portfolio_positions",
    "sector_fundamentals_med"  : "sector_fundamentals_median",
    "corporate_actions"        : "corporate_actions",
}

# ── Model Version ─────────────────────────────────────────────────────
# Update this after every retraining session
# Format: YYYYMMDD matching the date suffix on saved model files
MODEL_VERSION = "20260409"

# ── Model Directory ───────────────────────────────────────────────────
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ── Bhavcopy Config ───────────────────────────────────────────────────
# NSE archive URL pattern — {year} {month} {date} filled in by ingestion script
BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/historical/EQUITIES"
    "/{year}/{month}/cm{date}bhav.csv.zip"
)

# Local folder to store downloaded bhavcopy ZIPs temporarily
# ZIPs are deleted immediately after parsing to save disk space
BHAVCOPY_TEMP_DIR = os.path.join(PROJECT_ROOT, "data", "bhavcopy_temp")

# ── Data Source Flag ──────────────────────────────────────────────────
DATA_SOURCE = "bhavcopy"   # options: "bhavcopy", "yfinance"

# ── Sector Map ────────────────────────────────────────────────────────
# Auto-fetched from yfinance during ingestion — stored in sector_map.pkl
# This manual map is the fallback for any ticker yfinance can't classify
# Expand this as you add more tickers
SECTOR_MAP_FALLBACK = {
    "IT":            ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
                      "LTIM.NS", "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS"],
    "Banking":       ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS",
                      "AXISBANK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS",
                      "IDFCFIRSTB.NS", "PNB.NS", "BANKBARODA.NS",
                      "CANBK.NS", "UNIONBANK.NS"],
    "Finance":       ["BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS",
                      "SBILIFE.NS", "ICICIPRULI.NS", "MUTHOOTFIN.NS",
                      "CHOLAFIN.NS", "M&MFIN.NS", "MANAPPURAM.NS",
                      "RECLTD.NS", "PFC.NS", "IRFC.NS", "PAYTM.NS",
                      "POLICYBZR.NS"],
    "Auto":          ["MARUTI.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
                      "HEROMOTOCO.NS", "M&M.NS", "TATAMOTORS.NS"],
    "Pharma":        ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
                      "DIVISLAB.NS", "AUROPHARMA.NS", "GLAND.NS"],
    "Energy":        ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS",
                      "GAIL.NS", "COALINDIA.NS", "POWERGRID.NS", "NTPC.NS"],
    "Metals":        ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS",
                      "VEDL.NS", "NMDC.NS", "SAIL.NS", "JINDALSTEL.NS"],
    "FMCG":          ["HINDUNILVR.NS", "ITC.NS", "BRITANNIA.NS", "DABUR.NS",
                      "GODREJCP.NS", "MARICO.NS", "NESTLEIND.NS"],
    "Infra":         ["LT.NS", "ADANIPORTS.NS", "ADANIENT.NS", "HAL.NS",
                      "SIEMENS.NS", "INDUSTOWER.NS"],
    "RealEstate":    ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS",
                      "PRESTIGE.NS", "PHOENIXLTD.NS"],
    "Consumer":      ["ASIANPAINT.NS", "TITAN.NS", "PIDILITIND.NS",
                      "BERGEPAINT.NS", "HAVELLS.NS", "VOLTAS.NS",
                      "DMART.NS", "TRENT.NS", "INDHOTEL.NS", "IRCTC.NS"],
    "Cement":        ["ULTRACEMCO.NS", "SHREECEM.NS", "AMBUJACEM.NS",
                      "ACC.NS", "RAMCOCEM.NS", "GRASIM.NS"],
    "Telecom":       ["BHARTIARTL.NS", "TATACOMM.NS"],
    "Chemicals":     [],   # populated as tickers are identified
    "Logistics":     [],
    "Defence":       [],
    "Media":         [],
    "Others":        [],   # catch-all for unclassified tickers
}

# ── Portfolio Constraints ─────────────────────────────────────────────
# All constraints from Constraints doc — single source of truth
#
# IMPORTANT — CAP vs TARGET distinction:
#   "max_*"  = hard ceiling — the portfolio engine NEVER exceeds this
#   "target_*" = ideal operating range — the engine AIM for this
#   There are NO forced minimums on long/short exposure.
#   The model is free to hold less short (or zero short) if signals
#   don't justify it. Forcing minimum exposure would mean taking bad
#   trades just to meet an arbitrary floor — that defeats the purpose.
PORTFOLIO = {
    # Stock count limits
    "min_stocks"              : 30,     # minimum positions (regime-dependent floor)
    "max_stocks"              : 55,     # maximum positions
    "max_stocks_hard_limit"   : 60,     # absolute hard cap (only in extreme cases +5)

    # Individual position limits (% of NAV)
    "max_position_pct"        : 0.05,   # HARD CAP: 5% max per stock at cost
    "min_position_pct"        : 0.001,  # 0.1% minimum — below this, skip the position

    # Sector limits
    # max_sector_long_pct  = HARD CAP on gross long per sector (not a target)
    #   e.g. if IT is at 25%, no new IT longs are opened until existing ones reduce
    # max_sector_short_pct = HARD CAP on gross short per sector (not a target)
    #   the model is NOT forced to short any sector at all
    "max_sector_long_pct"     : 0.25,   # HARD CAP: 25% max gross long per sector
    "max_sector_short_pct"    : 0.15,   # HARD CAP: 15% max gross short per sector

    # Long/short book structure — CAPS ONLY, no forced minimums
    # The model targets 110-120% long and 10-20% short when signals exist.
    # If there are no valid short signals, short book can be zero.
    # These are caps to prevent over-leverage, not floors to force exposure.
    "max_long_exposure"       : 1.20,   # HARD CAP: never exceed 120% long
    "target_long_exposure"    : 1.15,   # TARGET: aim for ~115% long when signals exist
    "max_short_exposure"      : 0.20,   # HARD CAP: never exceed 20% short
    "target_short_exposure"   : 0.15,   # TARGET: aim for ~15% short when signals exist
    # Note: short book can be 0% if no valid short signals exist

    # Long book composition (targets, not hard rules)
    "nifty100_anchor_pct"     : 0.65,   # TARGET: 60-65% of long book in Nifty 100
    "alpha_midcap_pct"        : 0.35,   # TARGET: 35-40% in Nifty 100-500
    "max_midcap_exposure"     : 0.35,   # HARD CAP: 35% max mid-cap total

    # Cash buffer — both min and max are hard rules
    # Min cash protects against forced selling during RBI surprises / FII sell-offs
    "min_cash_pct"            : 0.08,   # HARD FLOOR: always keep 8% cash minimum
    "max_cash_pct"            : 0.12,   # SOFT CEILING: above 12% = deploy or explain why

    # Drawdown limits
    "monthly_drawdown_trigger": 0.05,   # -5% in 21 days → mandatory risk review
    "peak_trough_hard_stop"   : 0.10,   # -10% from peak → reduce book 30% within 5 days

    # Stop losses
    "long_stop_loss"          : 0.15,   # -15% from entry → exit long immediately
    "short_stop_loss"         : 0.10,   # +10% against short → cover immediately

    # Transaction cost model (basis points = 0.01%)
    # These are deducted in backtesting and shown separately in the dashboard UI
    "brokerage_bps"           : 3,      # ~3bps per leg (Zerodha/discount broker)
    "stamp_duty_bps"          : 1.5,    # 0.015% on buy side only
    "stt_bps"                 : 10,     # 0.1% STT on sell side (equity delivery)
    "slippage_bps"            : 5,      # estimated market impact / slippage
    # Total round-trip cost = (3+1.5+5) buy + (3+10+5) sell = ~27.5 bps ≈ 0.275%
}

# Regime-aware deployable capital multipliers
# In bear/elevated regimes, keep more cash
DEPLOY_PCT = {
    "Bull"     : 0.92,   # deploy 92% — keep 8% cash minimum
    "Sideways" : 0.90,
    "Bear"     : 0.80,   # deploy 80% — keep 20% cash in bear
    "Elevated" : 0.75,   # highest cash buffer in crisis regime
}

# ── Macro Data Config ─────────────────────────────────────────────────
MACRO_YFINANCE = {
    "India_VIX" : "^INDIAVIX",
    "USDINR"    : "INR=X",
    "Crude_Oil" : "CL=F",
    "Gold"      : "GC=F",
}

MACRO_FRED = {
    # GDP_India: India Real GDP from IMF via FRED
    # NOTE: This is quarterly data. Forward-filled to daily in features.py.
    # The prototype used 999999.9999 as placeholder — that bug is fixed here
    # by using the correct FRED series ID and validating on load.
    "GDP_India"      : "NGDPRNSAXDCINQ",   # India Real GDP (IMF, quarterly, USD)
    "CPI_India"      : "INDCPIALLMINMEI",   # India CPI (monthly)
    "Fed_Funds_Rate" : "FEDFUNDS",          # US Fed Funds Rate (monthly)
    "US_CPI"         : "CPIAUCSL",          # US CPI (monthly)
    "US_10Y_Bond"    : "DGS10",             # US 10-Year Treasury (daily)
}

# FII/DII Daily Flows
# Source: NSE publishes FII/DII daily activity data free on their website
# Script: data/fii_dii.py (separate dedicated script)
# Table: stored in macro_indicators alongside other daily macro data
# NSE endpoint: https://www.nseindia.com/api/fiidiiTradeReact
# Columns added to macro_indicators: FII_Net_Buy_Cr, DII_Net_Buy_Cr
# These are critical Indian market signals — FII sell-off > Rs 5000 Cr
# in a session triggers the risk manager's defensive protocol

# RBI DBIE series — repo rate, IIP, forex reserves
# Script: data/rbi_macro.py (separate dedicated script)
# RBI does not have a clean public API — data is downloaded as CSV
# from RBI DBIE portal and parsed
MACRO_RBI = {
    "Repo_Rate"         : "rbi_repo_rate",      # fetched by rbi_macro.py
    "IIP_Growth"        : "rbi_iip",            # Index of Industrial Production
    "Forex_Reserves"    : "rbi_forex_reserves", # USD billions
}

# ── Fundamentals Data Source ──────────────────────────────────────────
# Source: Screener.in (primary) — 10+ years of clean quarterly data
# Script: data/screener_fundamentals.py
# yfinance is NOT used for fundamentals at scale — it is unreliable
# for Indian stocks (missing quarters, wrong currency, inconsistent columns)
# Screener.in provides structured quarterly P&L, balance sheet, cash flow
# for all NSE-listed stocks going back 10+ years for free
FUNDAMENTALS_SOURCE = "screener"   # options: "screener", "yfinance" (testing only)

# ── Sentiment Config ──────────────────────────────────────────────────
SENTIMENT = {
    "sources"                  : ["gdelt", "newsapi", "et_rss"],
    "max_headlines_per_stock"  : 10,
    "model"                    : "ProsusAI/finbert",
    "hf_api"                   : True,
    "lookback_days"            : 3,
}

ET_RSS_FEEDS = {
    "markets" : "https://economictimes.indiatimes.com/markets/rss.cms",
    "stocks"  : "https://economictimes.indiatimes.com/markets/stocks/rss.cms",
    "economy" : "https://economictimes.indiatimes.com/news/economy/rss.cms",
    "results" : "https://economictimes.indiatimes.com/markets/earnings/rss.cms",
}

# ── Liquidity Filter ──────────────────────────────────────────────────
# Stocks below this average daily value are excluded from portfolio
# Units: INR Crores
LIQUIDITY_FILTER = {
    "long_book_min_adv_cr"  : 5.0,    # min 5 Cr ADV for long positions
    "short_book_min_adv_cr" : 10.0,   # min 10 Cr ADV for short (need to exit fast)
    "lookback_days"         : 90,     # 90-day average
}

# ── Data Quality Tier Thresholds ──────────────────────────────────────
# Used by data/data_quality.py to classify stocks
DATA_QUALITY = {
    "tier_a": {"min_years": 8,  "max_gap_pct": 5,  "min_adv_cr": 5},
    "tier_b": {"min_years": 4,  "max_gap_pct": 25, "min_adv_cr": 1},
    # Tier C = everything else
    # Tier X = fewer than 252 trading days total (excluded entirely)
    "tier_x_min_days": 252,
}