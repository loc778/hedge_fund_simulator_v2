# data/features.py — hedge_v2.3
# ═══════════════════════════════════════════════════════════
# FEATURE ENGINEERING PIPELINE
#
# Combines all raw data layers into features_master — the single
# unified table consumed by all ML training notebooks.
#
# INPUTS (must be populated):
#   nifty500_ohlcv            ~1.28M rows
#   nifty500_indicators       ~1.15M rows (computed on Adj_Close)
#   nifty500_fundamentals     ~4,000 rows (annual, leakage-free)
#   macro_indicators          ~4,246 rows (daily, ffilled at ingest)
#   nifty500_sentiment        ~152,826 rows (daily aggregate)
#   stock_data_quality        500 rows (Tier A/B/C/X classification)
#   market_regimes            ~4,000+ rows (daily regime labels)
#
# OUTPUTS (TRUNCATED and rebuilt on every run):
#   features_master               ~1.2-1.5M rows, 76 columns
#   sector_fundamentals_median    ~200-500 rows (Sector × Period)
#
# EXECUTION:
#   python data/features.py
#
# CHANGES v2.3 (IC fix — Fix A + Fix B):
#
#   Fix A — Cross-sectional momentum rank features (Phase 2 + Phase 3.5):
#     Mom_12_1              : Return_252d - Return_21d (raw, per-ticker)
#     Mom_6_1               : Return_126d - Return_21d (raw, per-ticker)
#     Mom_1M_Rank           : CS percentile rank of Return_21d per date
#     Mom_12_1_Rank         : CS percentile rank of Mom_12_1 per date
#     Mom_6_1_Rank          : CS percentile rank of Mom_6_1 per date
#     RS_21d_Rank           : CS percentile rank of RS_21d per date
#   Root cause addressed: raw returns are not cross-sectional signals.
#   A stock's Return_21d = +8% means nothing in isolation; being in the
#   top 10% of all stocks that month is the actionable signal. Momentum
#   rank consistently contributes IC 0.04–0.07 in Indian equity cross-section.
#
#   Fix B — Within-sector fundamental rank features (Phase 3.5):
#     ROA_Sector_Rank              : within-sector pct rank of ROA per date
#     EBITDA_Margin_Sector_Rank    : within-sector pct rank of EBITDA_Margin
#     Revenue_Growth_Sector_Rank   : within-sector pct rank of Revenue_YoY_Growth
#   Root cause addressed: annual fundamental ratios forward-filled for
#   up to 400 days are effectively static labels. ROA = 0.12 is a different
#   quality signal in Financials vs Pharma vs Manufacturing. Sector rank
#   converts static snapshot into cross-sectionally meaningful quality factor.
#   Minimum 3 tickers in sector with non-NULL value required; else NULL.
#
#   Implementation: Phase 3.5 (new) runs after Phase 3 (rank pass).
#   Both Fix A and Fix B are batch cross-sectional operations — they cannot
#   be computed in the per-ticker Phase 2 loop and must be applied as a
#   post-processing pass over the full features_master table.
#
# ARCHITECTURAL RULES:
#   - Tier X tickers excluded (from stock_data_quality, not hardcoded)
#   - Tier A+B+C included; Tier C rows get NULL ranks (not trained on)
#   - Target_Rank_21d computed cross-sectionally on Tier A+B only
#   - Fundamental availability lag = 60 days after fiscal year-end
#   - Sector median fallback requires ≥3 tickers with actual data
#   - All NULLs flow through — XGBoost handles natively, flags carry info
#   - No z-score normalization here — done in Colab per walk-forward fold
#   - Targets NULL for rows where forward window > max available date
#   - Sentiment joined with 1-day lag (t-1) to prevent same-day leakage
#
# FEATURE DESIGN PRINCIPLES (v2.3 — clean DB, no dead columns):
#   Every column stored in features_master is directly usable for training.
#   No DB-only columns, no raw price levels, no non-stationary series.
#
#   PRICE BLOCK:
#     - Raw OHLCV, Adj_Close, Volume, VWAP_Daily removed — non-stationary.
#     - Returns (1d/5d/21d/60d), log return, vol-20d kept — stationary.
#     - Volume_Ratio_20d kept (relative, stationary). ADV_20d_Cr removed
#       (absolute liquidity size — not an alpha signal).
#     - RS_21d / RS_60d added — relative strength vs Nifty500 benchmark.
#
#   MOMENTUM BLOCK (Fix A — NEW in v2.3):
#     - Mom_12_1 / Mom_6_1: standard price momentum signals (raw).
#     - *_Rank variants: cross-sectional percentile per date — the model
#       consumes both raw and ranked versions for complementary signal.
#
#   TECHNICAL BLOCK:
#     - Raw SMA/EMA levels removed — replaced by Price_to_SMA20/50/200 ratios.
#     - Raw BB_Upper/Middle/Lower removed — replaced by BB_Width + BB_PctB.
#     - Raw OBV removed — replaced by OBV_Change_5d (5d pct change).
#     - MACD + MACD_Signal removed — MACD_Hist is the actionable delta.
#     - Stoch_D removed — smoothed/lagged duplicate of Stoch_K.
#     - Price_to_52W_High / Price_to_52W_Low added.
#
#   FUNDAMENTAL BLOCK:
#     - EPS_Basic removed (non-comparable raw level across tickers).
#     - Added: Gross_Profit_Margin (banking-aware), Gross_Margin_Is_Proxy,
#              Debt_to_Assets, OCF_to_Net_Income, Delta_ROA, Delta_DebtEquity,
#              Rel_ROA, Rel_EBITDA_Margin.
#     - Within-sector rank columns (Fix B — NEW in v2.3):
#              ROA_Sector_Rank, EBITDA_Margin_Sector_Rank,
#              Revenue_Growth_Sector_Rank.
#
#   SENTIMENT BLOCK:
#     - Sentiment_Score (same-day) removed — replaced by Sentiment_Score_Lag1.
#     - News_Sentiment_Score, Positive_Score, Negative_Score removed —
#       highly correlated with Sentiment_Score_Lag1; Announcement_Score
#       kept as it captures discrete event signal.
#
#   MACRO BLOCK:
#     - Raw India_VIX, USDINR, Crude_Oil, Gold removed — cross-sectionally
#       identical for all stocks on a date; betas capture stock-specific
#       sensitivity to each factor far more effectively.
#     - FII_Momentum_5d / DII_Momentum_5d removed — low-cardinality ±1
#       signals; Beta_to_FII and Beta_to_DII are strictly superior.
#     - RepoRate_x_DebtEquity removed — interaction on a level variable
#       (Repo_Rate) has same cross-sectional flatness problem as raw levels.
#     - Repo_Rate_Change added — first-difference captures policy shift events.
#     - Beta_to_DII added — DII often contrarian to FII; independent signal.
#     - Regime_Int added — joined from market_regimes (0-3 state).
#
#   COLUMN COUNT:
#     Previous (v2.2): 67 columns
#     This version  : 76 columns — 9 new (Fix A: 6, Fix B: 3)
# ═══════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from config import TABLES, SENTIMENT_START_DATE, FEATURES, get_sector
from data.db import get_engine, save_to_db

engine = get_engine()

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════

FUND_LAG_DAYS          = FEATURES["fundamentals_availability_lag_days"]         # 60
SECTOR_MED_MIN_TICKERS = FEATURES["sector_median_min_tickers"]                  # 3
FUND_MAX_FFILL_DAYS    = FEATURES["fundamentals_max_forward_fill_days"]         # 400
PRICE_GAP_THRESHOLD    = FEATURES["price_gap_flag_threshold_days"]              # 5
TARGET_HORIZON         = FEATURES["target_horizon_days"]                        # 21
TARGET_VOL_WINDOW      = FEATURES["target_vol_window_days"]                     # 5
TRADING_DAYS_PER_YEAR  = FEATURES["trading_days_per_year"]                      # 252

# Minimum sector size for Fix B sector rank computation.
# Requires at least this many tickers with non-NULL values before
# computing a percentile rank (otherwise rank is meaningless).
SECTOR_RANK_MIN_TICKERS = 3

# Tier string → numeric mapping
TIER_MAP = {"A": 1, "B": 2, "C": 3}

# Sectors where Gross_Profit is structurally NULL (no COGS concept).
# Net_Income is used as fallback numerator for Gross_Profit_Margin,
# and Gross_Margin_Is_Proxy=1 flags these rows for the model.
BANKING_SECTORS = {
    "Banks", "Finance", "Financial Services",
    "Insurance", "NBFC", "Diversified Financials",
}

# Fundamental ratio columns computed from annual source data.
# Delta columns computed after base columns (order matters).
# Gross_Margin_Is_Proxy excluded from sector-median aggregation (it is a flag).
RATIO_COLS = [
    "Revenue_YoY_Growth",    # (Revenue - Revenue_prev) / Revenue_prev
    "EBITDA_Margin",         # EBITDA / Revenue
    "Net_Margin",            # Net_Income / Revenue
    "FCF_Margin",            # Free_Cash_Flow / Revenue
    "Asset_Turnover",        # Revenue / Total_Assets
    "ROA",                   # sourced directly from fundamentals table
    "Debt_to_Equity",        # sourced directly from fundamentals table
    "Gross_Profit_Margin",   # Gross_Profit / Revenue; Net_Income/Revenue for banking
    "Gross_Margin_Is_Proxy", # 1 if banking/NBFC sector fallback used
    "Debt_to_Assets",        # Total_Debt / Total_Assets
    "OCF_to_Net_Income",     # Operating_CF / Net_Income — earnings quality
    "Delta_ROA",             # ROA minus prior-period ROA — profitability direction
    "Delta_DebtEquity",      # Debt_to_Equity minus prior-period — leverage direction
]

# Sector-relative columns computed inside merge_fundamentals after medians exist.
REL_COLS = [
    "Rel_ROA",           # stock ROA - sector median ROA
    "Rel_EBITDA_Margin", # stock EBITDA_Margin - sector median EBITDA_Margin
]

# ── Final features_master column order ────────────────────────────────
# Every column here is directly training-usable.
# Must match setup_db.py DDL exactly (minus auto-increment id).
FEATURES_COLUMNS = [
    "Date", "Ticker",

    # ── Returns & volatility (stationary) ─────────────────────────────
    "Return_1d",
    "Log_Return_1d",
    "Return_5d",
    "Return_21d",
    "Return_60d",
    "Volatility_20d",
    "Volume_Ratio_20d",

    # ── Relative strength vs Nifty500 ────────────────────────────────
    "RS_21d",            # Return_21d - Nifty500_Return_21d
    "RS_60d",            # Return_60d - Nifty500_Return_60d

    # ── Cross-sectional momentum signals (Fix A — new in v2.3) ────────
    "Mom_12_1",          # Return_252d - Return_21d (raw per-ticker)
    "Mom_6_1",           # Return_126d - Return_21d (raw per-ticker)
    "Mom_1M_Rank",       # CS rank of Return_21d per date (Phase 3.5)
    "Mom_12_1_Rank",     # CS rank of Mom_12_1 per date (Phase 3.5)
    "Mom_6_1_Rank",      # CS rank of Mom_6_1 per date (Phase 3.5)
    "RS_21d_Rank",       # CS rank of RS_21d per date (Phase 3.5)

    # ── Technical: stationary price-position ratios ───────────────────
    "Price_to_SMA20",    # Adj_Close / SMA_20  — short-term trend position
    "Price_to_SMA50",    # Adj_Close / SMA_50  — medium-term trend position
    "Price_to_SMA200",   # Adj_Close / SMA_200 — long-term trend position (NULL < 200 bars)
    "Price_to_52W_High", # Adj_Close / 252d rolling max(High)
    "Price_to_52W_Low",  # Adj_Close / 252d rolling min(Low)

    # ── Technical indicators (all stationary) ─────────────────────────
    "MACD_Hist",         # MACD - Signal line — momentum delta
    "RSI_14",            # Relative Strength Index (0–100)
    "BB_Width",          # (BB_Upper - BB_Lower) / BB_Middle — volatility regime
    "BB_PctB",           # (Close - BB_Lower) / (BB_Upper - BB_Lower) — band position
    "ATR_14",            # Average True Range — intraday volatility
    "Stoch_K",           # Stochastic %K — momentum oscillator
    "ADX_14",            # Average Directional Index — trend strength
    "OBV_Change_5d",     # OBV.pct_change(5) — stationary volume trend
    "VWAP_Dev",          # Deviation from VWAP — intraday price vs. volume-weighted avg

    # ── Fundamental ratios ────────────────────────────────────────────
    "Revenue_YoY_Growth",
    "EBITDA_Margin",
    "Net_Margin",
    "FCF_Margin",
    "Asset_Turnover",
    "ROA",
    "Debt_to_Equity",
    "Gross_Profit_Margin",
    "Gross_Margin_Is_Proxy",  # TINYINT flag: 1 = banking sector Net_Income proxy
    "Debt_to_Assets",
    "OCF_to_Net_Income",
    "Delta_ROA",
    "Delta_DebtEquity",

    # ── Sector-relative fundamentals ──────────────────────────────────
    "Rel_ROA",
    "Rel_EBITDA_Margin",

    # ── Within-sector fundamental rank features (Fix B — new in v2.3) ─
    "ROA_Sector_Rank",            # within-sector percentile rank of ROA
    "EBITDA_Margin_Sector_Rank",  # within-sector percentile rank of EBITDA_Margin
    "Revenue_Growth_Sector_Rank", # within-sector percentile rank of Revenue_YoY_Growth

    # ── Sentiment (t-1 lagged — leakage-free) ────────────────────────
    "Announcement_Score",    # NSE point-in-time event score (kept same-day — discrete)
                             # TIMING ASSUMPTION: prediction made after market close (EOD).
                             # If used in pre-open prediction, shift(-1) must be applied here.
    "Sentiment_Score_Lag1",  # Composite sentiment score shifted 1 trading day back

    # ── Macro regime ──────────────────────────────────────────────────
    "Regime_Int",            # 0=Bear 1=Bull 2=HighVol 3=Sideways (from market_regimes)

    # ── Stock-specific macro sensitivities (rolling 252d, min 63d) ────
    "Beta_to_Nifty50",
    "Beta_to_Nifty500",
    "Beta_to_USDINR",
    "Beta_to_VIX",
    "Beta_to_Crude",
    "Beta_to_Gold",
    "Beta_to_FII",
    "Beta_to_DII",

    # ── Macro change signal ───────────────────────────────────────────
    "Repo_Rate_Change",          # diff(Repo_Rate) — nonzero only on cut/hike dates

    # ── Cross-sectional macro interaction ─────────────────────────────
    "USDINR_x_Revenue_Growth",   # USDINR_1d_Return × Revenue_YoY_Growth

    # ── Missing-data flags (carry information — NULLs not equal) ─────
    "Price_Gap_Flag",
    "SMA200_Available",
    "SMA50_Available",
    "ADX14_Available",
    "Volatility20d_Available",
    "Sentiment_Available",
    "Fundamentals_Available",
    "Data_Tier",

    # ── Target variables ──────────────────────────────────────────────
    "Target_Return_21d",         # Raw forward return (regression target)
    "Target_Rank_21d",           # Cross-sectional percentile rank — primary ML target
    "Target_Direction_Median",   # 1 if rank > 0.5 else 0 — binary classification
    "Target_Direction_Tertile",  # 1=top third, 0=bottom third, NULL=middle
    "Target_Vol_5d",             # Forward realized volatility (LSTM dual-head)
]


# ═══════════════════════════════════════════════════════════
# PHASE 0 — SETUP & VALIDATION
# ═══════════════════════════════════════════════════════════

def validate_inputs() -> bool:
    """Verify all source tables exist and have data."""
    required = {
        "ohlcv"         : "nifty500_ohlcv",
        "indicators"    : "nifty500_indicators",
        "fundamentals"  : "nifty500_fundamentals",
        "macro"         : "macro_indicators",
        "sentiment"     : "nifty500_sentiment",
        "data_quality"  : "stock_data_quality",
        "market_regimes": "market_regimes",
    }
    print("🔍 Validating input tables...")
    all_good = True
    with engine.connect() as conn:
        for key, tbl in required.items():
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
                if count == 0:
                    print(f"  ❌ {tbl}: EMPTY (0 rows)")
                    all_good = False
                else:
                    print(f"  ✅ {tbl}: {count:,} rows")
            except Exception as e:
                print(f"  ❌ {tbl}: ERROR — {e}")
                all_good = False
    return all_good


def truncate_output_tables():
    """Truncate features_master and sector_fundamentals_median before rebuild."""
    print("\n🧹 Truncating output tables...")
    with engine.connect() as conn:
        for key in ("features", "sector_median"):
            tbl = TABLES[key]
            try:
                conn.execute(text(f"TRUNCATE TABLE {tbl}"))
                conn.commit()
                print(f"  ✅ {tbl} truncated")
            except Exception as e:
                print(f"  ⚠️  {tbl}: {e}")


def load_tier_map() -> dict:
    """
    Load stock_data_quality → {Ticker: Data_Tier_numeric}.
    Tier X tickers excluded from returned dict.
    """
    df = pd.read_sql(
        f"SELECT Ticker, Data_Tier FROM {TABLES['data_quality']}",
        con=engine,
    )
    tier_map   = {}
    excluded_x = 0
    for _, row in df.iterrows():
        tier = str(row["Data_Tier"]).strip().upper()
        if tier == "X":
            excluded_x += 1
            continue
        if tier in TIER_MAP:
            tier_map[row["Ticker"]] = TIER_MAP[tier]
    print(f"  ✅ Loaded {len(tier_map)} tickers (Tier A+B+C) | Excluded {excluded_x} Tier X")
    return tier_map


def load_sector_map() -> dict:
    """
    Load {Ticker: Sector} for all tickers in features_master scope.
    Used by Phase 3.5 to assign sector to each ticker for within-sector rank.
    """
    df = pd.read_sql(
        f"SELECT Ticker, Data_Tier FROM {TABLES['data_quality']}",
        con=engine,
    )
    sector_map = {}
    for _, row in df.iterrows():
        ticker = row["Ticker"]
        sector_map[ticker] = get_sector(ticker)
    return sector_map


# ═══════════════════════════════════════════════════════════
# PHASE 1 — PRE-COMPUTATION
# ═══════════════════════════════════════════════════════════

def load_macro() -> pd.DataFrame:
    """
    Load macro_indicators and compute all derived series needed downstream.

    Returns a daily DataFrame with intermediate columns used only for
    computation (Ret_*, Nifty500_Return_*, USDINR_1d_Return, FII_Flow_Best,
    Ret_FII, Ret_DII). These are never written to features_master — they
    are consumed by compute_macro_sensitivities() and compute_relative_strength().

    Only Repo_Rate_Change reaches features_master as a stored column.

    DESIGN DECISIONS:
    - Raw India_VIX, USDINR, Crude_Oil, Gold NOT stored in features_master.
      Betas (Beta_to_VIX, Beta_to_USDINR etc.) capture stock-specific
      sensitivity to these factors, which is cross-sectionally meaningful.
      Raw levels are identical for every stock on a given date — zero
      discriminative power in cross-sectional training.
    - Beta_to_FII uses FII flow LEVEL (not pct_change). Monthly flow is a
      level signal (large positive = FII net buyer). pct_change on a flat
      forward-filled monthly series produces near-zero variance, collapsing
      beta to NaN for all historical rows.
    - Repo_Rate_Change (diff) is zero on most days; nonzero only on cut/hike
      dates. This is cross-sectionally valid — all stocks experience the same
      policy shock, but react differently via their Beta_to_Nifty50 etc.
    """
    df = pd.read_sql(
        f"""
        SELECT Date, India_VIX, USDINR, Crude_Oil, Gold,
               Nifty50_Close, Nifty500_Close,
               Repo_Rate,
               FII_Monthly_Net_Cr, FII_Daily_Net_Cr,
               DII_Monthly_Net_Cr, DII_Daily_Net_Cr,
               FII_Source_Flag
        FROM {TABLES['macro']}
        ORDER BY Date
        """,
        con=engine,
    )
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
    df = df.sort_values("Date").reset_index(drop=True)

    # ── Daily return series for rolling beta regression ─────────────────
    for col, ret_col in [
        ("Nifty50_Close",  "Ret_Nifty50"),
        ("Nifty500_Close", "Ret_Nifty500"),
        ("USDINR",         "Ret_USDINR"),
        ("India_VIX",      "Ret_VIX"),
        ("Crude_Oil",      "Ret_Crude"),
        ("Gold",           "Ret_Gold"),
    ]:
        df[ret_col] = pd.to_numeric(df[col], errors="coerce").pct_change()

    # ── Nifty500 rolling returns for RS_21d / RS_60d ────────────────────
    nifty500 = pd.to_numeric(df["Nifty500_Close"], errors="coerce")
    df["Nifty500_Return_21d"] = nifty500.pct_change(21)
    df["Nifty500_Return_60d"] = nifty500.pct_change(60)

    # ── FII best flow: daily if available, else monthly ──────────────────
    fii_daily   = pd.to_numeric(df["FII_Daily_Net_Cr"],   errors="coerce")
    fii_monthly = pd.to_numeric(df["FII_Monthly_Net_Cr"], errors="coerce")
    dii_monthly = pd.to_numeric(df["DII_Monthly_Net_Cr"], errors="coerce")
    df["FII_Flow_Best"] = fii_daily.where(df["FII_Source_Flag"] == "daily", fii_monthly)

    # Factor series for beta regressions
    fii_mean = df["FII_Flow_Best"].rolling(252, min_periods=63).mean()
    fii_std  = df["FII_Flow_Best"].rolling(252, min_periods=63).std().replace(0, np.nan)
    df["Ret_FII"] = (df["FII_Flow_Best"] - fii_mean) / fii_std  # z-scored level

    dii_mean = dii_monthly.rolling(252, min_periods=63).mean()
    dii_std  = dii_monthly.rolling(252, min_periods=63).std().replace(0, np.nan)
    df["Ret_DII"] = (dii_monthly - dii_mean) / dii_std           # z-scored level

    # ── Repo_Rate_Change: first-difference ──────────────────────────────
    df["Repo_Rate_Change"] = pd.to_numeric(df["Repo_Rate"], errors="coerce").diff()

    # ── USDINR 1-day return for interaction term ─────────────────────────
    df["USDINR_1d_Return"] = df["Ret_USDINR"]

    print(f"  ✅ Macro: {len(df):,} rows | {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def load_sentiment() -> pd.DataFrame:
    """
    Load nifty500_sentiment aggregate (one row per Ticker+Date).

    Only Announcement_Score and Sentiment_Score are loaded:
      - Announcement_Score: discrete NSE event signal, kept same-day (point-in-time)
      - Sentiment_Score: will be shifted by 1 trading day inside process_ticker()
        to become Sentiment_Score_Lag1 — prevents same-day leakage

    News_Sentiment_Score, Positive_Score, Negative_Score are excluded:
      - Highly correlated with Sentiment_Score
      - Their marginal information is already captured by Sentiment_Score_Lag1
      - Dropping them reduces feature redundancy and collinearity
    """
    df = pd.read_sql(
        f"""
        SELECT Ticker, Date, Announcement_Score, Sentiment_Score
        FROM {TABLES['sentiment']}
        """,
        con=engine,
    )
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
    print(f"  ✅ Sentiment: {len(df):,} rows | {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def load_market_regimes() -> pd.DataFrame:
    """
    Load market_regimes → daily Regime_Int per Date.
    Regime_Int: 0=Bear, 1=Bull, 2=HighVol, 3=Sideways.
    Cross-sectionally valid — same regime for all stocks on a date.
    Fails gracefully: returns empty DataFrame; Regime_Int will be NULL.
    """
    try:
        df = pd.read_sql(
            "SELECT Date, Regime_Int FROM market_regimes ORDER BY Date",
            con=engine,
        )
        df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
        print(f"  ✅ Market regimes: {len(df):,} rows | {df['Date'].min().date()} → {df['Date'].max().date()}")
        return df
    except Exception as e:
        print(f"  ⚠️  market_regimes load failed ({e}) — Regime_Int will be NULL")
        return pd.DataFrame(columns=["Date", "Regime_Int"])


def compute_fundamental_ratios(fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all annual-derived ratios from raw fundamentals.

    INPUT COLUMNS REQUIRED:
      Ticker, Period, Revenue, Gross_Profit, EBITDA, Net_Income,
      Free_Cash_Flow, Total_Assets, Total_Debt, Operating_CF,
      ROA, Debt_to_Equity

    EPS_Basic is excluded — raw EPS is non-comparable across tickers
    (MRF ~₹80,000 vs a PSU bank at ₹5). Delta_ROA captures profitability
    trajectory without the scale problem.

    BANKING/NBFC GROSS MARGIN:
      Gross_Profit is structurally NULL for banks and NBFCs (no COGS).
      Fallback: use Net_Income as numerator (closest analogue to net
      interest margin). Gross_Margin_Is_Proxy=1 flags these rows.

    DELTA COLUMNS:
      Computed as period-over-period diff within each ticker (sorted by
      Period ascending). First period per ticker is NaN — correct behaviour.
    """
    df = fund_df.copy()
    df["Period"] = pd.to_datetime(df["Period"])
    df = df.sort_values(["Ticker", "Period"]).reset_index(drop=True)

    def safe_div(num, den):
        num = pd.to_numeric(num, errors="coerce")
        den = pd.to_numeric(den, errors="coerce")
        return (num / den).replace([np.inf, -np.inf], np.nan)

    # ── Revenue YoY growth ──────────────────────────────────────────────
    df["Revenue_prev"]       = df.groupby("Ticker")["Revenue"].shift(1)
    df["Revenue_YoY_Growth"] = safe_div(
        df["Revenue"] - df["Revenue_prev"], df["Revenue_prev"]
    )

    # ── Margin ratios ────────────────────────────────────────────────────
    df["EBITDA_Margin"] = safe_div(df["EBITDA"],         df["Revenue"])
    df["Net_Margin"]    = safe_div(df["Net_Income"],     df["Revenue"])
    df["FCF_Margin"]    = safe_div(df["Free_Cash_Flow"], df["Revenue"])

    # ── Asset efficiency ─────────────────────────────────────────────────
    df["Asset_Turnover"] = safe_div(df["Revenue"], df["Total_Assets"])

    # ── Solvency ─────────────────────────────────────────────────────────
    df["Debt_to_Assets"] = safe_div(df["Total_Debt"], df["Total_Assets"])

    # ── Earnings quality ─────────────────────────────────────────────────
    df["OCF_to_Net_Income"] = safe_div(df["Operating_CF"], df["Net_Income"])

    # ── Gross Profit Margin (sector-aware fallback) ───────────────────────
    df["Sector"] = df["Ticker"].map(lambda t: get_sector(t))

    gross_profit  = pd.to_numeric(df["Gross_Profit"], errors="coerce")
    net_income    = pd.to_numeric(df["Net_Income"],   errors="coerce")
    revenue       = pd.to_numeric(df["Revenue"],      errors="coerce")
    is_financial  = df["Sector"].isin(BANKING_SECTORS)
    fallback_mask = is_financial | gross_profit.isna()

    numerator = gross_profit.copy()
    numerator[fallback_mask] = net_income[fallback_mask]

    df["Gross_Profit_Margin"]  = safe_div(numerator, revenue)
    df["Gross_Margin_Is_Proxy"] = fallback_mask.astype(float)

    # ── ROA and Debt_to_Equity sourced directly from fundamentals ────────
    # (already in fund_df — no computation needed)

    # ── Delta features (YoY direction) ───────────────────────────────────
    df["Delta_ROA"]        = df.groupby("Ticker")["ROA"].diff()
    df["Delta_DebtEquity"] = df.groupby("Ticker")["Debt_to_Equity"].diff()

    # ── Effective date = Period + 60-day availability lag ────────────────
    # Forced to ns precision — Timedelta addition produces datetime64[s] in
    # some pandas versions, breaking merge_asof against datetime64[us] from MySQL.
    df["Effective_Date"] = (
        df["Period"] + pd.Timedelta(days=FUND_LAG_DAYS)
    ).astype("datetime64[ns]")
    df["Period"] = df["Period"].astype("datetime64[ns]")

    keep_cols = ["Ticker", "Period", "Effective_Date", "Sector"] + RATIO_COLS
    return df[keep_cols]


def compute_sector_medians(ratios_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per (Sector, Period) median of each ratio column.
    Uses only tickers with actual (non-NaN) values.
    Returns NaN for a column if < SECTOR_MED_MIN_TICKERS tickers have data.

    Gross_Margin_Is_Proxy excluded from aggregation (it is a flag, not a ratio).
    """
    print("\n📊 Computing sector medians...")

    agg_cols = [c for c in RATIO_COLS if c != "Gross_Margin_Is_Proxy"]
    records  = []

    for (sector, period), group in ratios_df.groupby(["Sector", "Period"]):
        row = {"Sector": sector, "Period": period.date(), "Ticker_Count": len(group)}
        for col in agg_cols:
            non_null = group[col].dropna()
            row[col] = float(non_null.median()) if len(non_null) >= SECTOR_MED_MIN_TICKERS else None
        records.append(row)

    med_df = pd.DataFrame(records)
    print(f"  ✅ {len(med_df):,} sector-period combinations")

    if not med_df.empty:
        save_to_db(med_df, TABLES["sector_median"], engine)

    return med_df


def build_sector_median_lookup(med_df: pd.DataFrame) -> dict:
    """
    Build {Sector: DataFrame(Period, SecMed_*)} sorted by Period.
    Used for fast merge_asof-based fallback and Rel_* computation.
    """
    agg_cols = [c for c in RATIO_COLS if c != "Gross_Margin_Is_Proxy"]
    lookup   = {}
    for sector, group in med_df.groupby("Sector"):
        g = group.copy()
        g["Period"] = pd.to_datetime(g["Period"]).astype("datetime64[ns]")
        g = g.sort_values("Period").reset_index(drop=True)
        g.rename(columns={col: f"SecMed_{col}" for col in agg_cols}, inplace=True)
        lookup[sector] = g[["Period"] + [f"SecMed_{c}" for c in agg_cols]]
    return lookup


# ═══════════════════════════════════════════════════════════
# PHASE 2 — PER-TICKER FEATURE CONSTRUCTION
# ═══════════════════════════════════════════════════════════

def load_ticker_ohlcv(ticker: str) -> pd.DataFrame:
    """
    Load OHLCV for one ticker.
    High and Low are needed for Price_to_52W_High/Low computation.
    Adj_Close is needed for all return and ratio computations.
    Raw Open/High/Low/Close/Adj_Close/Volume/VWAP_Daily are NOT written
    to features_master — they are used here only as computation inputs.
    """
    df = pd.read_sql(
        f"""
        SELECT Date, Ticker, High, Low, Adj_Close, Volume, VWAP_Daily
        FROM {TABLES['ohlcv']}
        WHERE Ticker = %s
        ORDER BY Date
        """,
        con=engine,
        params=(ticker,),
    )
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
    return df


def load_ticker_indicators(ticker: str) -> pd.DataFrame:
    """
    Load indicators for one ticker.
    Raw SMA/EMA/BB/OBV levels are fetched because they are needed to compute
    the stationary derived features (Price_to_SMA*, BB_PctB, OBV_Change_5d).
    The raw levels themselves are NOT written to features_master.
    MACD and MACD_Signal fetched to verify MACD_Hist; only Hist stored.
    Stoch_D fetched but discarded — only Stoch_K stored.
    """
    df = pd.read_sql(
        f"""
        SELECT Date,
               SMA_20, SMA_50, SMA_200,
               MACD_Hist, RSI_14,
               BB_Upper, BB_Middle, BB_Lower,
               ATR_14, Stoch_K, ADX_14, OBV, VWAP_Dev
        FROM {TABLES['indicators']}
        WHERE Ticker = %s
        ORDER BY Date
        """,
        con=engine,
        params=(ticker,),
    )
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
    return df


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Inf-guarded division. Zero denominators → NaN."""
    return (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def compute_price_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all stationary price-derived features from OHLCV + indicator inputs.

    All computations use Adj_Close (split-adjusted). Raw price levels
    (Adj_Close, High, Low, Volume) are used only as intermediate inputs
    and are dropped from the returned DataFrame via FEATURES_COLUMNS selection
    in process_ticker().

    WHAT IS COMPUTED:
      Returns      : 1d, 5d, 21d, 60d pct_change + log return
      Volatility   : annualized 20d realized vol from log returns
      Volume       : Volume_Ratio_20d (relative — stationary)
      Trend ratios : Price_to_SMA20/50/200 (stationary: ratio to moving average)
      Momentum     : Price_to_52W_High, Price_to_52W_Low (252d rolling)
      Bollinger    : BB_Width (bandwidth) + BB_PctB (position within band)
      OBV          : OBV_Change_5d (5d pct change — stationary)

    Fix A (v2.3) — Momentum raw signals:
      Mom_12_1     : Return_252d - Return_21d (12-1 month momentum)
      Mom_6_1      : Return_126d - Return_21d (6-1 month momentum)
      These are raw per-ticker values. Cross-sectional ranks (Mom_12_1_Rank etc.)
      are computed in Phase 3.5 after all tickers are written to features_master.

    NOTE: RS_21d / RS_60d need Nifty500 benchmark from macro_df and are
          computed in process_ticker() after the macro join.
    """
    df   = df.copy()
    adj  = pd.to_numeric(df["Adj_Close"], errors="coerce")
    vol  = pd.to_numeric(df["Volume"],    errors="coerce")
    high = pd.to_numeric(df["High"],      errors="coerce")
    low  = pd.to_numeric(df["Low"],       errors="coerce")

    # ── Returns ──────────────────────────────────────────────────────────
    df["Return_1d"]     = adj.pct_change(1)
    df["Return_5d"]     = adj.pct_change(5)
    df["Return_21d"]    = adj.pct_change(21)
    df["Return_60d"]    = adj.pct_change(60)
    df["Log_Return_1d"] = np.log(adj / adj.shift(1))

    # ── Fix A: raw momentum signals (cross-sectional ranks in Phase 3.5) ─
    # Return_252d: 12-month trailing return
    # Return_126d: 6-month trailing return
    # Mom_12_1 = 12m return - 1m return (skips most recent month,
    #             standard momentum construction to avoid short-term reversal)
    # Mom_6_1  = 6m return - 1m return (same logic, shorter horizon)
    df["_Return_252d"] = adj.pct_change(252)
    df["_Return_126d"] = adj.pct_change(126)
    df["Mom_12_1"]     = df["_Return_252d"] - df["Return_21d"]
    df["Mom_6_1"]      = df["_Return_126d"] - df["Return_21d"]

    # ── Annualized realized volatility ───────────────────────────────────
    df["Volatility_20d"] = (
        df["Log_Return_1d"].rolling(window=20, min_periods=20).std(ddof=1)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    # ── Volume ratio (relative, stationary) ──────────────────────────────
    df["Volume_Ratio_20d"] = vol / vol.rolling(window=20, min_periods=20).mean()

    # ── Price-to-SMA ratios ───────────────────────────────────────────────
    sma20  = pd.to_numeric(df["SMA_20"],  errors="coerce") if "SMA_20"  in df.columns else pd.Series(np.nan, index=df.index)
    sma50  = pd.to_numeric(df["SMA_50"],  errors="coerce") if "SMA_50"  in df.columns else pd.Series(np.nan, index=df.index)
    sma200 = pd.to_numeric(df["SMA_200"], errors="coerce") if "SMA_200" in df.columns else pd.Series(np.nan, index=df.index)

    df["Price_to_SMA20"]  = safe_ratio(adj, sma20)
    df["Price_to_SMA50"]  = safe_ratio(adj, sma50)
    df["Price_to_SMA200"] = safe_ratio(adj, sma200)

    # ── 52-week high/low proximity (252 trading days, min 63) ────────────
    df["Price_to_52W_High"] = safe_ratio(adj, high.rolling(252, min_periods=63).max())
    df["Price_to_52W_Low"]  = safe_ratio(adj, low.rolling(252,  min_periods=63).min())

    # ── Bollinger Band features ───────────────────────────────────────────
    bb_u = pd.to_numeric(df["BB_Upper"],  errors="coerce") if "BB_Upper"  in df.columns else pd.Series(np.nan, index=df.index)
    bb_l = pd.to_numeric(df["BB_Lower"],  errors="coerce") if "BB_Lower"  in df.columns else pd.Series(np.nan, index=df.index)
    bb_m = pd.to_numeric(df["BB_Middle"], errors="coerce") if "BB_Middle" in df.columns else pd.Series(np.nan, index=df.index)
    bb_range = (bb_u - bb_l).replace(0, np.nan)

    df["BB_Width"] = safe_ratio(bb_range, bb_m)
    df["BB_PctB"]  = (adj - bb_l) / bb_range   # 0=lower band, 1=upper band

    # ── OBV 5-day rate-of-change ──────────────────────────────────────────
    obv = pd.to_numeric(df["OBV"], errors="coerce") if "OBV" in df.columns else pd.Series(np.nan, index=df.index)
    df["OBV_Change_5d"] = obv.pct_change(5)

    return df


def compute_price_gap_flag(df: pd.DataFrame) -> pd.Series:
    """1 if prior row gap > PRICE_GAP_THRESHOLD calendar days, else 0."""
    return (df["Date"].diff().dt.days.fillna(0) > PRICE_GAP_THRESHOLD).astype("int8")


def compute_targets(df: pd.DataFrame, max_date: pd.Timestamp) -> pd.DataFrame:
    """
    Compute forward-looking targets. NULLed where window extends past max_date.

    Target_Return_21d : (Adj_Close[t+21] / Adj_Close[t]) - 1
    Target_Vol_5d     : annualized std of log_returns[t+1..t+5], ddof=1
    Rank targets      : computed in Phase 3 (cross-sectional, needs all tickers)
    """
    df  = df.copy()
    adj = pd.to_numeric(df["Adj_Close"], errors="coerce")

    df["Target_Return_21d"] = (adj.shift(-TARGET_HORIZON) / adj) - 1

    future_returns = pd.concat(
        [df["Log_Return_1d"].shift(-i) for i in range(1, TARGET_VOL_WINDOW + 1)],
        axis=1
    )
    df["Target_Vol_5d"] = future_returns.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

    cutoff = df["Date"] > (max_date - pd.Timedelta(days=TARGET_HORIZON))
    df.loc[cutoff, ["Target_Return_21d", "Target_Vol_5d"]] = np.nan

    return df


def merge_fundamentals(ticker_df: pd.DataFrame,
                       ticker: str,
                       ratios_df: pd.DataFrame,
                       sector_lookup: dict) -> pd.DataFrame:
    """
    Attach fundamental ratios to ticker's daily rows using merge_asof.

    STEP 1 — Ticker's own fundamentals:
      Find the latest Effective_Date <= row Date. If stale
      (> FUND_MAX_FFILL_DAYS since period end) treat as missing.

    STEP 2 — Sector median fallback:
      For rows with no valid own-fundamental, use sector median
      at the equivalent period. Fundamentals_Available stays 0.

    STEP 3 — Sector-relative features:
      Rel_ROA = ROA - SecMed_ROA at matched period.
      Rel_EBITDA_Margin = EBITDA_Margin - SecMed_EBITDA_Margin.
      For rows using sector-median fallback, Rel_* = 0
      (stock value equals sector median by construction).
    """
    df = ticker_df.copy()

    tick_funds = ratios_df[ratios_df["Ticker"] == ticker].copy()
    tick_funds = tick_funds.sort_values("Effective_Date").reset_index(drop=True)

    all_fund_cols = RATIO_COLS + REL_COLS
    for col in all_fund_cols:
        df[col] = np.nan
    df["Fundamentals_Available"] = 0
    df["_Matched_Period"]        = pd.NaT

    # ── Step 1: Own fundamentals ──────────────────────────────────────────
    if not tick_funds.empty:
        df_sorted = (
            df.drop(columns=[c for c in all_fund_cols if c in df.columns])
              .sort_values("Date").reset_index()
        )
        df_sorted["Date"]            = df_sorted["Date"].astype("datetime64[ns]")
        tick_funds["Effective_Date"] = tick_funds["Effective_Date"].astype("datetime64[ns]")
        tick_funds["Period"]         = tick_funds["Period"].astype("datetime64[ns]")

        merged = pd.merge_asof(
            df_sorted,
            tick_funds[["Effective_Date", "Period"] + RATIO_COLS],
            left_on="Date",
            right_on="Effective_Date",
            direction="backward",
            allow_exact_matches=True,
        )

        days_since = (merged["Date"] - merged["Period"]).dt.days
        stale      = (days_since > FUND_MAX_FFILL_DAYS) | days_since.isna()
        merged     = merged.set_index("index").sort_index()
        stale.index = merged.index

        valid = ~stale & merged["Effective_Date"].notna()

        for col in RATIO_COLS:
            if col in merged.columns:
                df.loc[valid, col] = merged.loc[valid, col].values

        df.loc[valid, "Fundamentals_Available"] = 1
        df.loc[valid, "_Matched_Period"]        = merged.loc[valid, "Period"].values

    # ── Step 2: Sector median fallback ───────────────────────────────────
    sector      = get_sector(ticker)
    needs_fall  = df["Fundamentals_Available"] == 0

    if needs_fall.any() and sector in sector_lookup:
        sec_med = sector_lookup[sector].copy()
        sec_med["Period"] = sec_med["Period"].astype("datetime64[ns]")

        if not sec_med.empty:
            agg_cols = [c for c in RATIO_COLS if c != "Gross_Margin_Is_Proxy"]

            fb_rows = df.loc[needs_fall, ["Date"]].copy()
            fb_rows["Join_Date"] = (
                fb_rows["Date"] - pd.Timedelta(days=FUND_LAG_DAYS)
            ).astype("datetime64[ns]")
            fb_rows = fb_rows.sort_values("Join_Date")
            fb_rows["_orig_idx"] = fb_rows.index

            fb = pd.merge_asof(
                fb_rows, sec_med,
                left_on="Join_Date", right_on="Period",
                direction="backward", allow_exact_matches=True,
            ).set_index("_orig_idx").sort_index()

            for col in agg_cols:
                sc = f"SecMed_{col}"
                if sc in fb.columns:
                    valid_fb = fb[sc].notna()
                    df.loc[fb.index[valid_fb], col] = fb.loc[valid_fb, sc].values

            if "Period" in fb.columns:
                valid_per = fb["Period"].notna()
                df.loc[fb.index[valid_per], "_Matched_Period"] = fb.loc[valid_per, "Period"].values

    # ── Step 3: Sector-relative features ─────────────────────────────────
    if sector in sector_lookup:
        sec_med = sector_lookup[sector].copy()
        sec_med["Period"] = pd.to_datetime(sec_med["Period"]).astype("datetime64[ns]")

        p2roa    = sec_med.set_index("Period")["SecMed_ROA"].to_dict()           if "SecMed_ROA"           in sec_med.columns else {}
        p2ebitda = sec_med.set_index("Period")["SecMed_EBITDA_Margin"].to_dict() if "SecMed_EBITDA_Margin" in sec_med.columns else {}

        df["_sm_roa"]    = df["_Matched_Period"].map(p2roa)
        df["_sm_ebitda"] = df["_Matched_Period"].map(p2ebitda)

        df["Rel_ROA"]           = pd.to_numeric(df["ROA"],          errors="coerce") - pd.to_numeric(df["_sm_roa"],    errors="coerce")
        df["Rel_EBITDA_Margin"] = pd.to_numeric(df["EBITDA_Margin"], errors="coerce") - pd.to_numeric(df["_sm_ebitda"], errors="coerce")

        df.drop(columns=["_sm_roa", "_sm_ebitda"], inplace=True)
    else:
        df["Rel_ROA"]           = np.nan
        df["Rel_EBITDA_Margin"] = np.nan

    df.drop(columns=["_Matched_Period"], errors="ignore", inplace=True)
    return df


def rolling_beta(stock_ret: pd.Series, factor: pd.Series,
                 window: int = 252, min_periods: int = 63) -> pd.Series:
    """
    Rolling OLS beta: cov(stock, factor) / var(factor).
    window=252 (1 year), min_periods=63 (1 quarter) — NULL below threshold.
    """
    cov  = stock_ret.rolling(window, min_periods=min_periods).cov(factor)
    var  = factor.rolling(window, min_periods=min_periods).var()
    return (cov / var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def compute_macro_sensitivities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling macro betas and USDINR interaction term.
    Called after macro_df has been left-joined onto df.

    Betas are stock-specific — the same VIX level affects a high-beta
    stock very differently from a low-beta stock. This cross-sectional
    variation is what makes betas useful where raw macro levels are not.
    """
    stock_ret = pd.to_numeric(df["Return_1d"], errors="coerce")

    for beta_col, factor_col in [
        ("Beta_to_Nifty50",  "Ret_Nifty50"),
        ("Beta_to_Nifty500", "Ret_Nifty500"),
        ("Beta_to_USDINR",   "Ret_USDINR"),
        ("Beta_to_VIX",      "Ret_VIX"),
        ("Beta_to_Crude",    "Ret_Crude"),
        ("Beta_to_Gold",     "Ret_Gold"),
        ("Beta_to_FII",      "Ret_FII"),
        ("Beta_to_DII",      "Ret_DII"),
    ]:
        if factor_col in df.columns:
            df[beta_col] = rolling_beta(stock_ret, pd.to_numeric(df[factor_col], errors="coerce"))
        else:
            df[beta_col] = np.nan

    # USDINR × Revenue interaction — uses daily return (stationary), not level
    nan_s  = pd.Series(np.nan, index=df.index)
    usdinr = pd.to_numeric(df["USDINR_1d_Return"],   errors="coerce") if "USDINR_1d_Return"   in df.columns else nan_s
    rev_g  = pd.to_numeric(df["Revenue_YoY_Growth"], errors="coerce") if "Revenue_YoY_Growth" in df.columns else nan_s
    df["USDINR_x_Revenue_Growth"] = usdinr * rev_g

    return df


def compute_relative_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    RS_21d = stock Return_21d - Nifty500 Return_21d
    RS_60d = stock Return_60d - Nifty500 Return_60d
    Called after macro_df join (which carries Nifty500_Return_21d/60d).
    """
    s21 = pd.to_numeric(df["Return_21d"], errors="coerce")
    s60 = pd.to_numeric(df["Return_60d"], errors="coerce")
    i21 = pd.to_numeric(df["Nifty500_Return_21d"], errors="coerce") if "Nifty500_Return_21d" in df.columns else pd.Series(np.nan, index=df.index)
    i60 = pd.to_numeric(df["Nifty500_Return_60d"], errors="coerce") if "Nifty500_Return_60d" in df.columns else pd.Series(np.nan, index=df.index)

    df["RS_21d"] = s21 - i21
    df["RS_60d"] = s60 - i60
    return df


def process_ticker(ticker: str,
                   data_tier: int,
                   ratios_df: pd.DataFrame,
                   sector_lookup: dict,
                   macro_df: pd.DataFrame,
                   sentiment_df: pd.DataFrame,
                   regime_df: pd.DataFrame,
                   dataset_max_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build the complete feature row set for one ticker.

    Processing order (order is important — later steps depend on earlier ones):
      1.  Load OHLCV (High, Low, Adj_Close, Volume for computation only)
      2.  Load indicators (SMA/BB/OBV levels for ratio computation)
      3.  Merge OHLCV + indicators on Date
      4.  Compute all stationary price-derived features (incl. Mom_12_1, Mom_6_1)
      5.  Set availability flags
      6.  Compute forward targets
      7.  Merge fundamentals (ticker own → sector fallback → Rel_ features)
      8.  Join macro_df (brings Ret_* series, Nifty500 benchmarks, Repo_Rate_Change)
      9.  Compute macro betas + USDINR interaction
      10. Compute RS_21d / RS_60d (needs macro benchmark)
      11. Merge sentiment (Sentiment_Score shifted to Sentiment_Score_Lag1)
      12. Merge Regime_Int
      13. Finalize: column selection, type-casting, return

    Fix A rank columns (Mom_1M_Rank, Mom_12_1_Rank, Mom_6_1_Rank, RS_21d_Rank)
    and Fix B rank columns (ROA_Sector_Rank, EBITDA_Margin_Sector_Rank,
    Revenue_Growth_Sector_Rank) are NOT computed here — they require
    cross-sectional groupby across all tickers and are computed in Phase 3.5.
    They are initialized to NaN here and filled by Phase 3.5.
    """
    # ── 1-3. OHLCV + Indicators ──────────────────────────────────────────
    ohlcv = load_ticker_ohlcv(ticker)
    if ohlcv.empty:
        return pd.DataFrame()

    indic = load_ticker_indicators(ticker)
    df    = ohlcv.merge(indic, on="Date", how="left")

    # ── 4. Price-derived features (incl. Mom_12_1, Mom_6_1) ─────────────
    df = compute_price_derived(df)

    # ── 5. Availability flags ────────────────────────────────────────────
    df["Price_Gap_Flag"]          = compute_price_gap_flag(df)
    df["SMA200_Available"]        = df["SMA_200"].notna().astype("int8")
    df["SMA50_Available"]         = df["SMA_50"].notna().astype("int8")
    df["ADX14_Available"]         = df["ADX_14"].notna().astype("int8")
    df["Volatility20d_Available"] = df["Volatility_20d"].notna().astype("int8")

    # ── 6. Forward targets ───────────────────────────────────────────────
    df = compute_targets(df, dataset_max_date)

    # ── 7. Fundamentals ──────────────────────────────────────────────────
    df = merge_fundamentals(df, ticker, ratios_df, sector_lookup)

    # ── 8. Macro join ────────────────────────────────────────────────────
    df = df.merge(macro_df, on="Date", how="left")

    # ── 9. Macro betas + interaction ─────────────────────────────────────
    df = compute_macro_sensitivities(df)

    # ── 10. Relative strength ────────────────────────────────────────────
    df = compute_relative_strength(df)

    # ── 11. Sentiment (t-1 lag) ──────────────────────────────────────────
    tick_sent = (
        sentiment_df[sentiment_df["Ticker"] == ticker]
        .drop(columns=["Ticker"])
        .sort_values("Date")
        .copy()
    )
    tick_sent["Sentiment_Score_Lag1"] = tick_sent["Sentiment_Score"].shift(1)
    tick_sent.drop(columns=["Sentiment_Score"], inplace=True)
    df = df.merge(tick_sent, on="Date", how="left")

    df["Sentiment_Available"] = (
        (df["Date"] >= pd.Timestamp(SENTIMENT_START_DATE))
        & (df["Announcement_Score"].notna() | df["Sentiment_Score_Lag1"].notna())
    ).astype("int8")

    # ── 12. Market regime ────────────────────────────────────────────────
    if not regime_df.empty:
        df = df.merge(regime_df, on="Date", how="left")
    else:
        df["Regime_Int"] = np.nan

    # ── 13. Finalize ─────────────────────────────────────────────────────
    df["Data_Tier"]               = data_tier
    df["Target_Rank_21d"]         = np.nan
    df["Target_Direction_Median"]  = pd.NA
    df["Target_Direction_Tertile"] = pd.NA
    df["Ticker"]                  = ticker

    # Initialize Fix A rank columns (Phase 3.5 will fill these)
    df["Mom_1M_Rank"]   = np.nan
    df["Mom_12_1_Rank"] = np.nan
    df["Mom_6_1_Rank"]  = np.nan
    df["RS_21d_Rank"]   = np.nan

    # Initialize Fix B sector rank columns (Phase 3.5 will fill these)
    df["ROA_Sector_Rank"]            = np.nan
    df["EBITDA_Margin_Sector_Rank"]  = np.nan
    df["Revenue_Growth_Sector_Rank"] = np.nan

    # Guarantee all expected columns exist before selection
    for col in FEATURES_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Select only the clean final column set — raw intermediates dropped here
    df = df[FEATURES_COLUMNS].copy()

    # ── MySQL type compatibility ──────────────────────────────────────────
    for col in ["Target_Direction_Median", "Target_Direction_Tertile"]:
        df[col] = df[col].astype("object").where(df[col].notna(), None)

    df["Gross_Margin_Is_Proxy"] = df["Gross_Margin_Is_Proxy"].where(
        df["Gross_Margin_Is_Proxy"].notna(), None
    )

    df["Date"] = df["Date"].dt.date
    return df


# ═══════════════════════════════════════════════════════════
# PHASE 3 — CROSS-SECTIONAL RANK PASS (target ranks)
# ═══════════════════════════════════════════════════════════

def compute_and_write_ranks():
    """
    Cross-sectional percentile rank of Target_Return_21d across
    Tier A+B (Data_Tier IN (1,2)) per Date.

    Writes back to features_master via temp-table JOIN UPDATE:
      - Target_Rank_21d          (percentile 0–1)
      - Target_Direction_Median  (1 if rank > 0.5 else 0)
      - Target_Direction_Tertile (1=top third, 0=bottom third, NULL=middle)

    Tier C rows remain NULL — not included in ranking universe.
    """
    print("\n🎯 Phase 3 — cross-sectional rank pass (target ranks)...")

    df = pd.read_sql(
        f"""
        SELECT Ticker, Date, Target_Return_21d
        FROM {TABLES['features']}
        WHERE Data_Tier IN (1, 2) AND Target_Return_21d IS NOT NULL
        """,
        con=engine,
    )
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"  📥 Loaded {len(df):,} Tier A+B rows with forward return")

    if df.empty:
        print("  ⚠️  No rows to rank — skipping")
        return

    df["Target_Rank_21d"] = (
        df.groupby("Date")["Target_Return_21d"].rank(method="average", pct=True)
    )
    df["Target_Direction_Median"] = (df["Target_Rank_21d"] > 0.5).astype("int8")

    def tertile(r):
        if pd.isna(r): return None
        if r > 2 / 3:  return 1
        if r < 1 / 3:  return 0
        return None

    df["Target_Direction_Tertile"] = df["Target_Rank_21d"].map(tertile)
    df["Target_Rank_21d"]          = df["Target_Rank_21d"].round(6)
    df["Date"]                     = df["Date"].dt.date

    upload_df = df[["Ticker", "Date", "Target_Rank_21d",
                    "Target_Direction_Median", "Target_Direction_Tertile"]].copy()
    upload_df["Target_Direction_Median"]  = upload_df["Target_Direction_Median"].astype("Int8")
    upload_df["Target_Direction_Tertile"] = upload_df["Target_Direction_Tertile"].astype("Int8")

    print("  📤 Writing target ranks to _tmp_ranks...")
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS _tmp_ranks"))
        conn.execute(text("""
            CREATE TABLE _tmp_ranks (
                Ticker                   VARCHAR(20) NOT NULL,
                Date                     DATE        NOT NULL,
                Target_Rank_21d          DECIMAL(10,6),
                Target_Direction_Median  TINYINT,
                Target_Direction_Tertile TINYINT,
                PRIMARY KEY (Ticker, Date)
            )
        """))
        conn.commit()

    upload_df.to_sql("_tmp_ranks", con=engine, if_exists="append",
                     index=False, chunksize=10000)
    print(f"  ✅ Wrote {len(upload_df):,} rank rows")

    print("  🔄 Applying target ranks to features_master via JOIN UPDATE...")
    with engine.connect() as conn:
        conn.execute(text(f"""
            UPDATE {TABLES['features']} f
            JOIN _tmp_ranks t ON f.Ticker = t.Ticker AND f.Date = t.Date
            SET f.Target_Rank_21d          = t.Target_Rank_21d,
                f.Target_Direction_Median  = t.Target_Direction_Median,
                f.Target_Direction_Tertile = t.Target_Direction_Tertile
        """))
        conn.commit()
        conn.execute(text("DROP TABLE IF EXISTS _tmp_ranks"))
        conn.commit()

    print("  ✅ Target ranks applied; temp table dropped")


# ═══════════════════════════════════════════════════════════
# PHASE 3.5 — CROSS-SECTIONAL FEATURE RANK PASS (Fix A + Fix B)
# ═══════════════════════════════════════════════════════════

def compute_and_write_feature_ranks(sector_map: dict):
    """
    Fix A — Cross-sectional momentum ranks (all tickers, per Date):
      Mom_1M_Rank   : percentile rank of Return_21d per date
      Mom_12_1_Rank : percentile rank of Mom_12_1 per date
      Mom_6_1_Rank  : percentile rank of Mom_6_1 per date
      RS_21d_Rank   : percentile rank of RS_21d per date

    Fix B — Within-sector fundamental ranks (per Sector × Date):
      ROA_Sector_Rank            : percentile rank of ROA within sector per date
      EBITDA_Margin_Sector_Rank  : percentile rank of EBITDA_Margin within sector per date
      Revenue_Growth_Sector_Rank : percentile rank of Revenue_YoY_Growth within sector per date

    All rank columns use method='average', pct=True → values in [0, 1].
    NULL where < SECTOR_RANK_MIN_TICKERS have non-NULL values in a group.
    Computed via temp-table JOIN UPDATE — not per-row SQL updates.

    DESIGN:
    - Fix A ranks are computed across ALL tickers on each date (Tier A+B+C
      all participate, making the rank more stable with ~380–400 stocks).
    - Fix B sector ranks are computed within each (Sector, Date) group.
      A minimum of SECTOR_RANK_MIN_TICKERS=3 non-NULL values is required.
    - Both fixes use the same temp-table pattern as Phase 3 for efficiency.
    """
    print("\n🎯 Phase 3.5 — feature rank pass (Fix A: momentum ranks, Fix B: sector ranks)...")

    # ── Load relevant columns from features_master ────────────────────────
    print("  📥 Loading momentum and fundamental columns from features_master...")
    df = pd.read_sql(
        f"""
        SELECT Ticker, Date,
               Return_21d, Mom_12_1, Mom_6_1, RS_21d,
               ROA, EBITDA_Margin, Revenue_YoY_Growth
        FROM {TABLES['features']}
        """,
        con=engine,
    )
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"  ✅ Loaded {len(df):,} rows")

    # ── Attach sector from sector_map ─────────────────────────────────────
    df["Sector"] = df["Ticker"].map(sector_map).fillna("Unknown")

    # ── Fix A: Cross-sectional momentum ranks (all tickers per date) ──────
    print("  📊 Computing Fix A: cross-sectional momentum ranks...")

    def cs_rank(series: pd.Series) -> pd.Series:
        """Percentile rank with NaN preserved. Returns NaN if < 2 non-null values."""
        valid = series.dropna()
        if len(valid) < 2:
            return pd.Series(np.nan, index=series.index)
        ranks = series.rank(method="average", pct=True)
        return ranks

    df["Mom_1M_Rank"]   = df.groupby("Date")["Return_21d"].transform(cs_rank)
    df["Mom_12_1_Rank"] = df.groupby("Date")["Mom_12_1"].transform(cs_rank)
    df["Mom_6_1_Rank"]  = df.groupby("Date")["Mom_6_1"].transform(cs_rank)
    df["RS_21d_Rank"]   = df.groupby("Date")["RS_21d"].transform(cs_rank)

    # ── Fix B: Within-sector fundamental ranks ────────────────────────────
    print("  📊 Computing Fix B: within-sector fundamental ranks...")

    def sector_rank_with_min(series: pd.Series) -> pd.Series:
        """
        Percentile rank within group.
        Returns NaN for entire group if < SECTOR_RANK_MIN_TICKERS have non-NULL values.
        """
        valid_count = series.notna().sum()
        if valid_count < SECTOR_RANK_MIN_TICKERS:
            return pd.Series(np.nan, index=series.index)
        return series.rank(method="average", pct=True)

    df["ROA_Sector_Rank"] = (
        df.groupby(["Date", "Sector"])["ROA"]
        .transform(sector_rank_with_min)
    )
    df["EBITDA_Margin_Sector_Rank"] = (
        df.groupby(["Date", "Sector"])["EBITDA_Margin"]
        .transform(sector_rank_with_min)
    )
    df["Revenue_Growth_Sector_Rank"] = (
        df.groupby(["Date", "Sector"])["Revenue_YoY_Growth"]
        .transform(sector_rank_with_min)
    )

    # Round all rank columns
    rank_cols = [
        "Mom_1M_Rank", "Mom_12_1_Rank", "Mom_6_1_Rank", "RS_21d_Rank",
        "ROA_Sector_Rank", "EBITDA_Margin_Sector_Rank", "Revenue_Growth_Sector_Rank",
    ]
    for col in rank_cols:
        df[col] = df[col].round(6)

    df["Date"] = df["Date"].dt.date

    # ── Upload via temp table + JOIN UPDATE ───────────────────────────────
    upload_df = df[["Ticker", "Date"] + rank_cols].copy()

    print(f"  📤 Writing {len(upload_df):,} rows to _tmp_feat_ranks...")
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS _tmp_feat_ranks"))
        conn.execute(text("""
            CREATE TABLE _tmp_feat_ranks (
                Ticker                      VARCHAR(20) NOT NULL,
                Date                        DATE        NOT NULL,
                Mom_1M_Rank                 DECIMAL(10,6),
                Mom_12_1_Rank               DECIMAL(10,6),
                Mom_6_1_Rank                DECIMAL(10,6),
                RS_21d_Rank                 DECIMAL(10,6),
                ROA_Sector_Rank             DECIMAL(10,6),
                EBITDA_Margin_Sector_Rank   DECIMAL(10,6),
                Revenue_Growth_Sector_Rank  DECIMAL(10,6),
                PRIMARY KEY (Ticker, Date)
            )
        """))
        conn.commit()

    # Write in chunks to avoid memory issues on large datasets
    chunk_size = 50000
    total_chunks = (len(upload_df) // chunk_size) + 1
    for i in range(0, len(upload_df), chunk_size):
        chunk = upload_df.iloc[i:i + chunk_size]
        chunk.to_sql("_tmp_feat_ranks", con=engine, if_exists="append",
                     index=False, chunksize=10000)
        if (i // chunk_size + 1) % 5 == 0:
            print(f"    chunk {i // chunk_size + 1}/{total_chunks} written...")

    print(f"  ✅ Wrote {len(upload_df):,} feature rank rows")

    print("  🔄 Applying feature ranks to features_master via JOIN UPDATE...")
    with engine.connect() as conn:
        conn.execute(text(f"""
            UPDATE {TABLES['features']} f
            JOIN _tmp_feat_ranks t ON f.Ticker = t.Ticker AND f.Date = t.Date
            SET f.Mom_1M_Rank                = t.Mom_1M_Rank,
                f.Mom_12_1_Rank              = t.Mom_12_1_Rank,
                f.Mom_6_1_Rank               = t.Mom_6_1_Rank,
                f.RS_21d_Rank                = t.RS_21d_Rank,
                f.ROA_Sector_Rank            = t.ROA_Sector_Rank,
                f.EBITDA_Margin_Sector_Rank  = t.EBITDA_Margin_Sector_Rank,
                f.Revenue_Growth_Sector_Rank = t.Revenue_Growth_Sector_Rank
        """))
        conn.commit()
        conn.execute(text("DROP TABLE IF EXISTS _tmp_feat_ranks"))
        conn.commit()

    # ── Summary stats ─────────────────────────────────────────────────────
    non_null_counts = {col: df[col].notna().sum() for col in rank_cols}
    print("\n  Non-NULL counts for new rank features:")
    for col, count in non_null_counts.items():
        pct = count / len(df) * 100
        print(f"    {col:<35}: {count:>10,}  ({pct:.1f}%)")

    print("  ✅ Feature ranks applied; temp table dropped")


# ═══════════════════════════════════════════════════════════
# PHASE 4 — VERIFICATION
# ═══════════════════════════════════════════════════════════

def print_verification():
    """Post-run summary statistics for all feature groups."""
    print("\n" + "=" * 60)
    print("VERIFICATION — features_master")
    print("=" * 60)

    with engine.connect() as conn:

        total = conn.execute(text(f"SELECT COUNT(*) FROM {TABLES['features']}")).scalar()
        print(f"\nTotal rows : {total:,}")
        print(f"Columns    : {len(FEATURES_COLUMNS)} (all training-usable)")

        print("\nRows per tier:")
        for row in conn.execute(text(f"""
            SELECT Data_Tier, COUNT(*) AS n, COUNT(DISTINCT Ticker) AS t
            FROM {TABLES['features']}
            GROUP BY Data_Tier ORDER BY Data_Tier
        """)):
            name = {1:"A", 2:"B", 3:"C"}.get(row[0], str(row[0]))
            print(f"  Tier {name}: {row[1]:>10,} rows | {row[2]:>4} tickers")

        d = conn.execute(text(f"""
            SELECT MIN(Date), MAX(Date), COUNT(DISTINCT Date)
            FROM {TABLES['features']}
        """)).fetchone()
        print(f"\nDate range : {d[0]} → {d[1]} ({d[2]:,} trading days)")

        # ── Targets ───────────────────────────────────────────────────────
        t = conn.execute(text(f"""
            SELECT
                SUM(CASE WHEN Target_Return_21d        IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN Target_Rank_21d          IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN Target_Direction_Median  IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN Target_Direction_Tertile IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN Target_Vol_5d            IS NOT NULL THEN 1 ELSE 0 END)
            FROM {TABLES['features']}
        """)).fetchone()
        print("\nTarget coverage (non-NULL rows):")
        print(f"  Target_Return_21d       : {t[0]:>12,}")
        print(f"  Target_Rank_21d         : {t[1]:>12,}")
        print(f"  Target_Direction_Median : {t[2]:>12,}")
        print(f"  Target_Direction_Tertile: {t[3]:>12,}")
        print(f"  Target_Vol_5d           : {t[4]:>12,}")

        # ── Flags ─────────────────────────────────────────────────────────
        fl = conn.execute(text(f"""
            SELECT
                SUM(Price_Gap_Flag), SUM(SMA200_Available), SUM(SMA50_Available),
                SUM(ADX14_Available), SUM(Volatility20d_Available),
                SUM(Sentiment_Available), SUM(Fundamentals_Available)
            FROM {TABLES['features']}
        """)).fetchone()
        print("\nFlag distributions (rows = 1):")
        for name, val in zip(
            ["Price_Gap_Flag", "SMA200_Available", "SMA50_Available",
             "ADX14_Available", "Volatility20d_Available",
             "Sentiment_Available", "Fundamentals_Available"], fl
        ):
            print(f"  {name:<28}: {val:>12,}")

        # ── Fix A: momentum rank coverage ─────────────────────────────────
        ma = conn.execute(text(f"""
            SELECT
                SUM(CASE WHEN Mom_12_1      IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN Mom_6_1       IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN Mom_1M_Rank   IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN Mom_12_1_Rank IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN Mom_6_1_Rank  IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN RS_21d_Rank   IS NOT NULL THEN 1 ELSE 0 END)
            FROM {TABLES['features']}
        """)).fetchone()
        print("\nFix A — Cross-sectional momentum rank coverage (non-NULL rows):")
        for name, val in zip(
            ["Mom_12_1 (raw)", "Mom_6_1 (raw)",
             "Mom_1M_Rank", "Mom_12_1_Rank", "Mom_6_1_Rank", "RS_21d_Rank"], ma
        ):
            print(f"  {name:<28}: {val:>12,}")

        # ── Fix B: sector rank coverage ────────────────────────────────────
        mb = conn.execute(text(f"""
            SELECT
                SUM(CASE WHEN ROA_Sector_Rank            IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN EBITDA_Margin_Sector_Rank  IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN Revenue_Growth_Sector_Rank IS NOT NULL THEN 1 ELSE 0 END)
            FROM {TABLES['features']}
        """)).fetchone()
        print("\nFix B — Within-sector fundamental rank coverage (non-NULL rows):")
        for name, val in zip(
            ["ROA_Sector_Rank", "EBITDA_Margin_Sector_Rank",
             "Revenue_Growth_Sector_Rank"], mb
        ):
            print(f"  {name:<35}: {val:>12,}")

        # ── Spot-check: rank distributions ────────────────────────────────
        print("\nRank distribution spot-check (should be near 0.5 mean for valid ranks):")
        for col in ["Mom_1M_Rank", "Mom_12_1_Rank", "ROA_Sector_Rank"]:
            row = conn.execute(text(f"""
                SELECT AVG({col}), MIN({col}), MAX({col})
                FROM {TABLES['features']}
                WHERE {col} IS NOT NULL
            """)).fetchone()
            if row and row[0] is not None:
                print(f"  {col:<35}: avg={float(row[0]):.4f}  min={float(row[1]):.4f}  max={float(row[2]):.4f}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("FEATURES PIPELINE — hedge_v2.3")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not validate_inputs():
        print("\n❌ Input validation failed. Exiting.")
        return

    truncate_output_tables()

    # ── Phase 1: Pre-computation ─────────────────────────────────────────
    print("\n📥 Loading tier classification...")
    tier_map = load_tier_map()
    if not tier_map:
        print("❌ No tickers in stock_data_quality. Exiting.")
        return

    print("\n📥 Loading sector map for Phase 3.5...")
    sector_map = load_sector_map()

    print("\n📥 Phase 1 — pre-computation...")
    macro_df  = load_macro()
    sent_df   = load_sentiment()
    regime_df = load_market_regimes()

    print("\n📊 Loading and computing fundamental ratios...")
    raw_fund = pd.read_sql(
        f"""
        SELECT Ticker, Period,
               Revenue, Gross_Profit, EBITDA, Net_Income,
               Free_Cash_Flow, Total_Assets, Total_Debt,
               Operating_CF, ROA, Debt_to_Equity
        FROM {TABLES['fundamentals']}
        """,
        con=engine,
    )
    print(f"  Loaded {len(raw_fund):,} fundamental rows")

    ratios_df = compute_fundamental_ratios(raw_fund)
    print(f"  ✅ Ratios computed for {len(ratios_df):,} (Ticker, Period) pairs")

    med_df        = compute_sector_medians(ratios_df)
    sector_lookup = build_sector_median_lookup(med_df)

    dataset_max_date = pd.to_datetime(
        pd.read_sql(f"SELECT MAX(Date) AS d FROM {TABLES['ohlcv']}", con=engine)["d"].iloc[0]
    )
    print(f"\n📅 Dataset max date: {dataset_max_date.date()}")

    # ── Phase 2: Per-ticker feature construction ──────────────────────────
    print("\n🔄 Phase 2 — per-ticker feature construction...")
    tickers            = sorted(tier_map.keys())
    total_tickers      = len(tickers)
    total_rows_written = 0
    failed             = []

    for idx, ticker in enumerate(tickers, 1):
        try:
            tier = tier_map[ticker]
            df   = process_ticker(
                ticker, tier, ratios_df, sector_lookup,
                macro_df, sent_df, regime_df, dataset_max_date,
            )

            if df.empty:
                print(f"  [{idx}/{total_tickers}] {ticker}: no OHLCV rows, skipped")
                continue

            df = df.replace([np.inf, -np.inf], np.nan)
            save_to_db(df, TABLES["features"], engine)
            total_rows_written += len(df)

            if idx % 25 == 0 or idx == total_tickers:
                print(f"  [{idx}/{total_tickers}] {ticker}: +{len(df):,} rows | total {total_rows_written:,}")

        except Exception as e:
            print(f"  [{idx}/{total_tickers}] {ticker}: ❌ {e}")
            failed.append((ticker, str(e)))

    print(f"\n✅ Phase 2 complete: {total_rows_written:,} rows written | {len(failed)} failures")
    if failed:
        for t, err in failed[:10]:
            print(f"    {t}: {err}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")

    # ── Phase 3: Cross-sectional target rank pass ─────────────────────────
    compute_and_write_ranks()

    # ── Phase 3.5: Cross-sectional feature rank pass (Fix A + Fix B) ─────
    compute_and_write_feature_ranks(sector_map)

    # ── Phase 4: Verification ─────────────────────────────────────────────
    print_verification()

    print("\n" + "=" * 60)
    print(f"DONE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\n📋 Next: python data/export_features.py")


if __name__ == "__main__":
    main()