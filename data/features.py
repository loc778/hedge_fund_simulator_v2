# data/features.py — hedge_v2
# ═══════════════════════════════════════════════════════════
# FEATURE ENGINEERING PIPELINE
#
# Combines all raw data layers into features_master — the single
# unified table consumed by all ML training notebooks.
#
# INPUTS (must be populated):
#   nifty500_ohlcv            1.28M rows
#   nifty500_indicators       ~1.15M rows (on Adj_Close)
#   nifty500_fundamentals     ~4,000 rows (annual, leakage-free)
#   macro_indicators          ~4,246 rows (daily, ffilled at ingest)
#   nifty500_sentiment        ~152,826 rows (daily aggregate)
#   stock_data_quality        500 rows (Tier A/B/C/X classification)
#
# OUTPUTS (TRUNCATED and rebuilt on every run):
#   features_master               ~1.2-1.5M rows, ~79 columns
#   sector_fundamentals_median    ~200-500 rows (Sector × Period)
#
# EXECUTION:
#   python data/features.py
#
# ARCHITECTURAL RULES APPLIED:
#   - Tier X tickers excluded (from stock_data_quality, not hardcoded)
#   - Tier A+B+C included; Tier C rows get NULL ranks (not trained)
#   - Target_Rank_21d computed cross-sectionally on Tier A+B only
#   - Fundamental availability lag = 60 days after fiscal year-end
#   - Sector median fallback requires ≥3 tickers with actual data
#   - All NULLs flow through — XGBoost handles natively, flags carry info
#   - No z-score normalization here — done in Colab per walk-forward fold
#   - Targets NULL for rows where forward window > max available date
#
# FIXES (Apr 2026 — macro leakage remediation):
#   - Raw macro level columns replaced with stock-specific betas
#   - USDINR, Crude_Oil, Gold retained as raw features (daily, cross-sectional)
#   - India_VIX retained as raw feature (cross-sectional regime signal)
#   - Beta_to_FII uses FII flow level (not pct_change) — monthly flow is
#     a level signal; pct_change on a flat monthly series produces near-zero
#     variance which collapses beta to NaN
#   - FII_Momentum_5d / DII_Momentum_5d cast to float (not Int8) to survive
#     pandas merge without nullable integer dtype corruption
#   - Repo_Rate and USDINR_1d_Return accessed directly after macro join
#     with explicit column existence guard
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

from config import TABLES, SECTOR_MAP, SENTIMENT_START_DATE, FEATURES, get_sector
from data.db import get_engine, save_to_db

engine = get_engine()

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════

FUND_LAG_DAYS          = FEATURES["fundamentals_availability_lag_days"]          # 60
SECTOR_MED_MIN_TICKERS = FEATURES["sector_median_min_tickers"]                   # 3
FUND_MAX_FFILL_DAYS    = FEATURES["fundamentals_max_forward_fill_days"]          # 400
PRICE_GAP_THRESHOLD    = FEATURES["price_gap_flag_threshold_days"]               # 5
TARGET_HORIZON         = FEATURES["target_horizon_days"]                         # 21
TARGET_VOL_WINDOW      = FEATURES["target_vol_window_days"]                      # 5
TRADING_DAYS_PER_YEAR  = FEATURES["trading_days_per_year"]                       # 252

# Tier string → numeric mapping
TIER_MAP = {"A": 1, "B": 2, "C": 3}

# Ratio columns to compute from annual fundamentals
RATIO_COLS = [
    "Revenue_YoY_Growth", "EBITDA_Margin", "Net_Margin", "FCF_Margin",
    "Asset_Turnover", "ROA", "Debt_to_Equity", "EPS_Basic",
]

# Final features_master column order (must match setup_db.py DDL, minus id)
FEATURES_COLUMNS = [
    "Date", "Ticker",
    # Price & volume
    "Open", "High", "Low", "Close", "Adj_Close", "Volume", "VWAP_Daily",
    # Derived price features
    "Return_1d", "Log_Return_1d", "Return_5d", "Return_21d", "Return_60d",
    "Volatility_20d", "Volume_Ratio_20d", "ADV_20d_Cr",
    # Technical indicators
    "SMA_20", "SMA_50", "SMA_200", "EMA_9", "EMA_21",
    "MACD", "MACD_Signal", "MACD_Hist", "RSI_14",
    "BB_Upper", "BB_Middle", "BB_Lower", "BB_Width",
    "ATR_14", "Stoch_K", "Stoch_D", "ADX_14", "OBV", "VWAP_Dev",
    # Fundamental ratios (computed here)
    "Revenue_YoY_Growth", "EBITDA_Margin", "Net_Margin", "FCF_Margin",
    "Asset_Turnover", "ROA", "Debt_to_Equity", "EPS_Basic",
    # Sentiment (5 cols)
    "Announcement_Score", "News_Sentiment_Score", "Sentiment_Score",
    "Positive_Score", "Negative_Score",
    # Macro regime — daily market prices kept as raw features
    # (cross-sectionally valid: daily variation, different sector impact)
    "India_VIX", "USDINR", "Crude_Oil", "Gold",
    # Stock-specific macro sensitivities (rolling 252d, min 63d)
    "Beta_to_Nifty50", "Beta_to_Nifty500",
    "Beta_to_USDINR", "Beta_to_VIX", "Beta_to_Crude", "Beta_to_Gold",
    "Beta_to_FII",
    # Macro regime signals (market-wide, low-frequency)
    "FII_Momentum_5d", "DII_Momentum_5d",
    # Cross-sectional macro interactions
    "RepoRate_x_DebtEquity", "USDINR_x_Revenue_Growth",
    # Flags
    "Price_Gap_Flag", "SMA200_Available", "SMA50_Available",
    "ADX14_Available", "Volatility20d_Available", "Sentiment_Available",
    "Fundamentals_Available", "Data_Tier",
    # Targets (rank columns populated in Phase 3)
    "Target_Return_21d", "Target_Rank_21d",
    "Target_Direction_Median", "Target_Direction_Tertile",
    "Target_Vol_5d",
]


# ═══════════════════════════════════════════════════════════
# PHASE 0 — SETUP & VALIDATION
# ═══════════════════════════════════════════════════════════

def validate_inputs() -> bool:
    """Verify all source tables exist and have data."""
    required = {
        "ohlcv"        : "nifty500_ohlcv",
        "indicators"   : "nifty500_indicators",
        "fundamentals" : "nifty500_fundamentals",
        "macro"        : "macro_indicators",
        "sentiment"    : "nifty500_sentiment",
        "data_quality" : "stock_data_quality",
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
    tier_map = {}
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


# ═══════════════════════════════════════════════════════════
# PHASE 1 — PRE-COMPUTATION
# ═══════════════════════════════════════════════════════════

def load_macro() -> pd.DataFrame:
    """
    Load macro_indicators. Returns daily DataFrame with:
      - India_VIX, USDINR, Crude_Oil, Gold (raw — kept as features)
      - Daily return series for rolling beta regression
      - FII/DII flow columns for momentum signals and Beta_to_FII
      - Repo_Rate for interaction term

    FIX: Beta_to_FII uses FII flow LEVEL (not pct_change).
         Monthly FII flow is a level signal — large positive = FII buying.
         pct_change on a flat monthly series produces near-zero variance,
         collapsing rolling beta to NaN for all historical rows.

    FIX: FII_Momentum_5d / DII_Momentum_5d cast to float (not Int8).
         Nullable integer dtype (Int8) survives in-memory operations but
         corrupts to all-NULL after pandas merge in process_ticker().
         Float dtype merges cleanly.
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

    # Daily return series for rolling beta regression
    for col, ret_col in [
        ("Nifty50_Close",  "Ret_Nifty50"),
        ("Nifty500_Close", "Ret_Nifty500"),
        ("USDINR",         "Ret_USDINR"),
        ("India_VIX",      "Ret_VIX"),
        ("Crude_Oil",      "Ret_Crude"),
        ("Gold",           "Ret_Gold"),
    ]:
        df[ret_col] = pd.to_numeric(df[col], errors="coerce").pct_change()

    # FII net flow — prefer daily when available, else monthly
    fii_daily   = pd.to_numeric(df["FII_Daily_Net_Cr"],   errors="coerce")
    fii_monthly = pd.to_numeric(df["FII_Monthly_Net_Cr"], errors="coerce")
    dii_monthly = pd.to_numeric(df["DII_Monthly_Net_Cr"], errors="coerce")
    df["FII_Flow_Best"] = fii_daily.where(df["FII_Source_Flag"] == "daily", fii_monthly)

    # Beta_to_FII: use flow LEVEL as factor (not pct_change)
    # Monthly flow is a level signal — high positive = FII net buyers this month.
    # pct_change on monthly-stamped data is meaningless (same value 20+ days).
    df["Ret_FII"] = df["FII_Flow_Best"]

    # FII/DII momentum: sign of 5-day rolling sum
    # Cast to float — Int8 (nullable) corrupts through pandas merge to all-NULL
    df["FII_Momentum_5d"] = np.sign(
        df["FII_Flow_Best"].rolling(window=5, min_periods=3).sum()
    ).astype(float)
    df["DII_Momentum_5d"] = np.sign(
        dii_monthly.rolling(window=5, min_periods=3).sum()
    ).astype(float)

    # USDINR 1-day return — for USDINR_x_Revenue_Growth interaction
    df["USDINR_1d_Return"] = df["Ret_USDINR"]

    print(f"  ✅ Macro: {len(df):,} daily rows | {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def load_sentiment() -> pd.DataFrame:
    """Load nifty500_sentiment aggregate (one row per Ticker+Date)."""
    df = pd.read_sql(
        f"""
        SELECT Ticker, Date, Announcement_Score, News_Sentiment_Score,
               Sentiment_Score, Positive_Score, Negative_Score
        FROM {TABLES['sentiment']}
        """,
        con=engine,
    )
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
    print(f"  ✅ Sentiment: {len(df):,} rows | {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def compute_fundamental_ratios(fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annual-derived ratios from raw fundamentals.
    Input: nifty500_fundamentals rows
    Output: DataFrame with Ticker, Period, Effective_Date, + 8 ratio columns
    """
    df = fund_df.copy()
    df["Period"] = pd.to_datetime(df["Period"])
    df = df.sort_values(["Ticker", "Period"]).reset_index(drop=True)

    def safe_div(num, den):
        """Division guarded against zero/NaN denominators."""
        num = pd.to_numeric(num, errors="coerce")
        den = pd.to_numeric(den, errors="coerce")
        result = num / den
        result = result.replace([np.inf, -np.inf], np.nan)
        return result

    # Revenue YoY growth — within ticker, period-over-period
    df["Revenue_prev"] = df.groupby("Ticker")["Revenue"].shift(1)
    df["Revenue_YoY_Growth"] = safe_div(
        df["Revenue"] - df["Revenue_prev"], df["Revenue_prev"]
    )

    # Margins
    df["EBITDA_Margin"] = safe_div(df["EBITDA"],         df["Revenue"])
    df["Net_Margin"]    = safe_div(df["Net_Income"],     df["Revenue"])
    df["FCF_Margin"]    = safe_div(df["Free_Cash_Flow"], df["Revenue"])

    # Asset turnover
    df["Asset_Turnover"] = safe_div(df["Revenue"], df["Total_Assets"])

    # ROA, Debt_to_Equity, EPS_Basic sourced directly from screener_fundamentals.py

    # Effective date = Period + 60 days (lag for report availability)
    # Force ns precision — pd.Timedelta addition produces datetime64[s] in some
    # pandas versions, breaking merge_asof against datetime64[us] from MySQL.
    df["Effective_Date"] = (df["Period"] + pd.Timedelta(days=FUND_LAG_DAYS)).astype("datetime64[ns]")
    df["Period"]         = df["Period"].astype("datetime64[ns]")

    # Attach sector for median computation
    df["Sector"] = df["Ticker"].map(lambda t: get_sector(t))

    keep_cols = ["Ticker", "Period", "Effective_Date", "Sector"] + RATIO_COLS
    return df[keep_cols]


def compute_sector_medians(ratios_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per (Sector, Period) median of each ratio column.
    Only includes tickers with actual (non-NaN) values.
    Returns NaN if fewer than SECTOR_MED_MIN_TICKERS have actual data.
    """
    print("\n📊 Computing sector medians...")
    records = []

    grouped = ratios_df.groupby(["Sector", "Period"])
    for (sector, period), group in grouped:
        row = {
            "Sector": sector,
            "Period": period.date(),
            "Ticker_Count": len(group),
        }
        for col in RATIO_COLS:
            non_null = group[col].dropna()
            if len(non_null) >= SECTOR_MED_MIN_TICKERS:
                row[col] = float(non_null.median())
            else:
                row[col] = None
        records.append(row)

    med_df = pd.DataFrame(records)
    print(f"  ✅ {len(med_df):,} sector-period combinations")

    if not med_df.empty:
        save_to_db(med_df, TABLES["sector_median"], engine)

    return med_df


def build_sector_median_lookup(med_df: pd.DataFrame) -> dict:
    """
    Build nested dict {Sector: DataFrame of (Period, ratio_cols)}
    sorted by Period for fast merge_asof per ticker.
    """
    lookup = {}
    for sector, group in med_df.groupby("Sector"):
        g = group.copy()
        g["Period"] = pd.to_datetime(g["Period"]).astype("datetime64[ns]")
        g = g.sort_values("Period").reset_index(drop=True)
        for col in RATIO_COLS:
            g.rename(columns={col: f"SecMed_{col}"}, inplace=True)
        lookup[sector] = g[["Period"] + [f"SecMed_{c}" for c in RATIO_COLS]]
    return lookup


# ═══════════════════════════════════════════════════════════
# PHASE 2 — PER-TICKER FEATURE CONSTRUCTION
# ═══════════════════════════════════════════════════════════

def load_ticker_ohlcv(ticker: str) -> pd.DataFrame:
    """Load OHLCV rows for a single ticker, sorted by Date."""
    df = pd.read_sql(
        f"""
        SELECT Date, Ticker, Open, High, Low, Close, Adj_Close, Volume, VWAP_Daily
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
    """Load indicator rows for a single ticker, sorted by Date."""
    df = pd.read_sql(
        f"""
        SELECT Date, SMA_20, SMA_50, SMA_200, EMA_9, EMA_21,
               MACD, MACD_Signal, MACD_Hist, RSI_14,
               BB_Upper, BB_Middle, BB_Lower, ATR_14,
               Stoch_K, Stoch_D, ADX_14, OBV, VWAP_Dev
        FROM {TABLES['indicators']}
        WHERE Ticker = %s
        ORDER BY Date
        """,
        con=engine,
        params=(ticker,),
    )
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
    return df


def compute_price_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute returns, log returns, rolling vol, volume ratio, ADV.
    All computed on Adj_Close (split-adjusted).
    """
    df = df.copy()
    adj = pd.to_numeric(df["Adj_Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"],    errors="coerce")

    df["Return_1d"]  = adj.pct_change(1)
    df["Return_5d"]  = adj.pct_change(5)
    df["Return_21d"] = adj.pct_change(21)
    df["Return_60d"] = adj.pct_change(60)

    df["Log_Return_1d"] = np.log(adj / adj.shift(1))

    log_ret = df["Log_Return_1d"]
    df["Volatility_20d"] = (
        log_ret.rolling(window=20, min_periods=20).std(ddof=1)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    vol_mean_20 = vol.rolling(window=20, min_periods=20).mean()
    df["Volume_Ratio_20d"] = vol / vol_mean_20

    rupee_vol = vol * adj
    df["ADV_20d_Cr"] = (
        rupee_vol.rolling(window=20, min_periods=20).mean() / 1e7
    )

    return df


def compute_price_gap_flag(df: pd.DataFrame) -> pd.Series:
    """1 if prior row gap > PRICE_GAP_THRESHOLD calendar days, else 0."""
    date_diff = df["Date"].diff().dt.days.fillna(0)
    return (date_diff > PRICE_GAP_THRESHOLD).astype("int8")


def compute_targets(df: pd.DataFrame, max_date: pd.Timestamp) -> pd.DataFrame:
    """
    Compute forward-looking targets.
    Target_Return_21d: (Adj_Close[t+21] / Adj_Close[t]) - 1
    Target_Vol_5d:    annualized std of log_returns[t+1..t+5], ddof=1
    Both NULL where forward window extends past max_date in OHLCV.
    Rank-based targets computed in Phase 3.
    """
    df = df.copy()
    adj     = pd.to_numeric(df["Adj_Close"], errors="coerce")
    log_ret = df["Log_Return_1d"]

    future_adj = adj.shift(-TARGET_HORIZON)
    df["Target_Return_21d"] = (future_adj / adj) - 1

    forward_log_ret = log_ret.shift(-1)
    rolling_std = forward_log_ret.rolling(
        window=TARGET_VOL_WINDOW, min_periods=TARGET_VOL_WINDOW
    ).std(ddof=1)
    df["Target_Vol_5d"] = rolling_std.shift(-(TARGET_VOL_WINDOW - 1)) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return df


def merge_fundamentals(ticker_df: pd.DataFrame, ticker: str,
                       ratios_df: pd.DataFrame,
                       sector_lookup: dict) -> pd.DataFrame:
    """
    Attach fundamental ratios to ticker's daily rows using merge_asof.
    For each row's Date:
      - Find latest fundamental row where Effective_Date <= Date (within ticker)
      - If no match OR stale (>FUND_MAX_FFILL_DAYS) → use sector median
      - If sector median also NULL → leave NULL, Fundamentals_Available = 0
    """
    df = ticker_df.copy()

    tick_funds = ratios_df[ratios_df["Ticker"] == ticker].copy()
    tick_funds = tick_funds.sort_values("Effective_Date").reset_index(drop=True)

    for col in RATIO_COLS:
        df[col] = np.nan
    df["Fundamentals_Available"] = 0

    if not tick_funds.empty:
        df_sorted = (
            df.drop(columns=RATIO_COLS)
              .sort_values("Date")
              .reset_index()
        )

        df_sorted["Date"]              = df_sorted["Date"].astype("datetime64[ns]")
        tick_funds["Effective_Date"]   = tick_funds["Effective_Date"].astype("datetime64[ns]")
        tick_funds["Period"]           = tick_funds["Period"].astype("datetime64[ns]")

        merged = pd.merge_asof(
            df_sorted,
            tick_funds[["Effective_Date", "Period"] + RATIO_COLS],
            left_on="Date",
            right_on="Effective_Date",
            direction="backward",
            allow_exact_matches=True,
        )

        days_since_period = (merged["Date"] - merged["Period"]).dt.days
        stale_mask = (days_since_period > FUND_MAX_FFILL_DAYS) | days_since_period.isna()

        merged     = merged.set_index("index").sort_index()
        stale_mask.index = merged.index

        valid_mask = ~stale_mask & merged["Effective_Date"].notna()

        for col in RATIO_COLS:
            df.loc[valid_mask, col] = merged.loc[valid_mask, col].values

        df.loc[valid_mask, "Fundamentals_Available"] = 1

    sector = get_sector(ticker)
    needs_fallback_mask = df["Fundamentals_Available"] == 0

    if needs_fallback_mask.any() and sector in sector_lookup:
        sec_med = sector_lookup[sector]

        if not sec_med.empty:
            fallback_rows = df.loc[needs_fallback_mask, ["Date"]].copy()
            fallback_rows["Join_Date"] = (
                fallback_rows["Date"] - pd.Timedelta(days=FUND_LAG_DAYS)
            ).astype("datetime64[ns]")
            fallback_rows = fallback_rows.sort_values("Join_Date")
            fallback_rows["_orig_idx"] = fallback_rows.index

            sec_med = sec_med.copy()
            sec_med["Period"] = sec_med["Period"].astype("datetime64[ns]")

            fb_merged = pd.merge_asof(
                fallback_rows,
                sec_med,
                left_on="Join_Date",
                right_on="Period",
                direction="backward",
                allow_exact_matches=True,
            )

            fb_merged = fb_merged.set_index("_orig_idx").sort_index()

            for col in RATIO_COLS:
                secmed_col = f"SecMed_{col}"
                if secmed_col in fb_merged.columns:
                    fb_values = fb_merged[secmed_col]
                    valid_fb  = fb_values.notna()
                    idx       = fb_values.index[valid_fb]
                    df.loc[idx, col] = fb_values.loc[idx].values

    return df


def rolling_beta(stock_returns: pd.Series, factor_returns: pd.Series,
                 window: int = 252, min_periods: int = 63) -> pd.Series:
    """
    Rolling OLS beta of stock_returns on factor_returns.
    beta = cov(stock, factor) / var(factor)
    window     : 252 trading days (1 year)
    min_periods: 63 trading days (1 quarter) — NULL below this threshold
    """
    cov  = stock_returns.rolling(window=window, min_periods=min_periods).cov(factor_returns)
    var  = factor_returns.rolling(window=window, min_periods=min_periods).var()
    beta = cov / var.replace(0, np.nan)
    beta = beta.replace([np.inf, -np.inf], np.nan)
    return beta


def compute_macro_sensitivities(df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stock-specific macro sensitivities and interaction terms.
    Called after macro_df has been left-joined onto df.

    Beta columns: rolling 252d OLS beta (min 63d) of stock Return_1d
                  against each factor's daily return / level series.
    Interaction terms: stock's own fundamental × shared macro variable
                       (cross-sectionally valid — different value per stock).

    FIX: Column access uses direct df[col] with explicit existence guard
         instead of df.get(col, default) which silently returns None when
         the column exists but pandas .get() falls through on DataFrame.
    """
    stock_ret = pd.to_numeric(df["Return_1d"], errors="coerce")

    beta_pairs = [
        ("Beta_to_Nifty50",  "Ret_Nifty50"),
        ("Beta_to_Nifty500", "Ret_Nifty500"),
        ("Beta_to_USDINR",   "Ret_USDINR"),
        ("Beta_to_VIX",      "Ret_VIX"),
        ("Beta_to_Crude",    "Ret_Crude"),
        ("Beta_to_Gold",     "Ret_Gold"),
        ("Beta_to_FII",      "Ret_FII"),
    ]

    for beta_col, factor_col in beta_pairs:
        if factor_col in df.columns:
            factor = pd.to_numeric(df[factor_col], errors="coerce")
            df[beta_col] = rolling_beta(stock_ret, factor)
        else:
            df[beta_col] = np.nan

    # Interaction terms
    nan_series = pd.Series(np.nan, index=df.index)

    repo   = pd.to_numeric(df["Repo_Rate"],         errors="coerce") if "Repo_Rate"         in df.columns else nan_series
    d_e    = pd.to_numeric(df["Debt_to_Equity"],     errors="coerce") if "Debt_to_Equity"     in df.columns else nan_series
    usdinr = pd.to_numeric(df["USDINR_1d_Return"],   errors="coerce") if "USDINR_1d_Return"   in df.columns else nan_series
    rev_g  = pd.to_numeric(df["Revenue_YoY_Growth"], errors="coerce") if "Revenue_YoY_Growth" in df.columns else nan_series

    df["RepoRate_x_DebtEquity"]   = repo  * d_e
    df["USDINR_x_Revenue_Growth"] = usdinr * rev_g

    return df


def process_ticker(ticker: str,
                   data_tier: int,
                   ratios_df: pd.DataFrame,
                   sector_lookup: dict,
                   macro_df: pd.DataFrame,
                   sentiment_df: pd.DataFrame,
                   dataset_max_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build full feature row set for one ticker.
    Returns DataFrame ready for features_master insertion (minus rank cols).
    """
    ohlcv = load_ticker_ohlcv(ticker)
    if ohlcv.empty:
        return pd.DataFrame()

    indic = load_ticker_indicators(ticker)

    df = ohlcv.merge(indic, on="Date", how="left")

    df = compute_price_derived(df)

    bb_upper = pd.to_numeric(df["BB_Upper"],  errors="coerce")
    bb_lower = pd.to_numeric(df["BB_Lower"],  errors="coerce")
    bb_mid   = pd.to_numeric(df["BB_Middle"], errors="coerce")
    df["BB_Width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    df["Price_Gap_Flag"]          = compute_price_gap_flag(df)
    df["SMA200_Available"]        = df["SMA_200"].notna().astype("int8")
    df["SMA50_Available"]         = df["SMA_50"].notna().astype("int8")
    df["ADX14_Available"]         = df["ADX_14"].notna().astype("int8")
    df["Volatility20d_Available"] = df["Volatility_20d"].notna().astype("int8")

    df = compute_targets(df, dataset_max_date)

    df = merge_fundamentals(df, ticker, ratios_df, sector_lookup)

    # Macro join — brings in India_VIX, USDINR, Crude_Oil, Gold, Repo_Rate,
    # and all derived columns (Ret_*, FII_Momentum_5d, USDINR_1d_Return etc.)
    df = df.merge(macro_df, on="Date", how="left")

    # Stock-specific macro sensitivities + interaction terms
    df = compute_macro_sensitivities(df, macro_df)

    tick_sent = sentiment_df[sentiment_df["Ticker"] == ticker].drop(columns=["Ticker"])
    df = df.merge(tick_sent, on="Date", how="left")

    sent_start = pd.Timestamp(SENTIMENT_START_DATE)
    df["Sentiment_Available"] = (
        (df["Date"] >= sent_start) & df["Announcement_Score"].notna()
    ).astype("int8")

    df["Data_Tier"] = data_tier

    df["Target_Rank_21d"]          = np.nan
    df["Target_Direction_Median"]  = pd.NA
    df["Target_Direction_Tertile"] = pd.NA

    df["Ticker"] = ticker

    for col in FEATURES_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[FEATURES_COLUMNS].copy()

    # Convert nullable int targets for MySQL
    for col in ["Target_Direction_Median", "Target_Direction_Tertile"]:
        df[col] = df[col].astype("object").where(df[col].notna(), None)

    # FII_Momentum_5d / DII_Momentum_5d: replace NaN with None for MySQL TINYINT NULL
    for col in ["FII_Momentum_5d", "DII_Momentum_5d"]:
        df[col] = df[col].where(df[col].notna(), None)

    df["Date"] = df["Date"].dt.date

    return df


# ═══════════════════════════════════════════════════════════
# PHASE 3 — CROSS-SECTIONAL RANK PASS
# ═══════════════════════════════════════════════════════════

def compute_and_write_ranks():
    """
    Phase 3: cross-sectional percentile rank of Target_Return_21d
    computed across Tier A+B (Data_Tier IN (1,2)) per Date.
    Writes Target_Rank_21d, Target_Direction_Median, Target_Direction_Tertile
    back to features_master via temp-table JOIN UPDATE.
    Tier C rows remain NULL.
    """
    print("\n🎯 Phase 3 — cross-sectional rank pass...")

    df = pd.read_sql(
        f"""
        SELECT Ticker, Date, Target_Return_21d
        FROM {TABLES['features']}
        WHERE Data_Tier IN (1, 2)
          AND Target_Return_21d IS NOT NULL
        """,
        con=engine,
    )
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"  📥 Loaded {len(df):,} Tier A+B rows with forward return")

    if df.empty:
        print("  ⚠️  No rows to rank — skipping")
        return

    df["Target_Rank_21d"] = (
        df.groupby("Date")["Target_Return_21d"]
        .rank(method="average", pct=True)
    )

    df["Target_Direction_Median"] = (df["Target_Rank_21d"] > 0.5).astype("int8")

    def tertile(r):
        if pd.isna(r):
            return None
        if r > 2 / 3:
            return 1
        if r < 1 / 3:
            return 0
        return None

    df["Target_Direction_Tertile"] = df["Target_Rank_21d"].map(tertile)
    df["Target_Rank_21d"]          = df["Target_Rank_21d"].round(6)
    df["Date"]                     = df["Date"].dt.date

    upload_df = df[[
        "Ticker", "Date",
        "Target_Rank_21d", "Target_Direction_Median", "Target_Direction_Tertile"
    ]].copy()
    upload_df["Target_Direction_Median"]  = upload_df["Target_Direction_Median"].astype("Int8")
    upload_df["Target_Direction_Tertile"] = upload_df["Target_Direction_Tertile"].astype("Int8")

    print("  📤 Writing ranks to _tmp_ranks...")
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS _tmp_ranks"))
        conn.execute(text("""
            CREATE TABLE _tmp_ranks (
                Ticker                   VARCHAR(20) NOT NULL,
                Date                     DATE NOT NULL,
                Target_Rank_21d          DECIMAL(10,6),
                Target_Direction_Median  TINYINT,
                Target_Direction_Tertile TINYINT,
                PRIMARY KEY (Ticker, Date)
            )
        """))
        conn.commit()

    upload_df.to_sql(
        name="_tmp_ranks", con=engine, if_exists="append",
        index=False, chunksize=10000,
    )
    print(f"  ✅ Wrote {len(upload_df):,} rank rows to _tmp_ranks")

    print("  🔄 Applying ranks to features_master via JOIN UPDATE...")
    with engine.connect() as conn:
        conn.execute(text(f"""
            UPDATE {TABLES['features']} f
            JOIN _tmp_ranks t
              ON f.Ticker = t.Ticker AND f.Date = t.Date
            SET f.Target_Rank_21d          = t.Target_Rank_21d,
                f.Target_Direction_Median  = t.Target_Direction_Median,
                f.Target_Direction_Tertile = t.Target_Direction_Tertile
        """))
        conn.commit()
        conn.execute(text("DROP TABLE IF EXISTS _tmp_ranks"))
        conn.commit()

    print("  ✅ Ranks applied; temp table dropped")


# ═══════════════════════════════════════════════════════════
# PHASE 4 — VERIFICATION
# ═══════════════════════════════════════════════════════════

def print_verification():
    """Summary statistics after full run."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    with engine.connect() as conn:
        total = conn.execute(text(f"SELECT COUNT(*) FROM {TABLES['features']}")).scalar()
        print(f"\nTotal rows: {total:,}")

        print("\nRows per tier:")
        for row in conn.execute(text(f"""
            SELECT Data_Tier, COUNT(*) AS n, COUNT(DISTINCT Ticker) AS tickers
            FROM {TABLES['features']}
            GROUP BY Data_Tier ORDER BY Data_Tier
        """)):
            tier_name = {1: "A", 2: "B", 3: "C"}.get(row[0], str(row[0]))
            print(f"  Tier {tier_name}: {row[1]:>10,} rows | {row[2]:>4} tickers")

        dates = conn.execute(text(f"""
            SELECT MIN(Date), MAX(Date), COUNT(DISTINCT Date)
            FROM {TABLES['features']}
        """)).fetchone()
        print(f"\nDate range: {dates[0]} → {dates[1]} ({dates[2]:,} unique dates)")

        targets = conn.execute(text(f"""
            SELECT
                SUM(CASE WHEN Target_Return_21d        IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN Target_Rank_21d          IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN Target_Direction_Median  IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN Target_Direction_Tertile IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN Target_Vol_5d            IS NOT NULL THEN 1 ELSE 0 END)
            FROM {TABLES['features']}
        """)).fetchone()
        print("\nTarget coverage (non-NULL counts):")
        print(f"  Target_Return_21d       : {targets[0]:>12,}")
        print(f"  Target_Rank_21d         : {targets[1]:>12,}")
        print(f"  Target_Direction_Median : {targets[2]:>12,}")
        print(f"  Target_Direction_Tertile: {targets[3]:>12,}")
        print(f"  Target_Vol_5d           : {targets[4]:>12,}")

        flags = conn.execute(text(f"""
            SELECT
                SUM(Price_Gap_Flag),
                SUM(SMA200_Available),
                SUM(SMA50_Available),
                SUM(ADX14_Available),
                SUM(Volatility20d_Available),
                SUM(Sentiment_Available),
                SUM(Fundamentals_Available)
            FROM {TABLES['features']}
        """)).fetchone()
        print("\nFlag distributions (rows with flag=1):")
        print(f"  Price_Gap_Flag         : {flags[0]:>12,}")
        print(f"  SMA200_Available       : {flags[1]:>12,}")
        print(f"  SMA50_Available        : {flags[2]:>12,}")
        print(f"  ADX14_Available        : {flags[3]:>12,}")
        print(f"  Volatility20d_Available: {flags[4]:>12,}")
        print(f"  Sentiment_Available    : {flags[5]:>12,}")
        print(f"  Fundamentals_Available : {flags[6]:>12,}")

        new_cols = conn.execute(text(f"""
            SELECT
                SUM(CASE WHEN Beta_to_Nifty50      IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN Beta_to_FII          IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN FII_Momentum_5d      IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN DII_Momentum_5d      IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN RepoRate_x_DebtEquity IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN USDINR_x_Revenue_Growth IS NOT NULL THEN 1 ELSE 0 END)
            FROM {TABLES['features']}
        """)).fetchone()
        print("\nNew column coverage (non-NULL counts):")
        print(f"  Beta_to_Nifty50         : {new_cols[0]:>12,}")
        print(f"  Beta_to_FII             : {new_cols[1]:>12,}")
        print(f"  FII_Momentum_5d         : {new_cols[2]:>12,}")
        print(f"  DII_Momentum_5d         : {new_cols[3]:>12,}")
        print(f"  RepoRate_x_DebtEquity   : {new_cols[4]:>12,}")
        print(f"  USDINR_x_Revenue_Growth : {new_cols[5]:>12,}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("FEATURES PIPELINE — hedge_v2")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not validate_inputs():
        print("\n❌ Input validation failed. Exiting.")
        return

    truncate_output_tables()

    print("\n📥 Loading tier classification...")
    tier_map = load_tier_map()
    if not tier_map:
        print("❌ No tickers in stock_data_quality. Exiting.")
        return

    print("\n📥 Phase 1 — pre-computation...")
    macro_df     = load_macro()
    sentiment_df = load_sentiment()

    print("\n📊 Computing fundamental ratios...")
    raw_fund = pd.read_sql(
        f"""
        SELECT Ticker, Period, Revenue, EBITDA, Net_Income,
               Free_Cash_Flow, Total_Assets, ROA,
               Debt_to_Equity, EPS_Basic
        FROM {TABLES['fundamentals']}
        """,
        con=engine,
    )
    print(f"  Loaded {len(raw_fund):,} fundamental rows")

    ratios_df = compute_fundamental_ratios(raw_fund)
    print(f"  ✅ Computed ratios for {len(ratios_df):,} (Ticker, Period) pairs")

    med_df        = compute_sector_medians(ratios_df)
    sector_lookup = build_sector_median_lookup(med_df)

    dataset_max_date = pd.to_datetime(
        pd.read_sql(f"SELECT MAX(Date) AS d FROM {TABLES['ohlcv']}", con=engine)["d"].iloc[0]
    )
    print(f"\n📅 Dataset max date: {dataset_max_date.date()}")

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
                macro_df, sentiment_df, dataset_max_date,
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

    compute_and_write_ranks()

    print_verification()

    print("\n" + "=" * 60)
    print(f"DONE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\n📋 Next: python data/export_features.py")


if __name__ == "__main__":
    main()