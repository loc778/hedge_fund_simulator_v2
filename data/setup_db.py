# data/setup_db.py
# ═══════════════════════════════════════════════════════════
# ONE-TIME DATABASE SETUP — hedge_v2.3
# Run once on any new machine to create all tables.
# IF NOT EXISTS means re-running never breaks anything.
#
# CHANGES (Apr 2026) — v2.3 (features.py v4 alignment — IC fix):
#   features_master schema updated with 9 new columns (Fix A + Fix B):
#
#   Fix A — Cross-sectional momentum rank features:
#     Mom_12_1              : 252d return minus 21d return (raw momentum signal)
#     Mom_6_1               : 126d return minus 21d return (raw momentum signal)
#     Mom_1M_Rank           : cross-sectional percentile rank of Return_21d per date
#     Mom_12_1_Rank         : cross-sectional percentile rank of Mom_12_1 per date
#     Mom_6_1_Rank          : cross-sectional percentile rank of Mom_6_1 per date
#     RS_21d_Rank           : cross-sectional percentile rank of RS_21d per date
#   Rationale: cross-sectional momentum consistently produces IC 0.04–0.07 in
#   Indian equities. Raw returns are not cross-sectional signals; their rank
#   on each date is. The existing RS_21d is computed per-ticker; its rank
#   across all tickers on a date is what the model needs.
#
#   Fix B — Within-sector fundamental rank features:
#     ROA_Sector_Rank              : percentile rank of ROA within sector per date
#     EBITDA_Margin_Sector_Rank    : percentile rank of EBITDA_Margin within sector
#     Revenue_Growth_Sector_Rank   : percentile rank of Revenue_YoY_Growth within sector
#   Rationale: annual fundamental ratios forward-filled for up to 400 days are
#   essentially static labels — ROA of 0.12 means nothing without knowing if
#   that is top-quartile or bottom-quartile in the sector at that date. Sector
#   rank converts a static snapshot into a cross-sectionally meaningful signal.
#
# CHANGES (Apr 2026) — v2.2 (features.py v3 alignment):
#   features_master schema rewritten to match FEATURES_COLUMNS exactly:
#     REMOVED (non-stationary / redundant / dropped in v2.2):
#       * Raw price/volume levels: Open, High, Low, Close, Adj_Close,
#         Volume, VWAP_Daily, ADV_20d_Cr
#       * Raw indicator levels: SMA_20, SMA_50, SMA_200, EMA_9, EMA_21,
#         MACD, MACD_Signal, BB_Upper, BB_Middle, BB_Lower, Stoch_D, OBV
#       * Fundamental: EPS_Basic (non-comparable across tickers)
#       * Sentiment: News_Sentiment_Score, Sentiment_Score, Positive_Score,
#         Negative_Score (correlated; replaced by Sentiment_Score_Lag1)
#       * Macro levels: India_VIX, USDINR, Crude_Oil, Gold
#         (cross-sectionally flat; betas are superior)
#       * Macro signals: FII_Momentum_5d, DII_Momentum_5d
#         (low-cardinality ±1; betas are superior)
#       * Interaction: RepoRate_x_DebtEquity (level × level, cross-sec flat)
#     ADDED (v2.2 new features):
#       * Price: RS_21d, RS_60d (relative strength vs Nifty500)
#       * Technical: Price_to_SMA20/50/200, Price_to_52W_High/Low,
#         BB_PctB, OBV_Change_5d
#       * Fundamental: Gross_Profit_Margin, Gross_Margin_Is_Proxy,
#         Debt_to_Assets, OCF_to_Net_Income, Delta_ROA, Delta_DebtEquity,
#         Rel_ROA, Rel_EBITDA_Margin
#       * Sentiment: Sentiment_Score_Lag1 (t-1 lagged, leakage-free)
#       * Macro: Regime_Int, Beta_to_DII, Repo_Rate_Change,
#         USDINR_x_Revenue_Growth
#     PRECISION FIX:
#       * BB_Width: DECIMAL(10,4) → DECIMAL(10,6) (ratio column)
#
#   sector_fundamentals_median schema updated to match RATIO_COLS:
#     REMOVED: EPS_Basic (not in RATIO_COLS, not aggregated)
#     ADDED: Gross_Profit_Margin, Debt_to_Assets, OCF_to_Net_Income,
#            Delta_ROA, Delta_DebtEquity
#     NOTE: Gross_Margin_Is_Proxy intentionally excluded (flag, not ratio)
#
# BEFORE RE-RUNNING on an existing database (one-time manual SQL):
#   -- v2.3 migration: add 9 new columns to existing features_master
#   ALTER TABLE features_master
#       ADD COLUMN Mom_12_1                  DECIMAL(12,6) AFTER RS_60d,
#       ADD COLUMN Mom_6_1                   DECIMAL(12,6) AFTER Mom_12_1,
#       ADD COLUMN Mom_1M_Rank               DECIMAL(10,6) AFTER Mom_6_1,
#       ADD COLUMN Mom_12_1_Rank             DECIMAL(10,6) AFTER Mom_1M_Rank,
#       ADD COLUMN Mom_6_1_Rank              DECIMAL(10,6) AFTER Mom_12_1_Rank,
#       ADD COLUMN RS_21d_Rank               DECIMAL(10,6) AFTER Mom_6_1_Rank,
#       ADD COLUMN ROA_Sector_Rank           DECIMAL(10,6) AFTER Rel_EBITDA_Margin,
#       ADD COLUMN EBITDA_Margin_Sector_Rank DECIMAL(10,6) AFTER ROA_Sector_Rank,
#       ADD COLUMN Revenue_Growth_Sector_Rank DECIMAL(10,6) AFTER EBITDA_Margin_Sector_Rank;
#   -- OR on a fresh system just run: python data/setup_db.py
#   -- Then rebuild: python data/features.py
#
#   -- v2.2 migration (if upgrading from v2.1):
#   DROP TABLE IF EXISTS features_master;
#   DROP TABLE IF EXISTS sector_fundamentals_median;
#   ALTER TABLE nifty500_fundamentals
#       MODIFY COLUMN EPS_Basic   DECIMAL(12,4),
#       MODIFY COLUMN EPS_Diluted DECIMAL(12,4);
# ═══════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine
from sqlalchemy import text

TABLES = {

    # ── Layer 1: Price & Volume (Bhavcopy) ───────────────────────────
    "nifty500_ohlcv": """
        CREATE TABLE IF NOT EXISTS nifty500_ohlcv (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            Date            DATE            NOT NULL,
            Ticker          VARCHAR(20)     NOT NULL,
            Open            DECIMAL(10,4),
            High            DECIMAL(10,4),
            Low             DECIMAL(10,4),
            Close           DECIMAL(10,4),
            Adj_Close       DECIMAL(10,4),
            Volume          BIGINT,
            Typical_Price   DECIMAL(10,4),
            VWAP_Daily      DECIMAL(10,4),
            UNIQUE KEY unique_ticker_date (Ticker, Date)
        )
    """,

    # ── Corporate Actions ─────────────────────────────────────────────
    "corporate_actions": """
        CREATE TABLE IF NOT EXISTS corporate_actions (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            Ticker          VARCHAR(20)     NOT NULL,
            Date            DATE            NOT NULL,
            Action_Type     VARCHAR(20)     NOT NULL,
            Ratio           DECIMAL(10,6),
            Amount          DECIMAL(10,4),
            UNIQUE KEY unique_ticker_date_action (Ticker, Date, Action_Type)
        )
    """,

    # ── Layer 4: Technical Indicators ────────────────────────────────
    # Raw indicator levels are stored here in full — features.py reads
    # these to compute stationary derived features (Price_to_SMA*, BB_PctB,
    # OBV_Change_5d etc.) and then stores only the derived versions in
    # features_master. Raw levels intentionally kept here for auditability.
    "nifty500_indicators": """
        CREATE TABLE IF NOT EXISTS nifty500_indicators (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            Date            DATE            NOT NULL,
            Ticker          VARCHAR(20)     NOT NULL,
            SMA_20          DECIMAL(10,4),
            SMA_50          DECIMAL(10,4),
            SMA_200         DECIMAL(10,4),
            EMA_9           DECIMAL(10,4),
            EMA_21          DECIMAL(10,4),
            MACD            DECIMAL(10,4),
            MACD_Signal     DECIMAL(10,4),
            MACD_Hist       DECIMAL(10,4),
            RSI_14          DECIMAL(10,4),
            BB_Upper        DECIMAL(10,4),
            BB_Middle       DECIMAL(10,4),
            BB_Lower        DECIMAL(10,4),
            ATR_14          DECIMAL(10,4),
            Stoch_K         DECIMAL(10,4),
            Stoch_D         DECIMAL(10,4),
            ADX_14          DECIMAL(10,4),
            OBV             DECIMAL(20,4),
            VWAP_Dev        DECIMAL(10,4),
            UNIQUE KEY unique_ticker_date (Ticker, Date)
        )
    """,

    # ── Layer 2: Fundamental Data ─────────────────────────────────────
    # EPS_Basic / EPS_Diluted widened to DECIMAL(12,4) — high-EPS stocks
    # (MRF, PAGEIND, NESTLEIND) exceed the previous DECIMAL(10,4) max.
    # Raw annual source data. features.py computes all derived ratios
    # (margins, growth, delta columns) from these inputs; EPS_Basic itself
    # is not forwarded to features_master (non-comparable across tickers).
    "nifty500_fundamentals": """
        CREATE TABLE IF NOT EXISTS nifty500_fundamentals (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            Ticker              VARCHAR(20)     NOT NULL,
            Period              DATE            NOT NULL,
            Revenue             BIGINT,
            Gross_Profit        BIGINT,
            EBITDA              BIGINT,
            Net_Income          BIGINT,
            EPS_Basic           DECIMAL(12,4),
            EPS_Diluted         DECIMAL(12,4),
            Total_Assets        BIGINT,
            Total_Liabilities   BIGINT,
            Total_Equity        BIGINT,
            Cash                BIGINT,
            Total_Debt          BIGINT,
            Book_Value_PS       DECIMAL(10,4),
            Operating_CF        BIGINT,
            Capex               BIGINT,
            Free_Cash_Flow      BIGINT,
            Debt_to_Equity      DECIMAL(10,4),
            FCF_Yield           DECIMAL(10,4),
            ROCE                DECIMAL(10,4),
            PE_Ratio            DECIMAL(10,4),
            PB_Ratio            DECIMAL(10,4),
            EV_EBITDA           DECIMAL(10,4),
            Dividend_Yield      DECIMAL(10,4),
            ROE                 DECIMAL(10,4),
            ROA                 DECIMAL(10,4),
            FII_Holding         DECIMAL(10,4),
            DII_Holding         DECIMAL(10,4),
            UNIQUE KEY unique_ticker_period (Ticker, Period)
        )
    """,

    # ── Layer 3A: Macro & Economic Indicators ─────────────────────────
    # FII/DII columns — all written by fii_dii_stockedge.py:
    #   FII_Monthly_Net_Cr = monthly net flow stamped on every trading day (2010–present)
    #   DII_Monthly_Net_Cr = monthly net flow stamped on every trading day (2010–present)
    #   FII_Daily_Net_Cr   = actual daily value for last ~50 trading days
    #   DII_Daily_Net_Cr   = actual daily value for last ~50 trading days
    #   FII_Source_Flag    = 'monthly' or 'daily' — tells features.py data resolution
    # Raw macro levels stored here in full. features.py computes return
    # series (pct_change / diff) and rolling betas from these; raw levels
    # are not forwarded to features_master (cross-sectionally flat).
    "macro_indicators": """
        CREATE TABLE IF NOT EXISTS macro_indicators (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            Date                DATE        NOT NULL UNIQUE,
            India_VIX           DECIMAL(10,4),
            USDINR              DECIMAL(10,4),
            Crude_Oil           DECIMAL(10,4),
            Gold                DECIMAL(10,4),
            Nifty50_Close       DECIMAL(10,4),
            Nifty500_Close      DECIMAL(10,4),
            CPI_India           DECIMAL(10,4),
            GDP_India           DECIMAL(20,4),
            Fed_Funds_Rate      DECIMAL(10,4),
            US_CPI              DECIMAL(10,4),
            US_10Y_Bond         DECIMAL(10,4),
            Repo_Rate           DECIMAL(10,4),
            IIP_Growth          DECIMAL(10,4),
            Forex_Reserves_USD  DECIMAL(20,4),
            FII_Monthly_Net_Cr  DECIMAL(20,2),
            DII_Monthly_Net_Cr  DECIMAL(20,2),
            FII_Daily_Net_Cr    DECIMAL(20,2),
            DII_Daily_Net_Cr    DECIMAL(20,2),
            FII_Source_Flag     VARCHAR(10)
        )
    """,

    # ── Layer 3B: Sentiment Data — Daily Aggregate ────────────────────
    # NEVER TRUNCATE — historical news cannot be recovered.
    # Populated by sentiment.py --mode aggregate from nifty500_sentiment_raw.
    # One row per (Ticker, Date).
    # features.py reads Announcement_Score and Sentiment_Score from here:
    #   - Announcement_Score → stored same-day (point-in-time, EOD assumption)
    #   - Sentiment_Score    → shifted 1 day to become Sentiment_Score_Lag1
    # All other columns (News_Sentiment_Score, Positive_Score, Negative_Score)
    # are retained here for raw storage but not forwarded to features_master.
    "nifty500_sentiment": """
        CREATE TABLE IF NOT EXISTS nifty500_sentiment (
            id                    INT AUTO_INCREMENT PRIMARY KEY,
            Date                  DATE            NOT NULL,
            Ticker                VARCHAR(20)     NOT NULL,
            Announcement_Score    DECIMAL(10,4),
            News_Sentiment_Score  DECIMAL(10,4),
            Sentiment_Score       DECIMAL(10,4),
            Positive_Score        DECIMAL(10,4),
            Negative_Score        DECIMAL(10,4),
            Neutral_Score         DECIMAL(10,4),
            Events_Count          INT DEFAULT 0,
            Headlines_Count       INT DEFAULT 0,
            Has_Announcement      TINYINT DEFAULT 0,
            Has_News              TINYINT DEFAULT 0,
            Last_Updated          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                          ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY unique_ticker_date (Ticker, Date)
        )
    """,

    # ── Layer 3B: Sentiment Raw Events ────────────────────────────────
    # Per-event rows. One row per (Ticker, Date, Source, Headline).
    # NSE announcements: rule-scored at ingest time (Rule_Score populated).
    # Moneycontrol RSS: raw text stored, Rule_Score NULL.
    # FinBERT_Score/FinBERT_Label: populated by separate Colab notebook.
    # NEVER TRUNCATE.
    "nifty500_sentiment_raw": """
        CREATE TABLE IF NOT EXISTS nifty500_sentiment_raw (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            Date            DATE            NOT NULL,
            Ticker          VARCHAR(20)     NOT NULL,
            Source          VARCHAR(30)     NOT NULL,
            Event_Category  VARCHAR(50),
            Headline        TEXT,
            Raw_Text        TEXT,
            Rule_Score      DECIMAL(5,2),
            FinBERT_Score   DECIMAL(5,4),
            FinBERT_Label   VARCHAR(20),
            URL             VARCHAR(500),
            Ingested_At     TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY unique_event (Ticker, Date, Source, Headline(200))
        )
    """,

    # ── Stock Data Quality Classification ─────────────────────────────
    "stock_data_quality": """
        CREATE TABLE IF NOT EXISTS stock_data_quality (
            Ticker                VARCHAR(20)     NOT NULL PRIMARY KEY,
            Trading_Days_Total    INT,
            Data_Start_Date       DATE,
            Coverage_Years        DECIMAL(5,2),
            Gap_Pct               DECIMAL(5,2),
            Avg_Daily_Volume_Cr   DECIMAL(10,4),
            F_and_O_Listed        TINYINT,
            Data_Tier             CHAR(1),
            Tier_Assigned_Date    DATE
        )
    """,

    # ── Feature Engineering: Unified ML Training Dataset ─────────────
    # v2.3 schema (Apr 2026) — adds 9 cross-sectional momentum + fundamental
    # rank columns (Fix A + Fix B) to address low IC root cause.
    # Column count: 76 (excl. id, Date, Ticker)
    # Normalization: NOT done here — applied per walk-forward fold in Colab.
    # NULLs: flow through intentionally — XGBoost handles natively;
    #        availability flags carry structural missingness information.
    #
    # FIX A — Cross-sectional momentum ranks (computed in Phase 3.5):
    #   Mom_12_1, Mom_6_1: raw momentum signals (per-ticker, computed in Phase 2)
    #   Mom_1M_Rank, Mom_12_1_Rank, Mom_6_1_Rank, RS_21d_Rank:
    #     cross-sectional percentile ranks per date (computed in Phase 3.5)
    #
    # FIX B — Within-sector fundamental ranks (computed in Phase 3.5):
    #   ROA_Sector_Rank, EBITDA_Margin_Sector_Rank, Revenue_Growth_Sector_Rank:
    #     percentile rank within sector per date (computed in Phase 3.5)
    #     Uses Sector from stock_data_quality / config.get_sector().
    "features_master": """
        CREATE TABLE IF NOT EXISTS features_master (
            id                          INT AUTO_INCREMENT PRIMARY KEY,
            Date                        DATE            NOT NULL,
            Ticker                      VARCHAR(20)     NOT NULL,

            -- ── Returns & volatility (stationary) ────────────────────
            Return_1d                   DECIMAL(10,6),
            Log_Return_1d               DECIMAL(10,6),
            Return_5d                   DECIMAL(10,6),
            Return_21d                  DECIMAL(10,6),
            Return_60d                  DECIMAL(10,6),
            Volatility_20d              DECIMAL(10,6),
            Volume_Ratio_20d            DECIMAL(10,6),

            -- ── Relative strength vs Nifty500 ────────────────────────
            -- RS_21d = Return_21d - Nifty500_Return_21d
            -- RS_60d = Return_60d - Nifty500_Return_60d
            RS_21d                      DECIMAL(10,6),
            RS_60d                      DECIMAL(10,6),

            -- ── Cross-sectional momentum signals (Fix A) ─────────────
            -- Mom_12_1 / Mom_6_1: computed per-ticker in Phase 2.
            -- *_Rank columns: cross-sectional percentile per date,
            --   computed in Phase 3.5 across all Tier A+B tickers.
            -- NULL for Tier C, and for tickers with insufficient price
            --   history (< 252 days for Mom_12_1, < 126 days for Mom_6_1).
            -- These are the primary IC-boosting features — momentum rank
            --   consistently outperforms raw returns in cross-sectional models.
            Mom_12_1                    DECIMAL(12,6),   -- Return_252d - Return_21d
            Mom_6_1                     DECIMAL(12,6),   -- Return_126d - Return_21d
            Mom_1M_Rank                 DECIMAL(10,6),   -- CS rank of Return_21d per date
            Mom_12_1_Rank               DECIMAL(10,6),   -- CS rank of Mom_12_1 per date
            Mom_6_1_Rank                DECIMAL(10,6),   -- CS rank of Mom_6_1 per date
            RS_21d_Rank                 DECIMAL(10,6),   -- CS rank of RS_21d per date

            -- ── Technical: stationary price-position ratios ───────────
            -- Price_to_SMA* = Adj_Close / SMA_N (ratio, not level)
            -- Price_to_52W_* = Adj_Close / rolling 252d High/Low
            Price_to_SMA20              DECIMAL(10,6),
            Price_to_SMA50              DECIMAL(10,6),
            Price_to_SMA200             DECIMAL(10,6),
            Price_to_52W_High           DECIMAL(10,6),
            Price_to_52W_Low            DECIMAL(10,6),

            -- ── Technical indicators (all stationary) ─────────────────
            MACD_Hist                   DECIMAL(10,6),
            RSI_14                      DECIMAL(10,4),
            BB_Width                    DECIMAL(10,6),  -- (BB_Upper-BB_Lower)/BB_Middle
            BB_PctB                     DECIMAL(10,6),  -- (Close-BB_Lower)/(BB_Upper-BB_Lower)
            ATR_14                      DECIMAL(10,4),
            Stoch_K                     DECIMAL(10,4),
            ADX_14                      DECIMAL(10,4),
            OBV_Change_5d               DECIMAL(10,6),  -- OBV.pct_change(5)
            VWAP_Dev                    DECIMAL(10,6),

            -- ── Fundamental ratios (computed from annual data) ─────────
            -- Availability controlled by Effective_Date = Period + 60 days.
            -- Stale (>400d since period end) → NULL; sector median used instead.
            -- Normalization done in Colab per walk-forward fold, not here.
            Revenue_YoY_Growth          DECIMAL(12,6),
            EBITDA_Margin               DECIMAL(12,6),
            Net_Margin                  DECIMAL(12,6),
            FCF_Margin                  DECIMAL(12,6),
            Asset_Turnover              DECIMAL(12,6),
            ROA                         DECIMAL(12,6),
            Debt_to_Equity              DECIMAL(12,6),
            Gross_Profit_Margin         DECIMAL(12,6),  -- Net_Income/Revenue for banking
            Gross_Margin_Is_Proxy       TINYINT DEFAULT 0, -- 1 = banking sector fallback used
            Debt_to_Assets              DECIMAL(12,6),
            OCF_to_Net_Income           DECIMAL(12,6),  -- earnings quality
            Delta_ROA                   DECIMAL(12,6),  -- ROA YoY direction
            Delta_DebtEquity            DECIMAL(12,6),  -- Debt_to_Equity YoY direction

            -- ── Sector-relative fundamentals ──────────────────────────
            -- Rel_* = stock value - sector median for matched period.
            -- 0 for rows using sector-median fallback (by construction).
            Rel_ROA                     DECIMAL(12,6),
            Rel_EBITDA_Margin           DECIMAL(12,6),

            -- ── Within-sector fundamental rank features (Fix B) ───────
            -- Percentile rank of each fundamental metric within the stock's
            -- sector on each date. Computed in Phase 3.5 across all tickers
            -- (Tier A+B+C) that have non-NULL values for that metric on that date.
            -- Converts static annual snapshots into cross-sectionally meaningful
            -- quality factor signals — a ROA of 0.12 ranked 90th pct in Financials
            -- is a very different signal from 90th pct in Pharma.
            -- NULL when fewer than 3 tickers in sector have the metric that date.
            ROA_Sector_Rank             DECIMAL(10,6),  -- within-sector ROA rank
            EBITDA_Margin_Sector_Rank   DECIMAL(10,6),  -- within-sector EBITDA_Margin rank
            Revenue_Growth_Sector_Rank  DECIMAL(10,6),  -- within-sector Revenue_YoY_Growth rank

            -- ── Sentiment (leakage-controlled) ────────────────────────
            -- Announcement_Score: NSE event score, kept same-day.
            --   TIMING ASSUMPTION: valid for EOD/next-open predictions only.
            --   If used in pre-open prediction, apply shift(-1) instead.
            -- Sentiment_Score_Lag1: composite score shifted 1 trading day
            --   back within each ticker's own timeline — leakage-free.
            Announcement_Score          DECIMAL(10,4),
            Sentiment_Score_Lag1        DECIMAL(10,4),

            -- ── Market regime ─────────────────────────────────────────
            -- 0=Bear, 1=Bull, 2=HighVol, 3=Sideways (from market_regimes)
            Regime_Int                  TINYINT,

            -- ── Stock-specific macro sensitivities (rolling 252d) ─────
            -- rolling OLS beta: cov(stock_ret, factor) / var(factor)
            -- NULL where fewer than 63 trading days of history available.
            -- FII/DII factors are z-scored levels (not pct_change) to avoid
            -- near-zero variance from forward-filled monthly series.
            Beta_to_Nifty50             DECIMAL(10,6),
            Beta_to_Nifty500            DECIMAL(10,6),
            Beta_to_USDINR              DECIMAL(10,6),
            Beta_to_VIX                 DECIMAL(10,6),
            Beta_to_Crude               DECIMAL(10,6),
            Beta_to_Gold                DECIMAL(10,6),
            Beta_to_FII                 DECIMAL(10,6),
            Beta_to_DII                 DECIMAL(10,6),

            -- ── Macro change signal ───────────────────────────────────
            -- diff(Repo_Rate): nonzero only on RBI cut/hike dates.
            Repo_Rate_Change            DECIMAL(10,6),

            -- ── Cross-sectional macro interaction ─────────────────────
            -- USDINR_1d_Return × Revenue_YoY_Growth (stationary × stationary)
            USDINR_x_Revenue_Growth     DECIMAL(12,6),

            -- ── Missing-data flags ────────────────────────────────────
            -- Carry structural information — NULLs in feature columns are
            -- not equivalent (startup window vs genuine absence differ).
            -- Sentiment_Available: 1 if Announcement_Score OR Sentiment_Score_Lag1
            --   is non-NULL AND Date >= SENTIMENT_START_DATE.
            Price_Gap_Flag              TINYINT DEFAULT 0,
            SMA200_Available            TINYINT DEFAULT 0,
            SMA50_Available             TINYINT DEFAULT 0,
            ADX14_Available             TINYINT DEFAULT 0,
            Volatility20d_Available     TINYINT DEFAULT 0,
            Sentiment_Available         TINYINT DEFAULT 0,
            Fundamentals_Available      TINYINT DEFAULT 0,
            Data_Tier                   TINYINT DEFAULT 3,

            -- ── Target variables ──────────────────────────────────────
            -- Target_Return_21d: raw 21d forward return (regression reference)
            -- Target_Rank_21d: cross-sectional percentile among Tier A+B
            --   (primary ML target — computed in Phase 3, needs all tickers)
            -- Target_Direction_Median:  1 if rank > 0.5 else 0
            -- Target_Direction_Tertile: 1=top third, 0=bottom third, NULL=middle
            -- Target_Vol_5d: annualized std(log_returns[t+1..t+5]), ddof=1
            --   (LSTM dual-head volatility target)
            --   NULL for rows where forward window extends past dataset max date.
            Target_Return_21d           DECIMAL(10,6),
            Target_Rank_21d             DECIMAL(10,6),
            Target_Direction_Median     TINYINT,
            Target_Direction_Tertile    TINYINT,
            Target_Vol_5d               DECIMAL(10,6),

            UNIQUE KEY unique_ticker_date (Ticker, Date)
        )
    """,

    # ── Portfolio Positions Tracker ───────────────────────────────────
    "portfolio_positions": """
        CREATE TABLE IF NOT EXISTS portfolio_positions (
            id                      INT AUTO_INCREMENT PRIMARY KEY,
            Ticker                  VARCHAR(20)     NOT NULL,
            Entry_Date              DATE            NOT NULL,
            Entry_Price             DECIMAL(10,4)   NOT NULL,
            Direction               ENUM('long','short') NOT NULL,
            Position_Class          ENUM('core_long','alpha_long',
                                         'short','tactical') NOT NULL,
            Target_Exit_Date        DATE,
            Stop_Loss_Price         DECIMAL(10,4),
            Thesis_Notes            TEXT,
            Exit_Date               DATE,
            Exit_Price              DECIMAL(10,4),
            Exit_Reason             VARCHAR(100),
            NAV_Weight_At_Entry     DECIMAL(10,4),
            Shares                  INT,
            Status                  ENUM('open','closed') DEFAULT 'open',
            UNIQUE KEY unique_ticker_entry (Ticker, Entry_Date, Direction)
        )
    """,

    # ── Sector Fundamentals Median ────────────────────────────────────
    # v2.2 schema — matches RATIO_COLS in features.py exactly.
    # Populated by features.py Phase 1 — fallback for tickers with missing
    # own fundamentals (>400d since period end triggers substitution).
    # Computed per (Sector, Period) from actual non-imputed values only.
    # If <3 tickers in sector have actual data for that Period → NULL.
    # Gross_Margin_Is_Proxy intentionally excluded (flag, not a ratio;
    # meaningless to aggregate across tickers).
    "sector_fundamentals_median": """
        CREATE TABLE IF NOT EXISTS sector_fundamentals_median (
            id                      INT AUTO_INCREMENT PRIMARY KEY,
            Sector                  VARCHAR(50)     NOT NULL,
            Period                  DATE            NOT NULL,
            Ticker_Count            INT,
            Revenue_YoY_Growth      DECIMAL(12,6),
            EBITDA_Margin           DECIMAL(12,6),
            Net_Margin              DECIMAL(12,6),
            FCF_Margin              DECIMAL(12,6),
            Asset_Turnover          DECIMAL(12,6),
            ROA                     DECIMAL(12,6),
            Debt_to_Equity          DECIMAL(12,6),
            Gross_Profit_Margin     DECIMAL(12,6),
            Debt_to_Assets          DECIMAL(12,6),
            OCF_to_Net_Income       DECIMAL(12,6),
            Delta_ROA               DECIMAL(12,6),
            Delta_DebtEquity        DECIMAL(12,6),
            UNIQUE KEY unique_sector_period (Sector, Period)
        )
    """,

    # ── Market Regimes ────────────────────────────────────────────────
    # Daily regime classification. One row per date (all tickers share
    # the same regime on a given day — cross-sectionally valid).
    # Regime_Int: 0=Bear, 1=Bull, 2=HighVol, 3=Sideways
    "market_regimes": """
        CREATE TABLE IF NOT EXISTS market_regimes (
            Date            DATE            NOT NULL,
            Regime_Label    VARCHAR(10)     NOT NULL,
            Regime_Int      TINYINT         NOT NULL,
            Prob_Bull       DECIMAL(6,4)    DEFAULT NULL,
            Prob_Bear       DECIMAL(6,4)    DEFAULT NULL,
            Prob_HighVol    DECIMAL(6,4)    DEFAULT NULL,
            Prob_Sideways   DECIMAL(6,4)    DEFAULT NULL,
            Model_Version   VARCHAR(20)     DEFAULT NULL,
            PRIMARY KEY (Date)
        )
    """
}


def run_setup():
    engine  = get_engine()
    created = []
    skipped = []
    failed  = []

    print("🔧 Setting up hedge_v2.3 tables...\n")

    with engine.connect() as conn:
        for table_name, ddl in TABLES.items():
            try:
                exists = conn.execute(text(
                    f"SELECT COUNT(*) FROM information_schema.tables "
                    f"WHERE table_schema = DATABASE() "
                    f"AND table_name = '{table_name}'"
                )).scalar()

                conn.execute(text(ddl))
                conn.commit()

                if exists:
                    skipped.append(table_name)
                    print(f"  ⏭️  {table_name} — already exists, skipped")
                else:
                    created.append(table_name)
                    print(f"  ✅ {table_name} — created")

            except Exception as e:
                failed.append(table_name)
                print(f"  ❌ {table_name} — {e}")

    print(f"""
{'='*55}
Setup Complete
{'='*55}
  Created : {len(created)} tables
  Skipped : {len(skipped)} tables (already existed)
  Failed  : {len(failed)}  tables → {failed if failed else 'none'}
{'='*55}
""")

    if not failed:
        print("✅ All tables ready. See README.md for the full run order.")
    else:
        print("⚠️  Fix the failed tables before running data scripts.")


if __name__ == "__main__":
    run_setup()