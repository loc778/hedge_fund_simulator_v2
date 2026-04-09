# data/setup_db.py
# ═══════════════════════════════════════════════════════════
# ONE-TIME DATABASE SETUP — hedge_v2
# Run once on any new machine to create all tables.
# IF NOT EXISTS means re-running never breaks anything.
#
# HOW TO ADD A NEW TABLE:
# Add entry to TABLES dict below, following existing format.
# Also add the name to TABLES dict in config.py.
# ═══════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine
from sqlalchemy import text

engine = get_engine()

TABLES = {

    # ── Layer 1: Price & Volume (Bhavcopy) ───────────────────────────
    # Source: NSE Bhavcopy archives (Jan 2010 → present)
    # Script: data/bhavcopy_ingestion.py
    # Adj_Close calculated using yfinance adjustment factors
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
    # Source: yfinance splits + dividends history
    # Used to calculate Adj_Close from raw bhavcopy Close prices
    # Script: data/bhavcopy_ingestion.py (fetched as part of ingestion)
    "corporate_actions": """
        CREATE TABLE IF NOT EXISTS corporate_actions (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            Ticker          VARCHAR(20)     NOT NULL,
            Date            DATE            NOT NULL,
            Action_Type     VARCHAR(20)     NOT NULL,   -- 'split' or 'dividend'
            Ratio           DECIMAL(10,6),              -- split ratio (e.g. 0.5 for 2:1 split)
            Amount          DECIMAL(10,4),              -- dividend amount in INR
            UNIQUE KEY unique_ticker_date_action (Ticker, Date, Action_Type)
        )
    """,

    # ── Layer 4: Technical Indicators ────────────────────────────────
    # Source: calculated from nifty500_ohlcv
    # Script: data/indicators.py
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
    # Source: Screener.in (primary, 10+ years quarterly data)
    #         yfinance only used as fallback for stocks not on Screener.in
    # Script: data/screener_fundamentals.py
    "nifty500_fundamentals": """
        CREATE TABLE IF NOT EXISTS nifty500_fundamentals (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            Ticker              VARCHAR(20)     NOT NULL,
            Period              DATE            NOT NULL,
            Revenue             BIGINT,
            Gross_Profit        BIGINT,
            EBITDA              BIGINT,
            Net_Income          BIGINT,
            EPS_Basic           DECIMAL(10,4),
            EPS_Diluted         DECIMAL(10,4),
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
    # Sources:
    #   yfinance       — India VIX, USDINR, Crude Oil, Gold (daily)
    #   FRED API       — CPI India, GDP India, Fed Rate, US CPI, US 10Y (monthly/quarterly)
    #   NSE API        — FII_Net_Buy_Cr, DII_Net_Buy_Cr (daily flows)
    #   RBI DBIE       — Repo_Rate, IIP_Growth, Forex_Reserves_USD (monthly)
    # Scripts: data/macro.py, data/fii_dii.py, data/rbi_macro.py
    "macro_indicators": """
        CREATE TABLE IF NOT EXISTS macro_indicators (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            Date                DATE        NOT NULL UNIQUE,
            India_VIX           DECIMAL(10,4),
            USDINR              DECIMAL(10,4),
            Crude_Oil           DECIMAL(10,4),
            Gold                DECIMAL(10,4),
            CPI_India           DECIMAL(10,4),
            GDP_India           DECIMAL(20,4),
            Fed_Funds_Rate      DECIMAL(10,4),
            US_CPI              DECIMAL(10,4),
            US_10Y_Bond         DECIMAL(10,4),
            Repo_Rate           DECIMAL(10,4),
            IIP_Growth          DECIMAL(10,4),
            Forex_Reserves_USD  DECIMAL(20,4),
            FII_Net_Buy_Cr      DECIMAL(15,4),
            DII_Net_Buy_Cr      DECIMAL(15,4)
        )
    """,

    # ── Layer 3B: Sentiment Data ──────────────────────────────────────
    # Source: GDELT + NewsAPI + ET RSS → FinBERT via HuggingFace API
    # Script: data/sentiment.py
    # NEVER TRUNCATE THIS TABLE — historical news cannot be recovered
    "nifty500_sentiment": """
        CREATE TABLE IF NOT EXISTS nifty500_sentiment (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            Date            DATE            NOT NULL,
            Ticker          VARCHAR(20)     NOT NULL,
            Sentiment_Score DECIMAL(10,4),
            Positive_Score  DECIMAL(10,4),
            Negative_Score  DECIMAL(10,4),
            Neutral_Score   DECIMAL(10,4),
            Headlines_Count INT,
            Source          VARCHAR(50),
            UNIQUE KEY unique_ticker_date (Ticker, Date)
        )
    """,

    # ── Stock Data Quality Classification ─────────────────────────────
    # Source: computed from nifty500_ohlcv by data/data_quality.py
    # Tier A: 8+ years, <5% gaps, ADV >= 5Cr
    # Tier B: 4+ years, <25% gaps, ADV >= 1Cr
    # Tier C: everything else
    # Tier X: < 252 trading days — excluded from all training
    "stock_data_quality": """
        CREATE TABLE IF NOT EXISTS stock_data_quality (
            Ticker                VARCHAR(20)     NOT NULL PRIMARY KEY,
            Trading_Days_Total    INT,
            Data_Start_Date       DATE,
            Coverage_Years        DECIMAL(5,2),
            Gap_Pct               DECIMAL(5,2),
            Avg_Daily_Volume_Cr   DECIMAL(10,4),
            F_and_O_Listed        TINYINT,
            Data_Tier             CHAR(1),        -- 'A', 'B', 'C', or 'X'
            Tier_Assigned_Date    DATE
        )
    """,

    # ── Feature Engineering: Unified ML Training Dataset ─────────────
    # Source: combined from all 5 data layers + missing data flags
    # Script: data/features.py
    # One row per stock per day — direct input to ML models
    "features_master": """
        CREATE TABLE IF NOT EXISTS features_master (
            id                      INT AUTO_INCREMENT PRIMARY KEY,
            Date                    DATE            NOT NULL,
            Ticker                  VARCHAR(20)     NOT NULL,

            -- From Layer 1 (OHLCV)
            Open                    DECIMAL(10,4),
            High                    DECIMAL(10,4),
            Low                     DECIMAL(10,4),
            Close                   DECIMAL(10,4),
            Adj_Close               DECIMAL(10,4),
            Volume                  BIGINT,
            VWAP_Daily              DECIMAL(10,4),

            -- Price derived features
            Return_1d               DECIMAL(10,4),
            Return_5d               DECIMAL(10,4),
            Return_21d              DECIMAL(10,4),
            Volatility_20d          DECIMAL(10,4),
            Volume_Ratio_20d        DECIMAL(10,4),

            -- From Layer 4 (Technical Indicators)
            SMA_20                  DECIMAL(10,4),
            SMA_50                  DECIMAL(10,4),
            SMA_200                 DECIMAL(10,4),
            EMA_9                   DECIMAL(10,4),
            EMA_21                  DECIMAL(10,4),
            MACD                    DECIMAL(10,4),
            MACD_Signal             DECIMAL(10,4),
            MACD_Hist               DECIMAL(10,4),
            RSI_14                  DECIMAL(10,4),
            BB_Upper                DECIMAL(10,4),
            BB_Middle               DECIMAL(10,4),
            BB_Lower                DECIMAL(10,4),
            BB_Width                DECIMAL(10,4),
            ATR_14                  DECIMAL(10,4),
            Stoch_K                 DECIMAL(10,4),
            Stoch_D                 DECIMAL(10,4),
            ADX_14                  DECIMAL(10,4),
            OBV                     DECIMAL(20,4),
            VWAP_Dev                DECIMAL(10,4),

            -- From Layer 2 (Fundamentals — forward filled quarterly)
            PE_Ratio                DECIMAL(10,4),
            PB_Ratio                DECIMAL(10,4),
            EV_EBITDA               DECIMAL(10,4),
            ROE                     DECIMAL(10,4),
            ROA                     DECIMAL(10,4),
            ROCE                    DECIMAL(10,4),
            Debt_to_Equity          DECIMAL(10,4),
            FCF_Yield               DECIMAL(10,4),
            Dividend_Yield          DECIMAL(10,4),
            EPS_Basic               DECIMAL(10,4),

            -- From Layer 3B (Sentiment — forward filled daily)
            Sentiment_Score         DECIMAL(10,4),
            Positive_Score          DECIMAL(10,4),
            Negative_Score          DECIMAL(10,4),

            -- From Layer 3A (Macro — daily)
            India_VIX               DECIMAL(10,4),
            USDINR                  DECIMAL(10,4),
            Crude_Oil               DECIMAL(10,4),
            Gold                    DECIMAL(10,4),
            CPI_India               DECIMAL(10,4),
            GDP_India               DECIMAL(20,4),
            Fed_Funds_Rate          DECIMAL(10,4),
            US_CPI                  DECIMAL(10,4),
            US_10Y_Bond             DECIMAL(10,4),
            Repo_Rate               DECIMAL(10,4),
            IIP_Growth              DECIMAL(10,4),
            FII_Net_Buy_Cr          DECIMAL(15,4),
            DII_Net_Buy_Cr          DECIMAL(15,4),

            -- Missing data flag columns (Section 3.4 of architecture framework)
            -- These tell models WHERE data is missing vs WHERE it is actually zero
            Price_Gap_Flag          TINYINT DEFAULT 0,   -- 1 if OHLCV gap > 5 days
            SMA200_Available        TINYINT DEFAULT 0,   -- 1 if 200 days of history exist
            Sentiment_Available     TINYINT DEFAULT 0,   -- 1 if actual sentiment score exists
            PE_Is_PB_Proxy          TINYINT DEFAULT 0,   -- 1 if PE replaced by PB for banks
            Fundamentals_Available  TINYINT DEFAULT 0,   -- 1 if actual quarterly data available
            Data_Tier               TINYINT DEFAULT 3,   -- 1=TierA, 2=TierB, 3=TierC

            -- Target variables (what the ML model predicts)
            Target_Return_21d       DECIMAL(10,4),
            Target_Direction        TINYINT,             -- 1=up, 0=down

            UNIQUE KEY unique_ticker_date (Ticker, Date)
        )
    """,

    # ── Portfolio Positions Tracker ───────────────────────────────────
    # Tracks every active and closed position with thesis and risk notes
    # Script: updated by portfolio engine and dashboard
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
    # Pre-computed sector median ratios per quarter
    # Used as fallback when a stock has missing fundamental data (> 4 quarters)
    # Script: data/features.py computes and saves this during feature engineering
    "sector_fundamentals_median": """
        CREATE TABLE IF NOT EXISTS sector_fundamentals_median (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            Sector          VARCHAR(50)     NOT NULL,
            Period          DATE            NOT NULL,
            PE_Ratio        DECIMAL(10,4),
            PB_Ratio        DECIMAL(10,4),
            EV_EBITDA       DECIMAL(10,4),
            ROE             DECIMAL(10,4),
            ROA             DECIMAL(10,4),
            ROCE            DECIMAL(10,4),
            Debt_to_Equity  DECIMAL(10,4),
            FCF_Yield       DECIMAL(10,4),
            Dividend_Yield  DECIMAL(10,4),
            EPS_Basic       DECIMAL(10,4),
            UNIQUE KEY unique_sector_period (Sector, Period)
        )
    """,

}

# ═══════════════════════════════════════════════════════════
# DO NOT EDIT BELOW THIS LINE
# ═══════════════════════════════════════════════════════════

print("🔧 Setting up hedge_v2_db tables...\n")

created = []
skipped = []
failed  = []

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
    print("✅ All tables created. See README.md for the full run order.")
else:
    print("⚠️  Fix the failed tables before running data scripts")