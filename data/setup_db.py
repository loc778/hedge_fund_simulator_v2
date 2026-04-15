# data/setup_db.py
# ═══════════════════════════════════════════════════════════
# ONE-TIME DATABASE SETUP — hedge_v2
# Run once on any new machine to create all tables.
# IF NOT EXISTS means re-running never breaks anything.
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
    # C5 FIX: FII column split into two:
    #   FII_Monthly_Net_Cr = monthly total ÷ trading days (historical, 2010–present)
    #                        Written by fii_dii_historical.py
    #                        Unit: Cr per trading day (approximation)
    #   FII_Daily_Net_Cr   = actual daily NSE-reported value (last ~30 days)
    #                        Written by fii_dii.py
    #                        Unit: Cr per day (exact)
    # features.py uses FII_Daily_Net_Cr when not NULL, else FII_Monthly_Net_Cr.
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
            FII_Monthly_Net_Cr  DECIMAL(15,4),
            FII_Daily_Net_Cr    DECIMAL(15,4),
            DII_Net_Buy_Cr      DECIMAL(15,4)
        )
    """,

    # ── Layer 3B: Sentiment Data ──────────────────────────────────────
    # NEVER TRUNCATE — historical news cannot be recovered
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
    "features_master": """
        CREATE TABLE IF NOT EXISTS features_master (
            id                      INT AUTO_INCREMENT PRIMARY KEY,
            Date                    DATE            NOT NULL,
            Ticker                  VARCHAR(20)     NOT NULL,

            Open                    DECIMAL(10,4),
            High                    DECIMAL(10,4),
            Low                     DECIMAL(10,4),
            Close                   DECIMAL(10,4),
            Adj_Close               DECIMAL(10,4),
            Volume                  BIGINT,
            VWAP_Daily              DECIMAL(10,4),

            Return_1d               DECIMAL(10,4),
            Return_5d               DECIMAL(10,4),
            Return_21d              DECIMAL(10,4),
            Volatility_20d          DECIMAL(10,4),
            Volume_Ratio_20d        DECIMAL(10,4),

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

            Sentiment_Score         DECIMAL(10,4),
            Positive_Score          DECIMAL(10,4),
            Negative_Score          DECIMAL(10,4),

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
            -- Unified FII feature: daily when available, monthly approx otherwise
            FII_Net_Cr              DECIMAL(15,4),
            DII_Net_Buy_Cr          DECIMAL(15,4),

            Price_Gap_Flag          TINYINT DEFAULT 0,
            SMA200_Available        TINYINT DEFAULT 0,
            Sentiment_Available     TINYINT DEFAULT 0,
            PE_Is_PB_Proxy          TINYINT DEFAULT 0,
            Fundamentals_Available  TINYINT DEFAULT 0,
            Data_Tier               TINYINT DEFAULT 3,

            Target_Return_21d       DECIMAL(10,4),
            Target_Direction        TINYINT,

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


def run_setup():
    engine  = get_engine()
    created = []
    skipped = []
    failed  = []

    print("🔧 Setting up hedge_v2_db tables...\n")

    # For existing macro_indicators table: add new FII columns if missing
    # (handles upgrade on machines that already ran the old setup)
    _migrate_macro_columns(engine)

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


def _migrate_macro_columns(engine):
    """
    Adds FII_Monthly_Net_Cr and FII_Daily_Net_Cr columns to macro_indicators
    if they don't exist yet (handles upgrade on machines with old schema).
    Also removes old FII_Net_Buy_Cr if it still exists from old schema.
    """
    migrations = [
        ("FII_Monthly_Net_Cr", "DECIMAL(15,4)",
         "ADD COLUMN FII_Monthly_Net_Cr DECIMAL(15,4) AFTER Forex_Reserves_USD"),
        ("FII_Daily_Net_Cr", "DECIMAL(15,4)",
         "ADD COLUMN FII_Daily_Net_Cr DECIMAL(15,4) AFTER FII_Monthly_Net_Cr"),
        ("DII_Net_Buy_Cr", "DECIMAL(15,4)",
         "ADD COLUMN DII_Net_Buy_Cr DECIMAL(15,4) AFTER FII_Daily_Net_Cr"),
    ]

    with engine.connect() as conn:
        # Check if macro_indicators exists at all
        exists = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'macro_indicators'"
        )).scalar()

        if not exists:
            return   # will be created fresh by the main DDL loop

        for col_name, _, alter_sql in migrations:
            col_exists = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_schema = DATABASE() "
                f"AND table_name = 'macro_indicators' "
                f"AND column_name = '{col_name}'"
            )).scalar()

            if not col_exists:
                try:
                    conn.execute(text(f"ALTER TABLE macro_indicators {alter_sql}"))
                    conn.commit()
                    print(f"  🔄 Migration: added {col_name} to macro_indicators")
                except Exception as e:
                    print(f"  ⚠️  Migration for {col_name} failed: {e}")

        # Migrate data from old FII_Net_Buy_Cr → FII_Monthly_Net_Cr if old column exists
        old_col_exists = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_schema = DATABASE() "
            "AND table_name = 'macro_indicators' "
            "AND column_name = 'FII_Net_Buy_Cr'"
        )).scalar()

        if old_col_exists:
            try:
                conn.execute(text(
                    "UPDATE macro_indicators "
                    "SET FII_Monthly_Net_Cr = FII_Net_Buy_Cr "
                    "WHERE FII_Monthly_Net_Cr IS NULL AND FII_Net_Buy_Cr IS NOT NULL"
                ))
                conn.execute(text(
                    "ALTER TABLE macro_indicators DROP COLUMN FII_Net_Buy_Cr"
                ))
                conn.commit()
                print("  🔄 Migration: FII_Net_Buy_Cr → FII_Monthly_Net_Cr, old column dropped")
            except Exception as e:
                print(f"  ⚠️  FII_Net_Buy_Cr migration failed: {e}")


if __name__ == "__main__":
    run_setup()