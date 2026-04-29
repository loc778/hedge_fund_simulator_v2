# data/setup_db.py
# ═══════════════════════════════════════════════════════════
# ONE-TIME DATABASE SETUP — hedge_v2
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
            
            RS_21d                      DECIMAL(10,6),
            RS_60d                      DECIMAL(10,6),

            -- ── Cross-sectional momentum signals ─────────────
            
            Mom_12_1                    DECIMAL(12,6),   -- Return_252d - Return_21d
            Mom_6_1                     DECIMAL(12,6),   -- Return_126d - Return_21d
            Mom_1M_Rank                 DECIMAL(10,6),   -- CS rank of Return_21d per date
            Mom_12_1_Rank               DECIMAL(10,6),   -- CS rank of Mom_12_1 per date
            Mom_6_1_Rank                DECIMAL(10,6),   -- CS rank of Mom_6_1 per date
            RS_21d_Rank                 DECIMAL(10,6),   -- CS rank of RS_21d per date

            -- ── Technical: stationary price-position ratios ───────────
            
            Price_to_SMA20              DECIMAL(10,6),
            Price_to_SMA50              DECIMAL(10,6),
            Price_to_SMA200             DECIMAL(10,6),
            Price_to_52W_High           DECIMAL(10,6),
            Price_to_52W_Low            DECIMAL(10,6),

            -- ── Technical indicators (all stationary) ─────────────────
            MACD_Hist                   DECIMAL(10,6),
            RSI_14                      DECIMAL(10,4),
            BB_Width                    DECIMAL(10,6),  
            BB_PctB                     DECIMAL(10,6),  
            ATR_14                      DECIMAL(10,4),
            Stoch_K                     DECIMAL(10,4),
            ADX_14                      DECIMAL(10,4),
            OBV_Change_5d               DECIMAL(10,6),  
            VWAP_Dev                    DECIMAL(10,6),
            Revenue_YoY_Growth          DECIMAL(12,6),
            EBITDA_Margin               DECIMAL(12,6),
            Net_Margin                  DECIMAL(12,6),
            FCF_Margin                  DECIMAL(12,6),
            Asset_Turnover              DECIMAL(12,6),
            ROA                         DECIMAL(12,6),
            Debt_to_Equity              DECIMAL(12,6),
            Gross_Profit_Margin         DECIMAL(12,6),  
            Gross_Margin_Is_Proxy       TINYINT DEFAULT 0, 
            Debt_to_Assets              DECIMAL(12,6),
            OCF_to_Net_Income           DECIMAL(12,6),  
            Delta_ROA                   DECIMAL(12,6),  
            Delta_DebtEquity            DECIMAL(12,6),  
            Rel_ROA                     DECIMAL(12,6),
            Rel_EBITDA_Margin           DECIMAL(12,6),
            ROA_Sector_Rank             DECIMAL(10,6),  
            EBITDA_Margin_Sector_Rank   DECIMAL(10,6),  
            Revenue_Growth_Sector_Rank  DECIMAL(10,6),  
            Announcement_Score          DECIMAL(10,4),
            Sentiment_Score_Lag1        DECIMAL(10,4),
            Regime_Int                  TINYINT,
            Beta_to_Nifty50             DECIMAL(10,6),
            Beta_to_Nifty500            DECIMAL(10,6),
            Beta_to_USDINR              DECIMAL(10,6),
            Beta_to_VIX                 DECIMAL(10,6),
            Beta_to_Crude               DECIMAL(10,6),
            Beta_to_Gold                DECIMAL(10,6),
            Beta_to_FII                 DECIMAL(10,6),
            Beta_to_DII                 DECIMAL(10,6),
            Repo_Rate_Change            DECIMAL(10,6),
            USDINR_x_Revenue_Growth     DECIMAL(12,6),
            Price_Gap_Flag              TINYINT DEFAULT 0,
            SMA200_Available            TINYINT DEFAULT 0,
            SMA50_Available             TINYINT DEFAULT 0,
            ADX14_Available             TINYINT DEFAULT 0,
            Volatility20d_Available     TINYINT DEFAULT 0,
            Sentiment_Available         TINYINT DEFAULT 0,
            Fundamentals_Available      TINYINT DEFAULT 0,
            Data_Tier                   TINYINT DEFAULT 3,
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