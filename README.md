# AI Hedge Fund Simulator v2 — NSE Nifty 500

**AI-driven long/short equity system for NSE Nifty 500.**
Generates ranked BUY/SELL/HOLD signals with confidence scores across ~467 actively traded stocks.
Built on 15 years of historical NSE data (2010–present).

GitHub: https://github.com/loc778/hedge_fund_simulator_v2

---

## 1. Setup — First Run

### Prerequisites

- Python 3.11 or below
- MySQL 8.x running locally
- `.env` file at project root (see below)
- `files/nifty500_tickers.csv` downloaded from NSE
- `files/nifty500_sectors.csv` (NSE official sector CSV)
- `files/fo_list.csv` (F&O eligible list from NSE)
- FRED API key (free: https://fred.stlouisfed.org/docs/api/api_key.html)
  - create account in FRED
  - select request API key
  - give a reason for api request example "need api for macro data "
  - submit and copy the api key and paste in .env file at fred_api_key

### `.env` format

```
DB_HOST=localhost
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_NAME=hedge_v2_db
FRED_API_KEY=your_fred_api_key
```

### Installation

```bash
# STEP 1:
git clone https://github.com/loc778/hedge_fund_simulator_v2.git
#clone in desktop

# STEP 2 : Create virtual environment and choose python 3.11 or below
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# STEP 3: Create database
In mysql "CREATE DATABASE hedge_v2_db;"

# STEP 4:
# create .env file. use the same format in .env.example


# STEP 5 Run full setup
python setup_pipeline.py
```

### ML Model Training (after setup completes)

```bash
# 1. Export features
python data/export_features.py

# 2. Upload exports/ to Google Drive (folder: "Hedge_Fund_Simulator_V2") → run xgboost.ipynb on Google Colab → download .pkl outputs → ML_models/ at local system

# 3. Upload exports/ to Kaggle dataset → run lstm.ipynb on Kaggle → download .keras + .pkl outputs → ML_models/

# 4. Run ensemble (after model files are in place)
python ML_scripts/ensemble_final.py

# 5. Launch dashboard
streamlit run dashboard/app.py
```

---

## 2. Project Overview

This system simulates an AI-driven hedge fund operating on NSE-listed Nifty 500 equities.
It is structured as a full end-to-end pipeline:

```
Raw Market Data → Feature Engineering → ML Models → Ensemble Signals → Portfolio Construction → Dashboard
```

**Signal horizon:** 21 trading days (XGBoost/LightGBM), 10 trading days (LSTM).
**Universe:** ~467 Nifty 500 stocks after excluding Tier X (recent IPOs < 252 trading days).
**Strategy type:** Long/short equity with regime-aware position sizing.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER (Local)                       │
│                                                                 │
│  NSE Bhavcopy (OHLCV) → nifty500_ohlcv                          │
│  Screener.in (Fundamentals) → nifty500_fundamentals             │
│  StockEdge CM API (FII/DII) → fii_dii_flow                      │
│  FRED + yfinance (Macro) → macro_indicators                     │
│  RBI Repo Rate (hardcoded history) → macro_indicators           │
│  Technical Indicators (ta library) → nifty500_indicators        │
│  Data Quality Classification → stock_data_quality               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   FEATURE ENGINEERING (Local)                   │
│                                                                 │
│  features.py → features_master (MySQL)                          │
│  export_features.py → exports/ (Parquet, CSV)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
┌─────────────▼──────────┐  ┌──────────▼──────────────────────────┐
│  HMM (Local)           │  │  XGBoost + LightGBM (Google Colab)  │
│  ML_scripts/hmm.py     │  │  ML_scripts/xgboost.ipynb           │
│  4-state regime model  │  │  21-day return ranking target       │
│  → market_regimes      │  │  → ML_models/xgboost_v*.pkl         │
│  → ML_models/hmm_*.pkl │  │  → ML_models/lightgbm_v*.pkl        │
└────────────────────────┘  └─────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
┌─────────────▼─────────────────────────▼────────────────────────┐
│               LSTM Dual-Head (Kaggle)                          │
│               ML_scripts/lstm.ipynb                            │
│               Head 1: 10-day return rank                       │
│               Head 2: 5-day realized volatility                │
│               → ML_models/lstm_v*.keras                        │
│               → ML_models/lstm_norm_v*.pkl                     │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                ENSEMBLE (Local)                                 │
│                ML_scripts/ensemble_final.py                     │
│                Rank calibration → confidence scores             │
│                BUY / SELL / HOLD signal generation              │
│                → exports/model_output/signals_*.csv             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│           PORTFOLIO + RISK + DASHBOARD (Local)                  │
│           portfolio/optimizer.py                                │
│           risk/risk_manager.py                                  │
│           dashboard/app.py (Streamlit)                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Folder Structure

```
hedge_fund_simulator_v2/
│
├── dashboard/
│   └── app.py                      # Streamlit UI
│
├── data/
│   ├── bhavcopy_ingestion.py       # NSE OHLCV download & ingestion
│   ├── data_quality.py             # Tier A/B/C/X classification
│   ├── db.py                       # SQLAlchemy engine factory
│   ├── export_features.py          # Exports features_master → Parquet/CSV
│   ├── features.py                 # Feature engineering pipeline
│   ├── fii_dii_stockedge.py        # FII/DII via StockEdge CM API
│   ├── indicators.py               # Technical indicators (ta library)
│   ├── macro.py                    # Macro data: FRED + yfinance
│   ├── rbi_macro.py                # RBI repo rate → macro_indicators
│   ├── screener_fundamentals.py    # Fundamental data via Screener.in
│   └── setup_db.py                 # DB schema creation / migration
│
├── exports/
│   └── model_output/               # signals_*.csv, nav_series_*.csv
│
├── files/
│   ├── fo_list.csv                 # F&O eligible tickers (NSE)
│   ├── nifty500_sectors.csv        # Symbol → Industry mapping (NSE official)
│   └── nifty500_tickers.csv        # NSE watchlist CSV (re-download quarterly)
│
├── logs/                           # Auto-generated pipeline logs
│
├── ML_models/
│   ├── results/                    # HMM regime plots
│   ├── hmm_model_v*.pkl            # Trained HMM model
│   ├── hmm_scaler_v*.pkl           # HMM feature scaler
│   ├── hmm_statemap_v*.pkl         # State → regime label mapping
│   ├── xgboost_v*.pkl              # Trained XGBoost model
│   ├── lightgbm_v*.pkl             # Trained LightGBM model
│   ├── lstm_v*.keras               # Trained LSTM model
│   ├── lstm_norm_v*.pkl            # LSTM feature normalizer
│   ├── rank_calibration.pkl        # Rank → confidence score calibration
│   └── adj_factors.pkl             # Corporate action adjustment factors
│
├── ML_scripts/
│   ├── ensemble_final.py           # Ensemble combiner + signal generation
│   ├── hmm.py                      # HMM regime detection (local)
│   ├── lstm.ipynb                  # LSTM training notebook (Kaggle)
│   └── xgboost.ipynb               # XGBoost + LightGBM notebook (Google Colab)
│
├── models/                         # model artefacts
│
├── portfolio/
│   ├── __init__.py
│   └── optimizer.py                # Position sizing, sector cap, allocation
│
├── risk/
│   ├── __init__.py
│   └── risk_manager.py             # Pre-trade risk checks, stop-loss logic
│
├── .env                            # DB credentials, FRED API key (not committed)
├── .gitignore
├── config.py                       # Single source of truth for all parameters
├── daily_refresh.py                # Incremental daily update pipeline
├── README.md
├── requirements.txt
└── setup_pipeline.py               # Full first-run pipeline orchestrator
```

---

## 5. Data Sources

| Source           | Script                     | Table                   |
| ---------------- | -------------------------- | ----------------------- |
| NSE Bhavcopy     | `bhavcopy_ingestion.py`    | `nifty500_ohlcv`        |
| Screener.in      | `screener_fundamentals.py` | `nifty500_fundamentals` |
| StockEdge CM API | `fii_dii_stockedge.py`     | `macro_indicators`      |
| FRED API         | `macro.py`                 | `macro_indicators`      |
| yfinance 0.2.48  | `macro.py`                 | `macro_indicators`      |
| RBI Repo Rate    | `rbi_macro.py`             | `macro_indicators`      |
| ta library       | `indicators.py`            | `nifty500_indicators`   |

- `pandas-ta` not used (Python 3.11+ incompatible).
- Historical DII = NULL. Never zero-fill. StockEdge cookie session required.
- Re-download `nifty500_tickers.csv` quarterly at SEBI rebalance (Mar/Jun/Sep/Dec).

---

## 6. Database Schema

**Database:** `hedge_v2_db` (MySQL) — managed via `data/setup_db.py` (safe to re-run).

| Table                        | Description                                       |
| ---------------------------- | ------------------------------------------------- |
| `nifty500_ohlcv`             | Daily OHLCV + Adj_Close (splits-adjusted)         |
| `nifty500_indicators`        | 30+ technical indicators                          |
| `nifty500_fundamentals`      | Quarterly fundamentals. C2 lookahead fix applied. |
| `macro_indicators`           | Macro + FII/DII data. Historical DII = NULL.      |
| `stock_data_quality`         | Tier A/B/C/X per ticker                           |
| `features_master`            | Full feature set per (Ticker, Date)               |
| `market_regimes`             | Daily HMM regime labels (0–3)                     |
| `portfolio_positions`        | Active and historical positions                   |
| `corporate_actions`          | Splits/bonus events for Adj_Close                 |
| `sector_fundamentals_median` | Sector-level median fundamentals                  |

---

## 7. Pipeline — Local (Windows)

### First-time setup

```bash
# From project root
python setup_pipeline.py
```

Runs in order:

1. `pip install -r requirements.txt`
2. `data/setup_db.py`
3. `data/bhavcopy_ingestion.py`
4. `data/indicators.py`
5. `data/screener_fundamentals.py`
6. `data/macro.py`
7. `data/fii_dii_stockedge.py`
8. `data/rbi_macro.py`
9. `data/data_quality.py`
10. `ML_scripts/hmm.py`
11. `data/features.py`
12. `data/export_features.py`
13. `ML_scripts/ensemble_final.py`

### Data tier classification

`data/data_quality.py` classifies each ticker into:

| Tier | Criteria                                    | Count (Apr 2026) |
| ---- | ------------------------------------------- | ---------------- |
| A    | ≥ 8 years, ≥ 93% coverage, ≥ ₹5 Cr ADV      | ~275             |
| B    | ≥ 4 years, ≥ 75% coverage, ≥ ₹1 Cr ADV      | ~107             |
| C    | Below Tier B thresholds but ≥ 252 days      | ~85              |
| X    | < 252 trading days (recent IPOs) — excluded | ~33              |

Tier X list is also maintained in `config.py → TIER_X_EXCLUDED` for scripts that run before `data_quality.py`.

### Lookahead leak prevention

- **C1:** All technical indicators use `Adj_Close`, not raw `Close`.
- **C2:** Fundamental point-in-time enforcement — PE_Ratio, ROE, ROCE, Dividend_Yield, Book_Value_PS set NULL for all historical rows except the most recent entry per ticker.
- **C3:** `Adj_Close` computed from `Close` using backward-adjusted corporate action factors.

---

## 8. ML Training — Remote

### XGBoost + LightGBM — Google Colab (T4 GPU)

**Notebook:** `ML_scripts/xgboost.ipynb`
**Run on:** Google Colab (free tier, T4 GPU)

- Input: `exports/` Parquet files (sync to Google Drive before training)
- Target: `Target_Return_21d` — 21-day forward return rank (cross-sectional percentile)
- Models: XGBoost ranker + LightGBM ranker
- Walk-forward expanding window CV with purge gap to prevent leakage
- Outputs: `xgboost_v*.pkl`, `lightgbm_v*.pkl`, `rank_calibration.pkl`

**Workflow:**

```
1. Run export_features.py locally → exports/features_*.parquet
2. Upload to Google Drive
3. Open xgboost.ipynb in Colab → mount Drive → run all cells
4. Download output .pkl files → place in ML_models/
```

### LSTM Dual-Head — Kaggle (GPU)

**Notebook:** `ML_scripts/lstm.ipynb`
**Run on:** Kaggle (P100/T4 GPU — more reliable than Colab for long training runs)

- Input: same Parquet exports
- Architecture: Shared LSTM encoder → two heads
  - Head 1: 10-day return rank (classification)
  - Head 2: 5-day realized volatility (regression)
- Sequence length: configurable (default 20 trading days lookback)
- Outputs: `lstm_v*.keras`, `lstm_norm_v*.pkl`

**Workflow:**

```
1. Upload features Parquet to Kaggle dataset
2. Open lstm.ipynb as Kaggle notebook → enable GPU accelerator → run all cells
3. Download output files → place in ML_models/
```

### HMM Regime Detection — Local

**Script:** `ML_scripts/hmm.py`
**Run on:** Local machine (CPU, included in daily_refresh pipeline)

- Features: Nifty daily return, India VIX, 20-day volatility, 5-day return, FII flow z-score
- 4 hidden states mapped to regime labels: Bull / Bear / High-Vol / Recovery
- 75 random restarts, selects best BIC
- Outputs: `hmm_model_v*.pkl`, `hmm_scaler_v*.pkl`, `hmm_statemap_v*.pkl`
- Writes daily regime labels to `market_regimes` table

---

## 9. Ensemble & Signals

**Script:** `ML_scripts/ensemble_final.py`
**Run on:** Local (CPU only — GPU disabled deliberately)

**Signal generation logic:**

1. Load features from `features_master`
2. Run XGBoost + LightGBM → 21-day return rank scores
3. Run LSTM → 10-day return rank + volatility scores
4. Load HMM regime from `market_regimes` → modulate position sizing
5. Combine scores via calibrated rank weighting (`rank_calibration.pkl`)
6. Apply cross-sectional normalization per date
7. Classify each stock: BUY / SELL / HOLD with confidence score (0–1)
8. Write signals to `exports/model_output/signals_YYYYMMDD.csv`
9. Write NAV backtest series to `exports/model_output/nav_series_YYYYMMDD.csv`

**Horizon note:** XGBoost/LightGBM (21-day) and LSTM (10-day) targets are not averaged directly. The ensemble combiner documents and accounts for the horizon difference via rank calibration — raw scores are not mixed.

---

## 10. Portfolio & Risk

### Optimizer (`portfolio/optimizer.py`)

- Reads latest signal CSV
- Applies sector cap: max 3 stocks per sector (prevents IT concentration)
- ATR-normalized position sizing: `ATR_14 / Close` prevents low-priced stock over-allocation
- Midcap vs. large-cap size differentiation
- Writes final allocations to `portfolio_positions` table

### Risk Manager (`risk/risk_manager.py`)

Pre-trade checks enforced before any position is committed:

- Max single core long position
- Max single midcap long position
- Max single short position
- Cash reserve minimum
- Minimum position size threshold
- Financial sector concentration limits

---

## 11. Dashboard

**Script:** `dashboard/app.py`
**Run:** `streamlit run dashboard/app.py` from project root

**Tabs:**

- **Signals** — Current BUY/SELL/HOLD table with confidence scores, sector, stop-loss
- **Portfolio** — Active positions, allocation %, NAV tracking
- **Regime** — Current HMM market regime + historical regime chart
- **Backtest** — NAV curve vs. Nifty 500 benchmark
- **Pipeline Status** — Last run date per data source

**Sidebar:**

- Starting NAV input (₹) for position sizing simulation
- Refresh Data button (triggers `daily_refresh.py`)

All DB access is read-only. No writes from dashboard.

---

## 12. Configuration

**`config.py` is the single file to modify for system-wide changes.**

Key parameters:

| Parameter          | Location    | Description                                       |
| ------------------ | ----------- | ------------------------------------------------- |
| `DATA_START`       | `config.py` | Historical data start date (currently 2010-01-01) |
| `TICKERS`          | `config.py` | Loaded from `files/nifty500_tickers.csv`          |
| `TIER_X_EXCLUDED`  | `config.py` | Recent IPO exclusion list                         |
| `NSE_HOLIDAYS`     | `config.py` | Updated annually                                  |
| `BANKING_TICKERS`  | `config.py` | Tickers using bank-specific P&L parsing           |
| `TABLES`           | `config.py` | All MySQL table name mappings                     |
| `MACRO_YFINANCE`   | `config.py` | yfinance macro symbol map                         |
| `MACRO_FRED`       | `config.py` | FRED series IDs                                   |
| `FEATURES`         | `config.py` | Feature engineering tunable params                |
| `DATA_QUALITY`     | `config.py` | Tier A/B/C thresholds                             |
| `RBI_REPO_HISTORY` | `config.py` | Full MPC rate history — append after each meeting |

**When adding new tables or libraries:** update `config.py → TABLES`, `data/setup_db.py`, and `requirements.txt`.

---

## 13. Daily Operations

**Scheduled via Windows Task Scheduler — run after market close (18:30 IST or later)**

```bash
python daily_refresh.py
```

Daily pipeline steps:

1. `bhavcopy_ingestion.py` — fetch today's OHLCV
2. `indicators.py` — compute today's indicators
3. `macro.py` — update macro data
4. `fii_dii_stockedge.py` — update FII/DII flows
5. `hmm.py` — update market regime
6. `features.py` — incremental feature computation
7. `export_features.py` — export updated features
8. `ensemble_final.py` — generate today's signals

All scripts resume from their last ingested date automatically. No manual date flags needed.

**Screener fundamentals** — not in daily pipeline. Run `screener_fundamentals.py` manually once per quarter after earnings season.

**NSE ticker CSV** — re-download from NSE at each SEBI rebalance (March, June, September, December). Replace `files/nifty500_tickers.csv` and update `TIER_X_EXCLUDED` in `config.py` for any new IPOs.

---

_Last updated: May 2026_
