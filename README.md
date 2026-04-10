# AI Hedge Fund Simulator — hedge_v2

**Indian Equity Markets | Nifty 500 Universe | Long/Short Strategy**

> ⚠️ **Update this file every session.** Add an entry to the Session Log at the bottom whenever you complete a meaningful piece of work. This file is the project's memory — treat it like a diary.

---

## What This Project Is

A fully automated, AI-driven equity management platform for Indian stock markets (NSE). The system ingests 15 years of historical market data across 500 Nifty stocks, trains a multi-model ML ensemble, and generates BUY/SELL/HOLD recommendations with intelligent portfolio allocation.

This is `hedge_v2` — the scaled production version. The prototype (`hedge_fund_simulator/`) has been left untouched as a reference.

---

## Folder Structure

```
hedge_v2/
├── config.py                    ← THE only file to change when scaling
├── .env                         ← credentials (never on GitHub)
├── requirements.txt             ← all libraries
├── nifty500_tickers.csv         ← NSE Nifty 500 watchlist (re-download quarterly)
├── README.md                    ← this file — update every session
│
├── data/
│   ├── db.py                    ← shared DB utilities (get_engine, save_to_db)
│   ├── setup_db.py              ← run once on any new machine to create all tables
│   ├── bhavcopy_ingestion.py    ← Layer 1: OHLCV via NSE bhavcopy archives
│   ├── indicators.py            ← Layer 4: Technical indicators (TODO)
│   ├── screener_fundamentals.py ← Layer 2: Fundamentals from Screener.in (TODO)
│   ├── macro.py                 ← Layer 3A: Macro data (TODO)
│   ├── fii_dii.py               ← Layer 3A: FII/DII daily flows from NSE (TODO)
│   ├── rbi_macro.py             ← Layer 3A: RBI DBIE data (TODO)
│   ├── sentiment.py             ← Layer 3B: News sentiment via FinBERT (TODO)
│   ├── data_quality.py          ← Stock tier classification A/B/C (TODO)
│   └── features.py              ← Feature engineering — unified ML table (TODO)
│
├── models/                      ← Saved model files (not on GitHub)
│   ├── adj_factors.pkl          ← yfinance adjustment factors (auto-generated)
│   └── ...
│
├── risk/
│   └── risk_manager.py          ← All position/sector/portfolio limits (TODO)
│
└── dashboard/
    └── app.py                   ← Streamlit dashboard (TODO)
```

---

## Script Run Order

Run these in order on a fresh machine after `python data/setup_db.py`:

```
1.  python data/bhavcopy_ingestion.py      ← ~30-45 mins (15 years of data)
2.  python data/indicators.py
3.  python data/screener_fundamentals.py   ← run during/after bhavcopy
4.  python data/macro.py
5.  python data/fii_dii.py
6.  python data/rbi_macro.py
7.  python data/sentiment.py               ← HOME LAPTOP ONLY (corporate firewall blocks on work laptop)
8.  python data/data_quality.py
9.  python data/features.py
10. python data/export_features.py         ← exports features_master.csv for Colab
```

**Then on Google Colab (in order):**

```
hedge_fund_xgboost.ipynb    ← XGBoost + LightGBM (free T4 GPU)
hedge_fund_lstm.ipynb       ← LSTM (free T4 GPU)
hedge_fund_hmm.ipynb        ← HMM regime detection (CPU only)
hedge_fund_ensemble.ipynb   ← Ensemble combiner + backtest
```

---

## Database

- **Database name:** `hedge_v2_db`
- **Credentials:** stored in `.env` file (never committed to GitHub)
- **Tables:** see `data/setup_db.py` for full schema

| Table                      | Purpose                        | Source                             |
| -------------------------- | ------------------------------ | ---------------------------------- |
| nifty500_ohlcv             | Daily OHLCV prices             | NSE Bhavcopy                       |
| nifty500_indicators        | Technical indicators           | Computed from OHLCV                |
| nifty500_fundamentals      | Quarterly P&L, balance sheet   | Screener.in                        |
| macro_indicators           | Daily/monthly macro signals    | yfinance + FRED + NSE + RBI        |
| nifty500_sentiment         | Daily news sentiment           | GDELT + NewsAPI + ET RSS → FinBERT |
| stock_data_quality         | Tier A/B/C classification      | Computed from OHLCV                |
| features_master            | Unified ML training dataset    | All layers combined                |
| portfolio_positions        | Active/closed position tracker | Portfolio engine                   |
| sector_fundamentals_median | Sector median ratios           | Computed for imputation fallback   |
| corporate_actions          | Split/dividend history         | yfinance                           |

---

## Key Design Decisions

- **Config-driven:** `config.py` is the single file changed when scaling. All table names, tickers, constraints live there.
- **INSERT IGNORE everywhere:** via `save_to_db()` — never use `.to_sql()` directly
- **nifty500_sentiment must NEVER be truncated** — historical news cannot be recovered
- **All ML training on Google Colab** — home laptop (AMD Ryzen 7, 8GB RAM, no GPU) cannot train models
- **Survivorship bias:** current implementation uses today's Nifty 500 only. Full fix requires NSE historical constituent files — deferred to production phase.
- **Fundamentals source:** Screener.in (not yfinance) — yfinance is unreliable for Indian quarterly data at scale

---

## Portfolio Constraints (from Constraints doc)

| Constraint                | Value                                      |
| ------------------------- | ------------------------------------------ |
| Min stocks in portfolio   | 30 (regime-dependent)                      |
| Max stocks in portfolio   | 55 (hard cap 60)                           |
| Max position size         | 5% of NAV                                  |
| Max sector long exposure  | 25% of NAV                                 |
| Max sector short exposure | 15% of NAV (cap only — no forced minimum)  |
| Max long book             | 120% of NAV                                |
| Max short book            | 20% of NAV (can be 0% if no valid signals) |
| Min cash buffer           | 8% always                                  |
| Long stop-loss            | -15% from entry                            |
| Short stop-loss           | +10% against position                      |

---

## Known Issues / Deferred Work

| Issue                             | Priority | Notes                                |
| --------------------------------- | -------- | ------------------------------------ |
| Survivorship bias in ticker list  | Medium   | Need NSE historical constituent CSVs |
| Sentiment historical backfill     | Medium   | GDELT bulk API — weeks of calls      |
| Screener.in scraper not yet built | High     | Next after bhavcopy ingestion        |
| RBI DBIE has no clean API         | Medium   | Manual CSV download approach         |
| FII/DII script not yet built      | High     | NSE API endpoint available           |

---

## Session Log

> Add a new entry here at the end of every working session. Format: **Date — What was done**

---

### 09 Apr 2026 — Project setup and first scripts

**Status:** hedge_v2 project initialized from scratch.

**Completed this session:**

- Created `hedge_v2/` folder structure alongside `hedge_fund_simulator/` prototype
- `config.py` — reads 500 tickers from NSE CSV automatically, all portfolio constraints from constraints doc, GDP_India and FII/DII added to macro config, fundamentals source correctly set to Screener.in
- `data/db.py` — shared DB utilities pointing to `hedge_v2_db`
- `data/setup_db.py` — all 8 tables created including 3 new ones (stock_data_quality, portfolio_positions, sector_fundamentals_median), FII/DII and GDP_India columns added to macro_indicators and features_master
- `data/bhavcopy_ingestion.py` — downloads 15 years of NSE bhavcopy ZIPs, calculates Adj_Close using yfinance adjustment factors, resume support, survivorship bias limitation documented
- `requirements.txt` — all libraries listed
- `nifty500_tickers.csv` — copied into project folder (re-download from NSE quarterly)

**Key decisions made:**

- Survivorship bias acknowledged but deferred — using current 500 tickers for now
- min_long/short exposure removed from constraints (caps only, no forced floors)
- Fundamentals: Screener.in (not yfinance) confirmed as source
- GDP_India added back properly via correct FRED series ID (was corrupt 999999.9999 in prototype)

**Next session:** Build `data/indicators.py`, `data/screener_fundamentals.py`, `data/macro.py`

---

---

### 10 Apr 2026 — Bhavcopy ingestion verified, config fixed, indicators built

**Status:** Data pipeline Layer 1 complete and verified. Layer 4 (indicators) built.

**Bhavcopy Data Quality (from total_days.csv audit):**

- 500 tickers ingested. Max observed coverage_ratio = 0.9454 (hard ceiling from
  NSE holidays not in the holidays list — data is correct, threshold was wrong).
- Tier A threshold corrected from 0.95 → 0.93 in config.py.
- Verified tier distribution:
  - Tier A: 275 stocks (8+ years, ≥93% coverage) — full model suite
  - Tier B: 107 stocks (4+ years, ≥75% coverage) — reduced model suite
  - Tier C: 85 stocks (below thresholds) — signal inheritance from sector
  - Tier X: 33 stocks (<252 days, all recent IPOs) — excluded from all training
- No stock has coverage below 0.776. Zero data corruption found.
- TIER_X_EXCLUDED list added to config.py — all 33 tickers hardcoded.

**Changes made:**

- `config.py` — DATA_QUALITY thresholds fixed (0.95→0.93), max_gap_pct replaced
  by min_coverage, TIER_X_EXCLUDED list added
- `data/indicators.py` — built for hedge_v2 (nifty500_ohlcv → nifty500_indicators),
  skips Tier X tickers, uses `ta` library

**Next session:** Run `python data/indicators.py`, then build `data/screener_fundamentals.py`

_End of session log. Add your next entry above this line._
