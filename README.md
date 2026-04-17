# AI Hedge Fund Simulator — hedge_v2

**Indian Equity Markets | Nifty 500 Universe | Long/Short Strategy**

> ⚠️ **Update this file every session.** Add an entry to the Session Log at the bottom whenever you complete a meaningful piece of work. This file is the project's memory — treat it like a diary.

---

## What This Project Is

A fully automated, AI-driven equity management platform for Indian stock markets (NSE). The system ingests 15 years of historical market data across 500 Nifty stocks, trains a multi-model ML ensemble, and generates BUY/SELL/HOLD recommendations with intelligent portfolio allocation.

This is `hedge_v2` — the scaled production version. The prototype (`hedge_fund_simulator/`) has been left untouched as a reference.

**Current project phase:** Data pipeline construction. Most ingestion scripts are built. Several critical data-quality issues identified in the Apr 15 audit must be resolved before feature engineering or model training can begin.

---

## Folder Structure

```
hedge_fund_simulator_v2/
├── config.py                    ← Central config (tickers, constraints, table names)
├── .env                         ← Credentials (never on GitHub)
├── requirements.txt             ← All libraries (⚠️ no versions pinned — fix pending)
├── README.md                    ← This file — update every session
│
├── files/
│   ├── nifty500_tickers.csv     ← NSE Nifty 500 watchlist (re-download quarterly)
│   ├── nifty500_sectors.csv     ← NSE official sector classification, all 500 tickers (loaded into config.py as SECTOR_MAP)
│   └── rbi/
│       └── repo_history.csv     ← Repo rate history 2010-present (update manually after each MPC meeting)
│
├── data/
│   ├── db.py                    ← Shared DB utilities (get_engine, save_to_db)
│   ├── setup_db.py              ← Run once on any new machine to create all tables
│   ├── bhavcopy_ingestion.py    ← Layer 1: OHLCV via NSE bhavcopy archives — DONE, VERIFIED
│   ├── indicators.py            ← Layer 4: Technical indicators — DONE, C1 FIXED (Adj_Close)
│   ├── screener_fundamentals.py ← Layer 2: Annual fundamentals from Screener.in — DONE, C2 FIXED
│   ├── macro.py                 ← Layer 3A: yfinance + FRED macro data — DONE, C6 FIXED (UPSERT)
│   ├── fii_dii_stockedge.py     ← Layer 3A: FII+DII monthly+daily via StockEdge API — DONE (replaces fii_dii.py + fii_dii_historical.py)
│   ├── rbi_macro.py             ← Layer 3A: Repo Rate + IIP + Forex — DONE, VERIFIED
│   ├── sentiment.py             ← Layer 3B: News sentiment via FinBERT — NOT BUILT (home laptop only)
│   ├── data_quality.py          ← Stock tier classification A/B/C/X — NOT BUILT (NEXT STEP)
│   ├── features.py              ← Feature engineering — unified ML table — NOT BUILT
│   └── export_features.py       ← Export features_master.csv for Colab — NOT BUILT
│
├── debugging/                   ← Debug scripts (should be in .gitignore)
│   ├── debug_screener.py
│   ├── insure_dbug.py
│   └── new_debug.py
│
├── models/                      ← Saved model files (not on GitHub)
│   └── adj_factors.pkl          ← yfinance adjustment factors (auto-generated)
│
├── risk/                        ← NOT BUILT — no files exist yet
│   └── risk_manager.py          ← TODO
│
└── dashboard/                   ← NOT BUILT — no files exist yet
    └── app.py                   ← TODO
```

---

## Script Run Order

Run these in order on a fresh machine after `python data/setup_db.py`:

```
1.  python data/bhavcopy_ingestion.py      ← DONE. 500 tickers, 1.3M rows, 2010-2026
2.  python data/indicators.py              ← DONE. C1 fixed — runs on Adj_Close
3.  python data/screener_fundamentals.py   ← DONE. C2 fixed — snapshot leak removed
4.  python data/macro.py                   ← DONE. C6 fixed — UPSERT resume
5.  python data/fii_dii_stockedge.py       ← DONE. Run MODE=monthly on 1st trading day of month, MODE=daily after market close daily
6.  python data/rbi_macro.py               ← DONE
7.  python data/sentiment.py               ← NOT BUILT (home laptop only — corporate firewall blocks HuggingFace/GDELT)
8.  python data/data_quality.py            ← NOT BUILT (NEXT STEP)
9.  python data/features.py                ← NOT BUILT
10. python data/export_features.py         ← NOT BUILT
```

> ⚠️ **fii_dii.py and fii_dii_historical.py are retired.** Do not run them. Replaced by fii_dii_stockedge.py.

**Then on Google Colab (in order):**

```
hedge_fund_xgboost.ipynb    ← XGBoost + LightGBM (free T4 GPU)
hedge_fund_lstm.ipynb       ← LSTM (free T4 GPU)
hedge_fund_hmm.ipynb        ← HMM regime detection (CPU only)
hedge_fund_ensemble.ipynb   ← Ensemble combiner + backtest
```

None of the Colab notebooks have been built yet.

---

## Database

- **Database name:** `hedge_v2_db`
- **Credentials:** stored in `.env` file (never committed to GitHub)
- **Tables:** see `data/setup_db.py` for full schema

| Table                      | Purpose                                | Source                      | Status                                                                               |
| -------------------------- | -------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------ |
| nifty500_ohlcv             | Daily OHLCV prices                     | NSE Bhavcopy                | POPULATED                                                                            |
| nifty500_indicators        | Technical indicators                   | Computed from OHLCV         | POPULATED (C1 FIXED)                                                                 |
| nifty500_fundamentals      | Annual P&L, balance sheet              | Screener.in                 | POPULATED (C2 FIXED)                                                                 |
| macro_indicators           | Daily/monthly macro signals            | yfinance + FRED + StockEdge | POPULATED (4246 rows, new FII/DII columns)                                           |
| nifty500_sentiment         | Daily news sentiment                   | FinBERT (planned)           | EMPTY                                                                                |
| stock_data_quality         | Tier A/B/C/X classification            | Computed from OHLCV         | EMPTY                                                                                |
| features_master            | Unified ML training dataset            | All layers combined         | EMPTY                                                                                |
| portfolio_positions        | Active/closed position tracker         | Portfolio engine            | EMPTY                                                                                |
| sector_fundamentals_median | Sector median ratios                   | Imputation fallback         | EMPTY                                                                                |
| corporate_actions          | Split/dividend history                 | yfinance                    | POPULATED (9455 events)                                                              |
| fii_dii_staging            | Staging table from Apr 17 FII overhaul | StockEdge API               | ⚠️ DROP THIS — data migrated to macro_indicators. Run: `DROP TABLE fii_dii_staging;` |

---

## Key Design Decisions

- **Config-driven:** `config.py` is the single source of truth. BANKING_TICKERS, NSE_HOLIDAYS, SECTOR_MAP (from `files/nifty500_sectors.csv`, covers all 500 tickers), and `get_sector()` helper all live here.
- **UPSERT for macro:** `macro.py` uses `ON DUPLICATE KEY UPDATE` — stale/preliminary FRED values (GDP, CPI) are corrected on re-run. All other scripts use `INSERT IGNORE` via `save_to_db()` — never use `.to_sql()` directly.
- **nifty500_sentiment must NEVER be truncated** — historical news cannot be recovered.
- **All ML training on Google Colab** — home laptop (AMD Ryzen 7, 8GB RAM, no GPU) cannot train models.
- **Survivorship bias:** current implementation uses today's Nifty 500 only. Full fix requires NSE historical constituent files — deferred.
- **Fundamentals source:** Screener.in annual data (not quarterly). Historical PE/ROE/ROCE/Dividend_Yield/BVPS are NULLed for all periods except latest — snapshot leak fixed (C2).
- **FII/DII source:** StockEdge CM Provisional API (not NSDL). Covers 2008-present, includes DII, consistent units. `fii_dii_stockedge.py` is the only active FII/DII script. `fii_dii.py` and `fii_dii_historical.py` are retired — do not run them.
- **Adj_Close:** computed from `corporate_actions` splits table (not yfinance ratio). Dividends excluded.

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

## Critical Issues (from Apr 15 audit — must fix before model training)

| #   | Issue                                            | Severity | Status          |
| --- | ------------------------------------------------ | -------- | --------------- |
| C1  | indicators.py uses raw Close, not Adj_Close      | CRITICAL | ✅ FIXED        |
| C2  | PE/ROE/ROCE/DivYield are snapshot, not history   | CRITICAL | ✅ FIXED        |
| C3  | Adj_Close via yfinance ratio — propagates errors | CRITICAL | ✅ FIXED        |
| C5  | FII column has monthly + daily values mixed      | CRITICAL | ✅ FIXED        |
| C6  | macro.py resume never corrects stale rows        | CRITICAL | ✅ FIXED        |
| H1  | features.py / data_quality.py not built          | HIGH     | OPEN            |
| H2  | Tier A/B/C is README prose only, not computed    | HIGH     | OPEN            |
| H3  | Fundamentals are annual, not quarterly           | HIGH     | OPEN (deferred) |
| H4  | Cash derivation for non-banks is wrong           | HIGH     | OPEN            |
| H7  | BE series excluded from bhavcopy parsing         | HIGH     | OPEN            |
| H14 | requirements.txt has zero version pins           | HIGH     | OPEN            |

Full issue list: see `audit_report.txt` in project files.

---

## Known Issues / Deferred Work

| Issue                                | Priority | Notes                                                                                                                              |
| ------------------------------------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Survivorship bias in ticker list     | Medium   | Need NSE historical constituent CSVs                                                                                               |
| Sentiment historical backfill        | Medium   | GDELT bulk API — weeks of calls                                                                                                    |
| Sector map covers 130/500 tickers    | High     | ✅ FIXED — nifty500_sectors.csv now covers all 500                                                                                 |
| OBV uses unadjusted volume           | Medium   | Discontinuity on every split date                                                                                                  |
| VWAP_Daily is actually Typical_Price | Medium   | Mislabeled column — (H+L+C)/3, not volume-weighted                                                                                 |
| BB_Width in schema but not computed  | Low      | Will remain NULL until indicators.py adds it                                                                                       |
| Screener ~124 tickers JS-rendered    | Low      | BeautifulSoup can't parse JS tables. These tickers will have NULL fundamentals — handled via sector median fallback in features.py |
| rbi_macro.py hardcodes repo rate     | Medium   | repo_history.csv created but rbi_macro.py not yet patched to read it                                                               |
| requirements.txt no version pins     | High     | Must pin — yfinance 1.2.1 breaks India_VIX/USDINR/Crude. Pin to yfinance==0.2.48                                                   |

---

## Session Log

> Add a new entry here at the end of every working session. Format: **Date — What was done**

---

### 09 Apr 2026 — Project setup and first scripts

**Status:** hedge_v2 project initialized from scratch.

**Completed this session:**

- Created `hedge_v2/` folder structure alongside `hedge_fund_simulator/` prototype
- `config.py` — reads 500 tickers from NSE CSV (located at `files/nifty500_tickers.csv`), all portfolio constraints from constraints doc, GDP_India and FII/DII added to macro config, fundamentals source correctly set to Screener.in
- `data/db.py` — shared DB utilities pointing to `hedge_v2_db`
- `data/setup_db.py` — 10 tables created including stock_data_quality, portfolio_positions, sector_fundamentals_median, corporate_actions
- `data/bhavcopy_ingestion.py` — downloads 15 years of NSE bhavcopy ZIPs, calculates Adj_Close using yfinance adjustment factor ratio, resume support, survivorship bias limitation documented
- `requirements.txt` — all libraries listed (no versions pinned)
- `nifty500_tickers.csv` — placed in `files/` subfolder

**Key decisions made:**

- Survivorship bias acknowledged but deferred — using current 500 tickers for now
- min_long/short exposure removed from constraints (caps only, no forced floors)
- Fundamentals: Screener.in (not yfinance) confirmed as source
- GDP_India added back properly via correct FRED series ID (was corrupt 999999.9999 in prototype)

**Next session:** Run bhavcopy ingestion, build indicators.py

---

### 10 Apr 2026 — Bhavcopy ingestion verified, config fixed, indicators built

**Status:** Data pipeline Layer 1 complete and verified. Layer 4 (indicators) built.

**Bhavcopy Data Quality (from total_days.csv audit):**

- 500 tickers ingested. Max observed coverage_ratio = 0.9454.
- Tier A threshold corrected from 0.95 → 0.93 in config.py.
- Verified tier distribution (estimated, not from data_quality.py):
  - Tier A: 275 stocks (8+ years, ≥93% coverage)
  - Tier B: 107 stocks (4+ years, ≥75% coverage)
  - Tier C: 85 stocks (below thresholds)
  - Tier X: 33 stocks (<252 days, all recent IPOs)
- TIER_X_EXCLUDED list added to config.py — all 33 tickers hardcoded.
- Note: these tier counts are from bhavcopy verification queries, NOT from data_quality.py (which does not exist yet).

**Changes made:**

- `config.py` — DATA_QUALITY thresholds fixed (0.95→0.93), TIER_X_EXCLUDED list added
- `data/indicators.py` — built for hedge_v2 (nifty500_ohlcv → nifty500_indicators), skips Tier X tickers, uses `ta` library
- ⚠️ **indicators.py computes all indicators on raw `Close`, not `Adj_Close`.** A fixed version `indicators1.py` was created later but is not wired into the pipeline. This is a critical bug — see audit_report.txt C1.

---

### 10 Apr 2026 — Bhavcopy ingestion bugs fixed, data pipeline Layer 1 fully verified

**Status:** Bhavcopy ingestion fully fixed and verified. 500 tickers, 15 years, data up to Apr 9 2026.

**Bugs found and fixed in bhavcopy_ingestion.py:**

- Root cause: NSE discontinued old bhavcopy format on July 8, 2024 (NSE Circular 62424). Old URL format returned 404 for all dates post Jul 5 2024.
- Fix 1 — Added UDiFF format URL: `BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip`
- Fix 2 — Added UDiFF column name mappings (TckrSymb→SYMBOL, OpnPric→OPEN, etc.)
- Fix 3 — Added NSE session cookie refresh every 20 requests
- Fix 4 — Removed `input()` confirmation prompt from verify_ingestion()

**Final verified data state:**

- Min Date: 2010-01-04
- Max Date: 2026-04-09
- Distinct Tickers: 500
- Total Rows: 1,297,850
- Trading days expected: 4,213 | Ingested: 4,010 | Coverage: 95.2%

**Next session:** Build screener_fundamentals.py

---

### 11 Apr 2026 — Screener fundamentals debugged, macro scripts built

**Status:** All data pipeline ingestion scripts built. Screener fundamentals parser debugged through 3 iterations.

---

**indicators.py — built (⚠️ known bug: uses raw Close)**

- Reads nifty500_ohlcv, skips 33 Tier X tickers, computes 20 indicators per stock
- Uses `ta` library (not pandas-ta — incompatible with Python 3.11+)
- Drops warmup rows where RSI/MACD not yet valid
- **BUG: computes all indicators on `Close` instead of `Adj_Close`. A fixed version `indicators1.py` exists using Adj_Close but is not the file referenced in the run order.**

---

**screener_fundamentals.py — three debug iterations, v4 is current**

Three bugs found and fixed:

- Bug 1 — Valuation ratios (PE, ROE, ROCE, Dividend Yield) were 100% NULL. Root cause: these live in `.company-ratios li` elements, not financial tables. Fixed by adding `parse_company_ratios()` function.
- Bug 2 — `get_years()` Windows locale bug with `strptime("%b")`. Fixed with manual MONTH_ORDER dict.
- Bug 3 — Banking/NBFC tickers use different P&L row names. Fixed with separate keyword lists.

**⚠️ KNOWN DATA QUALITY ISSUE:** PE_Ratio, ROE, ROCE, Dividend_Yield, Book_Value_PS are the CURRENT SNAPSHOT from screener.in, applied to EVERY historical year row. This is a look-ahead leak — the model will train on future information for these features. Must be fixed before model training. See audit_report.txt C2.

**⚠️ ANNUAL, NOT QUARTERLY:** The scraper parses Screener's annual P&L/BS/CF tables. The architecture framework claims "10-year quarterly." In reality, one row per year per ticker. Quarterly data would require scraping a different screener section.

Columns permanently NULL (not available from screener): EV_EBITDA, FII_Holding, DII_Holding.

---

**macro.py — built and verified**

- Sources: yfinance (India VIX, USDINR, Crude, Gold) + FRED (GDP, CPI, Fed Rate, US 10Y)
- 4242 rows in macro_indicators, date range 2010-01-01 → 2026-04-10
- GDP_India: 64 distinct quarterly values, correct range — prototype 999999.9999 bug fixed
- Resume support: checks latest date in DB, re-fetches last 5 days. **Caveat:** INSERT IGNORE means stale/preliminary values are never corrected on re-run.

**fii_dii.py — built**

- Source: NSE public API endpoint (fiidiiTradeReact)
- Returns last ~30 days only. Historical backfill via fii_dii_historical.py.
- SQLAlchemy 2.x text() wrapper fix applied.

**rbi_macro.py — built**

- Repo Rate: hardcoded MPC history (RBI has no public API). Current rate: 5.25%.
- IIP Growth: FRED INDPROINDMISMEI (YoY % computed in script)
- Forex Reserves: FRED TRESEGINM052N (USD mn ÷ 1000 = USD bn)

**features.py — NOT BUILT**

Session log previously mentioned "features.py ... sector median fallback loop is unvectorized." This was describing planned design, not existing code. features.py does not exist in the repo.

---

### 14-15 Apr 2026 — Macro pipeline verification + FII/DII historical data

**Status:** macro_indicators table fully populated. FII historical data loaded. rbi_macro.py verified.

---

**macro.py — VERIFIED**

- 4242 rows, 2010-01-01 → 2026-04-10
- India_VIX: 248 NULLs (weekends/holidays — acceptable, forward-filled in features.py)
- Crude_Oil/Gold: ~150 NULLs each (non-US trading day gaps)
- GDP_India: correct values, quarterly forward-fill confirmed
- Repo_Rate, IIP_Growth, Forex_Reserves_USD: populated by rbi_macro.py
- FII_Net_Buy_Cr: populated by fii_dii_historical.py

---

**fii_dii_historical.py — BUILT AND VERIFIED**

- Source: NSDL FPI portal (.xls files, one per calendar year)
- Files placed in `files/fii_nsdl/`
- Auto-detects year from column header, handles pre/post 2022 formats
- **⚠️ UNIT MISMATCH:** NSDL provides monthly totals. Each trading day in month M gets the full month M total. But fii_dii.py writes actual daily values for recent dates. Same column (`FII_Net_Buy_Cr`) contains two different measurement units. See audit_report.txt C5.
- Result: 192 monthly records parsed, 4171 trading days updated

---

**rbi_macro.py — rewritten and verified**

Sources changed from RBI DBIE (dead URLs) to:

| Column             | Source                                 |
| ------------------ | -------------------------------------- |
| Repo_Rate          | Hardcoded MPC history in script        |
| IIP_Growth         | FRED: INDPROINDMISMEI (YoY % computed) |
| Forex_Reserves_USD | FRED: TRESEGINM052N (USD mn ÷ 1000)    |

Repo_Rate stale-row fix (run in MySQL Workbench):

```sql
UPDATE macro_indicators SET Repo_Rate = 5.75 WHERE Date BETWEEN '2025-06-06' AND '2025-09-04';
UPDATE macro_indicators SET Repo_Rate = 5.50 WHERE Date BETWEEN '2025-09-05' AND '2025-12-04';
UPDATE macro_indicators SET Repo_Rate = 5.25 WHERE Date >= '2025-12-05';
```

---

**DII_Net_Buy_Cr — investigated, deferred**

No free bulk historical source exists for 15-year DII data. Remains NULL for historical period. Deferred until after first model training cycle confirms whether DII improves IC.

---

### 15 Apr 2026 — Critical codebase audit

**Status:** Full audit of all v2 source files completed. 5 critical bugs, 16 high-severity issues, 21 medium issues identified. See `audit_report.txt` for complete findings.

**Summary of critical findings:**

1. **C1 — indicators.py uses raw Close.** indicators1.py is the fixed version but not wired in. All indicator data in DB is unreliable.
2. **C2 — PE/ROE/ROCE/DivYield are current snapshot on every historical year.** Classic look-ahead leak.
3. **C3 — Adj_Close via yfinance ratio propagates yfinance errors into NSE data.** Should use splits from corporate_actions directly.
4. **C5 — FII column has monthly totals (historical) and daily values (recent) mixed.** Unit mismatch will confuse models.
5. **C6 — macro.py resume with INSERT IGNORE never corrects stale rows.**

**Remediation plan (Phase A — must complete before any further pipeline work):**

1. Delete indicators.py, rename indicators1.py → indicators.py
2. TRUNCATE nifty500_indicators, do not re-run until adj factor fix
3. NULL out snapshot PE/ROE/ROCE/DivYield/BVPS for all but latest period per ticker
4. Rebuild adjustment factors from corporate_actions (splits only)
5. Split FII into FII_Monthly_Net_Cr + FII_Daily_Net_Cr (or divide monthly by trading days)
6. Pin requirements.txt via pip freeze

**Next session:** Begin Phase A remediation.

### 15 Apr 2026 — Phase A remediation: all critical bugs fixed

**Status:** All 5 critical audit issues (C1–C6) resolved. Data pipeline is now clean for feature engineering.

**Changes made:**

- `indicators.py` — replaced with fixed version (was `indicators1.py`). Now computes all indicators on `Adj_Close`. Old `indicators1.py` deleted. `nifty500_indicators` truncated and recomputed. **C1 FIXED.**
- `screener_fundamentals.py` (v5) — PE_Ratio, ROE, ROCE, Dividend_Yield, Book_Value_PS NULLed for all periods except latest per ticker. Snapshot look-ahead leak eliminated. **C2 FIXED.**
- `bhavcopy_ingestion.py` — Adj_Close rebuilt from `corporate_actions` splits table (backward walk, cumulative ratio, bisect O(log N) lookup). Homepage cookie refresh calls removed (were causing 15s timeout waste per 20 iterations). Per-file sleep reduced 0.5s → 0.1s. **C3 FIXED.**
- `macro.py` — UPSERT resume (`ON DUPLICATE KEY UPDATE`). NaN→None fix for DB writes. **C6 FIXED.**
- `setup_db.py` — `macro_indicators` schema updated: `FII_Net_Buy_Cr` replaced with `FII_Monthly_Net_Cr` + `FII_Daily_Net_Cr`. Auto-migration function added.
- `config.py` — NSE_HOLIDAYS moved from bhavcopy_ingestion.py; BANKING_TICKERS moved from screener_fundamentals.py; SECTOR_MAP now loaded from `files/nifty500_sectors.csv` (covers all 500 tickers, replaces hardcoded 130-ticker fallback); `get_sector()` helper added; RBI_REPO_CSV path added.
- `files/nifty500_sectors.csv` — NEW. NSE official sector classification, all 500 tickers.
- `files/rbi/repo_history.csv` — NEW. Repo rate history 2010-2026. Update manually after each MPC meeting. Note: `rbi_macro.py` not yet patched to read this file.

**FII split (C5 partial fix — superseded by Apr 17 session):**
`FII_Monthly_Net_Cr` = monthly total ÷ trading days in month. `FII_Daily_Net_Cr` = actual NSE API daily values (~30 days). `fii_dii_historical.py` and `fii_dii.py` updated to write to respective columns.

**Known issue at end of session:** yfinance 1.2.1 installed — breaks `^INDIAVIX`, `INR=X`, `CL=F`. Fix: `pip install yfinance==0.2.48` then rerun `macro.py`. Must pin in `requirements.txt`.

**Next session:** FII/DII overhaul + data_quality.py.

---

### 17 Apr 2026 — FII/DII pipeline overhaul, StockEdge source adopted

**Status:** FII/DII data fully replaced with clean StockEdge CM Provisional source. All 5 critical bugs now closed. Next step: `data_quality.py`.

**Problem with old approach:**

- `fii_dii_historical.py` (NSDL): monthly totals divided by trading days — approximation, still not true daily data.
- `fii_dii.py` (NSE API): only last 30 days; DII column was NULL for all history.
- Both scripts retired permanently.

**New source — StockEdge CM Provisional API:**

- Endpoint: `https://api.stockedge.com/Api/FIIDashboardApi/GetFIIDIIProvisional`
- No authentication. Public API discovered via DevTools.
- `TimeSpan=M`: monthly FII+DII back to Jan 2008. `TimeSpan=D`: daily last ~50 trading days.
- NSE CM Provisional data — same-day estimates used by the market. Correct for regime detection.
- Verified: Mar 2020 FII = -65,816.70 Cr (COVID selloff), DII = +55,595.18 Cr (bought during crash). ✅

**New script: `data/fii_dii_stockedge.py`**

- MODE 1 (monthly): fetches all months back to 2008, stamps monthly total on every trading day via `nifty500_ohlcv` join. Resume logic skips months already in DB.
- MODE 2 (daily): fetches last ~50 days of actual daily values. Resume logic skips dates already in DB.
- `FII_Source_Flag` = `'monthly'` or `'daily'` — tells `features.py` data resolution per row.
- Run schedule: MODE 1 on 1st trading day of each month; MODE 2 daily after market close.

**Schema changes to `macro_indicators`:**

| Column             | Change                                                         |
| ------------------ | -------------------------------------------------------------- |
| FII_Monthly_Net_Cr | Monthly FII net flow (same value on all trading days in month) |
| DII_Monthly_Net_Cr | Monthly DII net flow — NEW, was not available before           |
| FII_Daily_Net_Cr   | Actual daily FII — last ~50 trading days                       |
| DII_Daily_Net_Cr   | Actual daily DII — NEW                                         |
| FII_Source_Flag    | `'monthly'` or `'daily'`                                       |
| DII_Net_Buy_Cr     | DROPPED — old broken NSDL column                               |

**Staging and migration:**

- Data first written to `fii_dii_staging` table for verification against NSDL source.
- Migration SQL run: `UPDATE macro_indicators JOIN fii_dii_staging ON Date`.
- All values confirmed in `macro_indicators`. **⚠️ `fii_dii_staging` table can now be dropped:**
  ```sql
  DROP TABLE fii_dii_staging;
  ```

**C5 FIXED.** All FII/DII audit issues closed.

**Next step:** Build `data/data_quality.py` — classifies all 500 tickers into Tier A/B/C/X based on `nifty500_ohlcv` coverage, gap%, ADV. Populates `stock_data_quality` table. Pending decision: F&O list source (CSV or fetch from NSE).

17 Apr 2026 — data_quality.py built and verified
Status: stock_data_quality table fully populated. Tier classification operational. Next step: features.py.
Completed this session:

Downloaded fo_mktlots.csv from NSE (Permitted lot size file) — saved as files/fo_list.csv. Added FO_LIST_CSV path to config.py.
Built data/data_quality.py — classifies all 500 tickers into Tier A/B/C/X based on OHLCV coverage, gap%, 90-day ADV, and F&O eligibility. Populates stock_data_quality table. Safe to re-run (TRUNCATE + reinsert).

Verified results:
TierCountCriteriaA292≥8 years, ≥93% coverage, ADV ≥5 CrB90≥4 years, ≥75% coverage, ADV ≥1 CrC85Below Tier B thresholdsX33<252 trading days (recent IPOs)

F&O listed: 213 / 500
Avg coverage (Tier A): 14.6 years
Spot checks confirmed: HDFCBANK/RELIANCE/ICICIBANK at correct ADV and Tier A; ETERNAL.NS and recent IPOs correctly Tier X.

Key decisions:

TIER_X_EXCLUDED hardcoded list in config.py is now superseded by the stock_data_quality table. All downstream scripts must query the DB for tier classification — do not use the hardcoded list.
Tier A count (292) is higher than the 275 estimated in the Apr 10 session log — the earlier estimate was from a manual bhavcopy query without ADV filtering. The new count from combined coverage + ADV logic is the authoritative figure.

_End of session log. Add your next entry above this line._
