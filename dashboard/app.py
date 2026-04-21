# dashboard/app.py
"""
AI Hedge Fund Simulator v2 — Streamlit Dashboard
=================================================
Run from project root:
    streamlit run dashboard/app.py

Reads:
    - exports/model_output/signals_*.csv          (ensemble output)
    - MySQL hedge_v2_db                            (all pipeline tables)
    - files/nifty500_sectors.csv                  (sector map)
    - risk/risk_manager.py                         (pre-trade checks)

All DB access is read-only. No writes.
"""

import glob
import os
import subprocess
import sys
import warnings
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Path setup ───────────────────────────────────────────────────────────────
_DASH_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_DASH_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# ── Config import ─────────────────────────────────────────────────────────────
try:
    import config as cfg
    DB_URL       = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST', 'localhost')}/{os.getenv('DB_NAME', 'hedge_v2_db')}"
    TABLES       = cfg.TABLES
    SECTORS_CSV  = os.path.join(_PROJECT_ROOT, "files", "nifty500_sectors.csv")
    SIGNALS_DIR  = os.path.join(_PROJECT_ROOT, "exports", "model_output")
    SCRIPTS_DIR  = os.path.join(_PROJECT_ROOT, "data")
except Exception as e:
    st.error(f"Config import failed: {e}")
    st.stop()

# ── Risk manager import ───────────────────────────────────────────────────────
try:
    from risk.risk_manager import (
        PortfolioState, OpenPosition, check_position, screen_signals,
        MAX_SINGLE_CORE_LONG_PCT, MAX_SINGLE_MIDCAP_LONG_PCT,
        MAX_SINGLE_SHORT_PCT, MAX_SECTOR_GROSS_LONG_PCT,
        MAX_SECTOR_GROSS_LONG_FIN_PCT, MAX_SECTOR_GROSS_SHORT_PCT,
        GROSS_EXPOSURE_CAP_PCT, NET_EXPOSURE_MIN_PCT, NET_EXPOSURE_MAX_PCT,
        CASH_RESERVE_MIN_PCT, MIDCAP_LONG_BOOK_MAX, FINANCIAL_SECTOR_NAMES,
        MIN_POSITION_SIZE_PCT,
    )
    from portfolio.optimizer import optimize_portfolio, OptimizedPortfolio
    RISK_AVAILABLE = True
except Exception as e:
    RISK_AVAILABLE = False
    st.warning(f"Risk manager / optimizer not available: {e}")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hedge Fund Simulator v2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark financial terminal aesthetic */
    [data-testid="stAppViewContainer"] { background-color: #0a0e1a; }
    [data-testid="stSidebar"]          { background-color: #0d1117; border-right: 1px solid #1e2d40; }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #0d1117;
        border: 1px solid #1e2d40;
        border-radius: 6px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"]  { color: #58a6ff !important; font-size: 1.4rem !important; font-family: 'JetBrains Mono', monospace; }
    [data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
    [data-testid="stMetricDelta"]  { font-size: 0.8rem !important; }

    /* Tab styling */
    /* Single tab bar — remove Streamlit default red underline */
    [data-testid="stTabs"] { border-bottom: none !important; }
    [data-testid="stTabs"] > div:first-child {
        border-bottom: 1px solid #1e2d40 !important;
    }
    button[data-baseweb="tab"] {
        color: #8b949e !important;
        font-size: 0.85rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        border-bottom: 2px solid transparent !important;
        background: transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }
    /* Hide the sliding red indicator */
    [data-testid="stTabs"] [role="tablist"] > div[data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Tables */
    [data-testid="stDataFrame"] { border: 1px solid #1e2d40 !important; border-radius: 6px; }
    thead tr th { background-color: #161b22 !important; color: #8b949e !important;
                  font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
    tbody tr:hover { background-color: #1a2233 !important; }

    /* Section headers */
    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #58a6ff;
        border-bottom: 1px solid #1e2d40;
        padding-bottom: 6px;
        margin: 20px 0 12px 0;
    }

    /* Signal badges */
    .badge-buy  { background:#0d4429; color:#3fb950; padding:2px 10px;
                  border-radius:4px; font-size:0.75rem; font-weight:600;
                  font-family:'JetBrains Mono',monospace; }
    .badge-sell { background:#3d0c10; color:#f85149; padding:2px 10px;
                  border-radius:4px; font-size:0.75rem; font-weight:600;
                  font-family:'JetBrains Mono',monospace; }
    .badge-hold { background:#1c2128; color:#8b949e; padding:2px 10px;
                  border-radius:4px; font-size:0.75rem; font-weight:600;
                  font-family:'JetBrains Mono',monospace; }

    /* Regime pill */
    .regime-bull    { background:#0d4429; color:#3fb950; }
    .regime-bear    { background:#3d0c10; color:#f85149; }
    .regime-highvol { background:#3d2d00; color:#e3b341; }
    .regime-side    { background:#1c2128; color:#8b949e; }
    .regime-pill {
        display:inline-block; padding:3px 14px; border-radius:20px;
        font-family:'JetBrains Mono',monospace; font-size:0.8rem; font-weight:600;
    }

    /* Risk gauge bar */
    .gauge-wrap { margin: 4px 0 10px 0; }
    .gauge-bg   { background:#1e2d40; border-radius:4px; height:8px; width:100%; }
    .gauge-fill { height:8px; border-radius:4px; transition:width 0.4s ease; }

    /* Progress override */
    .stProgress > div > div { background-color: #1e2d40 !important; }

    /* Buttons */
    .stButton > button {
        background-color: #161b22 !important;
        color: #58a6ff !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    .stButton > button:hover {
        border-color: #58a6ff !important;
        background-color: #0d2043 !important;
    }

    /* Global text */
    h1, h2, h3, p, li, span, div { color: #c9d1d9; }
    .stMarkdown p { color: #c9d1d9; }

    /* Number input */
    [data-testid="stNumberInput"] input { background:#161b22 !important; color:#c9d1d9 !important;
        border:1px solid #30363d !important; border-radius:6px !important; }

    /* Divider */
    hr { border-color: #1e2d40 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (all cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_engine():
    from sqlalchemy import create_engine
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
    url = (
        f"mysql+mysqlconnector://"
        f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST','localhost')}/{os.getenv('DB_NAME','hedge_v2_db')}"
    )
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_sectors() -> pd.DataFrame:
    """
    Load sector + company name from nifty500_sectors.csv.
    Returns DataFrame with columns: Ticker, Sector, Company_Name.
    Robust to all common NSE CSV column name variants.
    """
    try:
        df = pd.read_csv(SECTORS_CSV)
        df.columns = [c.strip() for c in df.columns]

        # ── Ticker column ──────────────────────────────────────────────────
        ticker_col = None
        for cand in ("Symbol","symbol","Ticker","ticker","SYMBOL","NSE Symbol","NSE_Symbol"):
            if cand in df.columns:
                ticker_col = cand
                break
        if ticker_col is None:
            # Last resort: first column
            ticker_col = df.columns[0]
        df = df.rename(columns={ticker_col: "Ticker"})
        df["Ticker"] = df["Ticker"].str.strip()

        # Ensure .NS suffix present for merge with signals (which use .NS tickers)
        df["Ticker"] = df["Ticker"].apply(
            lambda x: x if str(x).endswith(".NS") else str(x) + ".NS"
        )

        # ── Sector column ──────────────────────────────────────────────────
        sector_col = None
        for cand in ("Sector","sector","Industry","industry","SECTOR",
                     "Sector Name","sector_name","Industry Name"):
            if cand in df.columns:
                sector_col = cand
                break
        if sector_col:
            df = df.rename(columns={sector_col: "Sector"})
        else:
            df["Sector"] = "Unknown"

        # ── Company name column ────────────────────────────────────────────
        name_col = None
        for cand in ("Company Name","Company_Name","company_name","Name","name",
                     "COMPANY NAME","Issuer Name","issuer_name","Security Name"):
            if cand in df.columns:
                name_col = cand
                break
        if name_col:
            df = df.rename(columns={name_col: "Company_Name"})
        else:
            df["Company_Name"] = df["Ticker"].str.replace(".NS","",regex=False)

        out_cols = ["Ticker","Sector","Company_Name"]
        return df[out_cols].drop_duplicates("Ticker")

    except Exception as e:
        return pd.DataFrame(columns=["Ticker","Sector","Company_Name"])


@st.cache_data(ttl=300, show_spinner=False)
def load_signals() -> tuple[pd.DataFrame, str]:
    """Load latest signals CSV from exports/model_output/. Returns (df, filepath)."""
    pattern = os.path.join(SIGNALS_DIR, "signals_*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame(), ""
    latest = files[-1]
    df = pd.read_csv(latest, parse_dates=["Date"])
    return df, latest



@st.cache_data(ttl=300, show_spinner=False)
def load_macro_latest() -> pd.Series:
    """Latest row from macro_indicators."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(
                f"SELECT * FROM {TABLES.get('macro','macro_indicators')} "
                f"ORDER BY Date DESC LIMIT 1",
                conn
            )
        return df.iloc[0] if len(df) else pd.Series(dtype=object)
    except Exception:
        return pd.Series(dtype=object)


@st.cache_data(ttl=300, show_spinner=False)
def load_latest_prices(tickers: tuple) -> pd.DataFrame:
    """Latest Adj_Close + Volume from nifty500_ohlcv for given tickers."""
    if not tickers:
        return pd.DataFrame()
    try:
        engine = get_engine()
        ticker_list = ", ".join(f"'{t}'" for t in tickers)
        with engine.connect() as conn:
            df = pd.read_sql(
                f"""
                SELECT o.Ticker, o.Date, o.Adj_Close, o.Volume
                FROM {TABLES.get('ohlcv','nifty500_ohlcv')} o
                INNER JOIN (
                    SELECT Ticker, MAX(Date) AS MaxDate
                    FROM {TABLES.get('ohlcv','nifty500_ohlcv')}
                    WHERE Ticker IN ({ticker_list})
                    GROUP BY Ticker
                ) m ON o.Ticker = m.Ticker AND o.Date = m.MaxDate
                """,
                conn
            )
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_prices_on_date(tickers: tuple, as_of_date: str) -> pd.DataFrame:
    """
    Adj_Close + Volume from nifty500_ohlcv for given tickers on or before as_of_date.
    Used to show prices as of the signal date (handles historical signal files).
    """
    if not tickers:
        return pd.DataFrame()
    try:
        engine = get_engine()
        ticker_list = ", ".join(f"'{t}'" for t in tickers)
        with engine.connect() as conn:
            df = pd.read_sql(
                f"""
                SELECT o.Ticker, o.Date, o.Adj_Close, o.Volume
                FROM {TABLES.get('ohlcv','nifty500_ohlcv')} o
                INNER JOIN (
                    SELECT Ticker, MAX(Date) AS MaxDate
                    FROM {TABLES.get('ohlcv','nifty500_ohlcv')}
                    WHERE Ticker IN ({ticker_list})
                      AND Date <= '{as_of_date}'
                    GROUP BY Ticker
                ) m ON o.Ticker = m.Ticker AND o.Date = m.MaxDate
                """,
                conn
            )
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_pipeline_last_dates() -> dict:
    """Get last ingested date per table for the refresh panel."""
    tables_to_check = {
        "OHLCV":         (TABLES.get("ohlcv",      "nifty500_ohlcv"),       "Date"),
        "Indicators":    (TABLES.get("indicators",  "nifty500_indicators"),  "Date"),
        "Fundamentals":  (TABLES.get("fundamentals","nifty500_fundamentals"),"Period"),
        "Macro":         (TABLES.get("macro",       "macro_indicators"),     "Date"),
        "Sentiment":     (TABLES.get("sentiment",   "nifty500_sentiment"),   "Date"),
        "Data Quality":  (TABLES.get("data_quality","stock_data_quality"),   "Tier_Assigned_Date"),
        "Features":      (TABLES.get("features",    "features_master"),      "Date"),
    }
    result = {}
    try:
        engine = get_engine()
        with engine.connect() as conn:
            for label, (table, date_col) in tables_to_check.items():
                try:
                    row = pd.read_sql(
                        f"SELECT MAX({date_col}) AS last_date FROM {table}", conn
                    )
                    result[label] = str(row.iloc[0,0])[:10] if row.iloc[0,0] else "—"
                except Exception:
                    result[label] = "—"
    except Exception:
        result = {k: "—" for k in tables_to_check}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

REGIME_MAP   = {0: "Bull", 1: "Bear", 2: "High Vol", 3: "Sideways"}
REGIME_CLASS = {0: "regime-bull", 1: "regime-bear", 2: "regime-highvol", 3: "regime-side"}
TIER_LABEL   = {1: "A", 2: "B", 3: "C"}

def signal_badge(val: int) -> str:
    if val == 1:  return '<span class="badge-buy">BUY</span>'
    if val == -1: return '<span class="badge-sell">SELL</span>'
    return '<span class="badge-hold">HOLD</span>'

def regime_pill(regime_int: int) -> str:
    label = REGIME_MAP.get(regime_int, "Unknown")
    cls   = REGIME_CLASS.get(regime_int, "regime-side")
    return f'<span class="regime-pill {cls}">{label}</span>'

def gauge_html(pct: float, cap: float, label: str) -> str:
    """Render a mini progress bar with colour based on utilisation."""
    util  = min(pct / cap, 1.0) if cap > 0 else 0
    width = util * 100
    if util < 0.75:   color = "#3fb950"
    elif util < 0.90: color = "#e3b341"
    else:             color = "#f85149"
    return f"""
    <div style='margin-bottom:10px'>
      <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
        <span style='font-size:0.72rem;color:#8b949e;font-family:monospace'>{label}</span>
        <span style='font-size:0.72rem;color:{color};font-family:monospace'>{pct:.1%} / {cap:.0%}</span>
      </div>
      <div class='gauge-bg'>
        <div class='gauge-fill' style='width:{width:.1f}%;background:{color}'></div>
      </div>
    </div>
    """


def run_optimization(signals_today, sector_map, nav, regime_int):
    """
    Wrapper around optimize_portfolio() for dashboard use.
    Returns (OptimizedPortfolio | None).
    """
    if not RISK_AVAILABLE or signals_today.empty:
        return None
    try:
        engine = get_engine()
        result = optimize_portfolio(
            signals_df  = signals_today,
            nav         = nav,
            regime_int  = regime_int,
            engine      = engine,
            sector_map_df = sector_map,
        )
        return result
    except Exception as e:
        st.error(f"Optimizer error: {e}")
        return None


def run_script(script_name: str, args: list[str] = None) -> tuple[bool, str]:
    """Run a pipeline script as subprocess. Returns (success, output)."""
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, script_name)]
    if args:
        cmd.extend(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            cwd=_PROJECT_ROOT, timeout=3600
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timed out after 60 minutes."
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px 0'>
      <div style='font-family:monospace;font-size:1.1rem;color:#58a6ff;font-weight:700;
                  letter-spacing:0.04em'>AI HEDGE FUND</div>
      <div style='font-family:monospace;font-size:0.65rem;color:#8b949e;
                  letter-spacing:0.14em;text-transform:uppercase'>Simulator v2 · NSE Nifty 500</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # NAV input
    st.markdown('<p class="section-header">Simulation NAV</p>', unsafe_allow_html=True)
    nav_input = st.number_input(
        "Starting NAV (₹)",
        min_value=100_000,
        max_value=1_000_000_000,
        value=10_000_000,
        step=1_000_000,
        format="%d",
        help="Used for position sizing and allocation calculations. Does not affect signal generation.",
        label_visibility="collapsed",
    )
    st.markdown(
        f"<div style='font-family:monospace;font-size:0.8rem;color:#58a6ff'>₹{nav_input:,.0f}</div>",
        unsafe_allow_html=True
    )

    st.divider()

    # Pipeline status
    st.markdown('<p class="section-header">Pipeline Status</p>', unsafe_allow_html=True)
    last_dates = load_pipeline_last_dates()
    for label, last in last_dates.items():
        col1, col2 = st.columns([3, 2])
        col1.markdown(
            f"<span style='font-size:0.72rem;color:#8b949e'>{label}</span>",
            unsafe_allow_html=True
        )
        col2.markdown(
            f"<span style='font-size:0.72rem;font-family:monospace;color:#c9d1d9'>{last}</span>",
            unsafe_allow_html=True
        )

    st.divider()

    # Buttons
    st.markdown('<p class="section-header">Actions</p>', unsafe_allow_html=True)

    if st.button("⟳  Refresh Data", use_container_width=True,
                 help="Runs daily pipeline scripts to fetch data from last stored date to today."):
        with st.spinner("Running daily pipeline..."):
            DAILY_SCRIPTS = [
                ("bhavcopy_ingestion.py", []),
                ("indicators.py",         []),
                ("macro.py",              []),
                ("fii_dii_stockedge.py",  []),
                ("rbi_macro.py",          []),
                ("sentiment.py",          ["--mode", "daily"]),
                ("data_quality.py",       []),
            ]
            errors = []
            for script, args in DAILY_SCRIPTS:
                script_path = os.path.join(SCRIPTS_DIR, script)
                if not os.path.exists(script_path):
                    continue
                ok, out = run_script(script, args)
                if not ok:
                    errors.append(f"{script}: {out[:200]}")

            if errors:
                st.error("Some scripts failed:\n" + "\n".join(errors))
            else:
                st.success("Pipeline refreshed.")
                st.cache_data.clear()
                st.rerun()

    # Setup button — stub, wires up all scripts for fresh install
    if st.button("⚙  Setup Hedge Fund", use_container_width=True,
                 help="Full pipeline setup for a new device. Runs all scripts from scratch. "
                      "Warning: bhavcopy ingestion takes 30–45 minutes."):
        st.info(
            "Setup pipeline not yet automated. Run scripts manually in this order:\n\n"
            "1. `python data/setup_db.py`\n"
            "2. `python data/bhavcopy_ingestion.py`\n"
            "3. `python data/indicators.py`\n"
            "4. `python data/screener_fundamentals.py`\n"
            "5. `python data/macro.py`\n"
            "6. `python data/fii_dii_stockedge.py`\n"
            "7. `python data/rbi_macro.py`\n"
            "8. `python data/sentiment.py --mode backfill`\n"
            "9. `python data/data_quality.py`\n"
            "10. `python data/features.py`\n"
            "11. `python data/export_features.py`\n\n"
            "Then upload `exports/features_master_latest.parquet` to Google Colab and run the training notebooks."
        )

    st.divider()
    st.markdown(
        f"<div style='font-size:0.65rem;color:#484f58;font-family:monospace'>"
        f"Last loaded: {datetime.now().strftime('%H:%M:%S')}</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

signals_df, signals_path = load_signals()
sector_map               = load_sectors()
macro_row                = load_macro_latest()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

# Determine signal date and regime
if not signals_df.empty:
    signal_date  = signals_df["Date"].max()
    regime_int   = int(signals_df.loc[signals_df["Date"] == signal_date, "Regime_Int"].iloc[0])
    signals_today = signals_df[signals_df["Date"] == signal_date].copy()
    n_buy  = (signals_today["Signal"] == 1).sum()
    n_sell = (signals_today["Signal"] == -1).sum()
    n_hold = (signals_today["Signal"] == 0).sum()
else:
    signal_date   = date.today()
    regime_int    = 3
    signals_today = pd.DataFrame()
    n_buy = n_sell = n_hold = 0

# ─────────────────────────────────────────────────────────────────────────────
# RUN OPTIMIZER (must run before KPIs so opt_stats is available)
# ─────────────────────────────────────────────────────────────────────────────

opt_result = run_optimization(signals_today, sector_map, nav_input, regime_int)
if opt_result is not None:
    proposed_df = opt_result.positions
    ps_final    = opt_result.portfolio_state
    opt_stats   = opt_result.stats
    rejected_df = opt_result.rejected
else:
    proposed_df = pd.DataFrame()
    ps_final    = None
    opt_stats   = None
    rejected_df = pd.DataFrame()

# Top bar
col_title, col_date, col_regime = st.columns([3, 2, 2])
with col_title:
    st.markdown(
        "<h2 style='font-family:monospace;font-size:1.3rem;color:#c9d1d9;"
        "margin:0;padding:0;font-weight:600'>AI HEDGE FUND SIMULATOR</h2>"
        "<p style='font-family:monospace;font-size:0.7rem;color:#8b949e;"
        "text-transform:uppercase;letter-spacing:0.1em;margin:2px 0 0 0'>"
        "NSE Nifty 500 · Long/Short Equity · v2</p>",
        unsafe_allow_html=True
    )
with col_date:
    st.markdown(
        f"<div style='text-align:right'>"
        f"<div style='font-family:monospace;font-size:0.7rem;color:#8b949e;"
        f"text-transform:uppercase;letter-spacing:0.08em'>Signal Date</div>"
        f"<div style='font-family:monospace;font-size:1.1rem;color:#58a6ff;font-weight:600'>"
        f"{pd.Timestamp(signal_date).strftime('%d %b %Y')}</div>"
        f"</div>",
        unsafe_allow_html=True
    )
with col_regime:
    st.markdown(
        f"<div style='text-align:right'>"
        f"<div style='font-family:monospace;font-size:0.7rem;color:#8b949e;"
        f"text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px'>Market Regime</div>"
        f"{regime_pill(regime_int)}"
        f"</div>",
        unsafe_allow_html=True
    )

st.divider()

# Top KPIs
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Model BUY Signals",  n_buy,
          help="Stocks ranked in top 10% by ensemble. Raw model output — not all enter the portfolio.")
k2.metric("Model SELL Signals", n_sell,
          help="Stocks ranked in bottom 10% by ensemble. Raw model output.")
k3.metric("Neutral Signals",    n_hold,
          help="Stocks with no directional conviction — model rank between bottom and top decile.")
k4.metric("Universe",           500)
k5.metric("Portfolio Positions",
          opt_stats.n_total if opt_stats else "—",
          help="Actual positions after optimizer + risk checks. Max 55.")
k6.metric("Simulation NAV", f"₹{nav_input/1e7:.2f} Cr")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab5 = st.tabs([
    "SIGNALS", "PROPOSED BOOK", "RISK", "MACRO & REGIME"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

with tab1:

    if signals_today.empty:
        st.warning(
            f"No signals file found in `{SIGNALS_DIR}`. "
            "Download `signals_*.csv` from Google Drive and place it in `exports/model_output/`."
        )
    else:
        st.markdown(
            f"<div class='section-header'>Today's Signals — {pd.Timestamp(signal_date).strftime('%d %b %Y')}</div>",
            unsafe_allow_html=True
        )

        # Merge sector + company name — strip .NS for display only
        df_view = signals_today.copy()
        df_view["Ticker_Display"] = df_view["Ticker"].str.replace(".NS","",regex=False)
        df_view = df_view.merge(sector_map, on="Ticker", how="left")
        df_view["Sector"]       = df_view["Sector"].fillna("Unknown")
        df_view["Company_Name"] = df_view.get("Company_Name",
                                    df_view["Ticker_Display"]).fillna(df_view["Ticker_Display"])
        df_view["Tier"] = df_view["Data_Tier"].map(TIER_LABEL).fillna("?")

        # ── Filters: Signal selector + search bar (aligned) ─────────────
        fc1, fc2 = st.columns([2, 6])
        sig_filter = fc1.selectbox("Signal", ["ALL", "BUY", "SELL", "NEUTRAL"], index=0)

        with fc2:
            sc1, sc2 = st.columns([11, 1])
            search_str = sc1.text_input(
                "Search",
                value="",
                placeholder="Search ticker, company name or sector…",
            )
            # SVG magnifying glass icon aligned with input bottom
            sc2.markdown(
                """<div style='padding-top:32px;cursor:pointer;color:#8b949e' title='Search'>
                <svg xmlns='http://www.w3.org/2000/svg' width='18' height='18'
                     viewBox='0 0 24 24' fill='none' stroke='currentColor'
                     stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
                  <circle cx='11' cy='11' r='8'/>
                  <line x1='21' y1='21' x2='16.65' y2='16.65'/>
                </svg></div>""",
                unsafe_allow_html=True
            )

        # Apply filters — each is optional, any combination works (issue 3)
        mask = pd.Series([True] * len(df_view), index=df_view.index)
        if sig_filter != "ALL":
            sig_map = {"BUY": 1, "SELL": -1, "NEUTRAL": 0}
            mask &= df_view["Signal"] == sig_map[sig_filter]
        if search_str.strip():
            s = search_str.strip().upper()
            mask &= (
                df_view["Ticker_Display"].str.upper().str.contains(s, na=False) |
                df_view["Company_Name"].str.upper().str.contains(s, na=False)  |
                df_view["Sector"].str.upper().str.contains(s, na=False)
            )

        df_show = df_view[mask].sort_values("Final_Rank", ascending=False).reset_index(drop=True)

        # ── Fetch Close + Volume for signal date ──────────────────────────
        sig_tickers = tuple(df_show["Ticker"].tolist())
        sig_date_str = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
        price_data = load_prices_on_date(sig_tickers, sig_date_str)
        if not price_data.empty:
            price_idx = price_data.set_index("Ticker")[["Adj_Close","Volume"]]
        else:
            price_idx = pd.DataFrame(columns=["Adj_Close","Volume"])

        # ── Build display table ────────────────────────────────────────────
        df_table = pd.DataFrame({
            "Ticker":       df_show["Ticker_Display"],
            "Company":      df_show["Company_Name"],
            "Sector":       df_show["Sector"],
            "Signal":       df_show["Signal"].map({1:"BUY", -1:"SELL", 0:"NEUTRAL"}),
            "Close (₹)":    df_show["Ticker"].map(
                                price_idx["Adj_Close"].apply(
                                    lambda v: f"{v:.2f}" if pd.notna(v) else "—"
                                )
                            ).fillna("—"),
            "Volume":       df_show["Ticker"].map(
                                price_idx["Volume"].apply(
                                    lambda v: f"{v/1e5:.1f}L" if pd.notna(v) and v >= 1e5
                                    else (f"{int(v):,}" if pd.notna(v) else "—")
                                )
                            ).fillna("—"),
        })

        def colour_signal(val):
            if val == "BUY":     return "color: #3fb950; font-weight:600"
            if val == "SELL":    return "color: #f85149; font-weight:600"
            if val == "NEUTRAL": return "color: #8b949e"
            return ""

        st.dataframe(
            df_table.style.map(colour_signal, subset=["Signal"]),
            use_container_width=True, height=500, hide_index=True,
        )

        st.markdown(
            f"<div style='font-family:monospace;font-size:0.7rem;color:#8b949e;margin-top:6px'>"
            f"Showing {len(df_show)} of {len(df_view)} signals · "
            f"Source: {os.path.basename(signals_path)}</div>",
            unsafe_allow_html=True
        )

        # ── BUY signal sector distribution — plotly with axis labels ──────
        st.markdown("<div class='section-header'>BUY Signal Sector Distribution</div>",
                    unsafe_allow_html=True)
        buys_sector = df_view[df_view["Signal"] == 1]["Sector"].value_counts().reset_index()
        buys_sector.columns = ["Sector", "Count"]
        buys_sector = buys_sector.sort_values("Count", ascending=False)

        if not buys_sector.empty:
            try:
                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(
                    x=buys_sector["Sector"],
                    y=buys_sector["Count"],
                    marker_color="#3fb950",
                    hovertemplate="%{x}<br>BUY signals: %{y}<extra></extra>",
                ))
                fig.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=10, b=60),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="JetBrains Mono, monospace", color="#8b949e", size=11),
                    xaxis=dict(
                        title=dict(text="Sector", font=dict(color="#8b949e", size=11)),
                        tickfont=dict(color="#8b949e", size=10),
                        gridcolor="#1e2d40", linecolor="#1e2d40",
                    ),
                    yaxis=dict(
                        title=dict(text="Number of BUY Signals", font=dict(color="#8b949e", size=11)),
                        tickfont=dict(color="#8b949e", size=10),
                        gridcolor="#1e2d40", linecolor="#1e2d40",
                        dtick=1,
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(buys_sector.set_index("Sector")["Count"],
                             use_container_width=True, height=220, color="#3fb950")
        else:
            st.info("No BUY signals to display.")

        # ── SELL signal sector distribution ───────────────────────────────
        st.markdown("<div class='section-header'>SELL Signal Sector Distribution</div>",
                    unsafe_allow_html=True)
        sells_sector = df_view[df_view["Signal"] == -1]["Sector"].value_counts().reset_index()
        sells_sector.columns = ["Sector", "Count"]
        sells_sector = sells_sector.sort_values("Count", ascending=False)

        if not sells_sector.empty:
            try:
                import plotly.graph_objects as go
                fig_sell = go.Figure(go.Bar(
                    x=sells_sector["Sector"],
                    y=sells_sector["Count"],
                    marker_color="#f85149",
                    hovertemplate="%{x}<br>SELL signals: %{y}<extra></extra>",
                ))
                fig_sell.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=10, b=60),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="JetBrains Mono, monospace", color="#8b949e", size=11),
                    xaxis=dict(
                        title=dict(text="Sector", font=dict(color="#8b949e", size=11)),
                        tickfont=dict(color="#8b949e", size=10),
                        gridcolor="#1e2d40", linecolor="#1e2d40",
                    ),
                    yaxis=dict(
                        title=dict(text="Number of SELL Signals", font=dict(color="#8b949e", size=11)),
                        tickfont=dict(color="#8b949e", size=10),
                        gridcolor="#1e2d40", linecolor="#1e2d40",
                        dtick=1,
                    ),
                )
                st.plotly_chart(fig_sell, use_container_width=True)
            except ImportError:
                st.bar_chart(sells_sector.set_index("Sector")["Count"],
                             use_container_width=True, height=220, color="#f85149")
        else:
            st.info("No SELL signals to display.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROPOSED BOOK
# ══════════════════════════════════════════════════════════════════════════════

with tab2:

    if not RISK_AVAILABLE:
        st.error("Risk manager / optimizer not available.")
    elif proposed_df.empty:
        st.warning("No positions generated. Check signals file and risk manager logs.")
    else:
        longs  = proposed_df[proposed_df["Direction"] == "LONG"]
        shorts = proposed_df[proposed_df["Direction"] == "SHORT"]

        # ── Optimizer summary metrics ──────────────────────────────────────
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Total Positions",  opt_stats.n_total)
        m2.metric("Longs",            opt_stats.n_longs)
        m3.metric("Shorts",           opt_stats.n_shorts)
        m4.metric("Gross Long",       f"{opt_stats.gross_long_pct:.1%}")
        m5.metric("Gross Short",      f"{opt_stats.gross_short_pct:.1%}")
        m6.metric("Net Exposure",     f"{opt_stats.net_exposure_pct:.1%}")
        m7.metric("Cash",             f"{opt_stats.cash_pct:.1%}")

        st.divider()

        # ── Pie charts ────────────────────────────────────────────────────
        import plotly.graph_objects as go

        pc1, pc2 = st.columns(2)

        with pc1:
            st.markdown("<div class='section-header'>Portfolio Distribution</div>",
                        unsafe_allow_html=True)
            if not proposed_df.empty:
                pie_labels = proposed_df["Ticker"].str.replace(".NS","",regex=False)
                pie_values = proposed_df["Size_%NAV"]
                pie_colors = ["#3fb950" if d=="LONG" else "#f85149"
                              for d in proposed_df["Direction"]]
                fig_pie1 = go.Figure(go.Pie(
                    labels=pie_labels, values=pie_values,
                    marker=dict(colors=pie_colors, line=dict(color="#0a0e1a", width=1)),
                    hovertemplate="%{label}<br>Allocation: %{value:.2f}% NAV<extra></extra>",
                    textinfo="label+percent", textfont=dict(size=9, color="#c9d1d9"),
                    hole=0.35,
                ))
                fig_pie1.update_layout(
                    height=380, margin=dict(l=0,r=0,t=20,b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    font=dict(family="JetBrains Mono, monospace", color="#8b949e"),
                )
                st.plotly_chart(fig_pie1, use_container_width=True)

        with pc2:
            st.markdown("<div class='section-header'>Sector Allocation (% NAV)</div>",
                        unsafe_allow_html=True)
            if not proposed_df.empty:
                sec_nav = proposed_df.groupby("Sector")["Size_%NAV"].sum().reset_index()
                fig_pie2 = go.Figure(go.Pie(
                    labels=sec_nav["Sector"], values=sec_nav["Size_%NAV"],
                    hovertemplate="%{label}<br>NAV: %{value:.2f}%<br>Count: %{customdata}<extra></extra>",
                    customdata=proposed_df.groupby("Sector").size().reindex(sec_nav["Sector"]).values,
                    textinfo="label+percent", textfont=dict(size=9, color="#c9d1d9"),
                    hole=0.35,
                ))
                fig_pie2.update_layout(
                    height=380, margin=dict(l=0,r=0,t=20,b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    font=dict(family="JetBrains Mono, monospace", color="#8b949e"),
                )
                st.plotly_chart(fig_pie2, use_container_width=True)

        st.divider()

        # ── Long book ─────────────────────────────────────────────────────
        if not longs.empty:
            # Fetch prices for quantity calculation
            long_tickers = tuple(longs["Ticker"].tolist())
            sig_date_str_pb = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
            pb_prices = load_prices_on_date(long_tickers, sig_date_str_pb)
            pb_price_idx = pb_prices.set_index("Ticker")["Adj_Close"] if not pb_prices.empty else pd.Series(dtype=float)

            def build_book_table(book_df, nav_val, price_series):
                rows = []
                for _, row in book_df.iterrows():
                    ticker  = row["Ticker"]
                    company = row.get("Company_Name", ticker.replace(".NS",""))
                    sector  = row["Sector"]
                    alloc_rs= row["Alloc_Rs"]
                    size_pct= row["Size_%NAV"]
                    price   = price_series.get(ticker, float("nan"))
                    qty     = int(alloc_rs / price) if price and price > 0 else "—"
                    rows.append({
                        "Ticker":            ticker.replace(".NS",""),
                        "Company":           company,
                        "Sector":            sector,
                        "Stock Price (₹)":   f"{price:.2f}" if isinstance(price, float) and price > 0 else "—",
                        "Quantity":          qty,
                        "Allocation %":      f"{size_pct:.2f}%",
                        "Total Invested (₹)":f"₹{alloc_rs:,.0f}",
                    })
                return pd.DataFrame(rows)

            st.markdown("<div class='section-header'>Long Book</div>", unsafe_allow_html=True)
            long_table = build_book_table(
                longs.sort_values("Size_%NAV", ascending=False).reset_index(drop=True),
                nav_input, pb_price_idx
            )
            st.dataframe(long_table, use_container_width=True, hide_index=True,
                         height=min(50 + len(longs)*38, 500))

        # ── Short book ────────────────────────────────────────────────────
        if not shorts.empty:
            short_tickers = tuple(shorts["Ticker"].tolist())
            sp_prices = load_prices_on_date(short_tickers, sig_date_str_pb)
            sp_price_idx = sp_prices.set_index("Ticker")["Adj_Close"] if not sp_prices.empty else pd.Series(dtype=float)

            st.markdown("<div class='section-header'>Short Book</div>", unsafe_allow_html=True)
            short_table = build_book_table(
                shorts.sort_values("Size_%NAV", ascending=False).reset_index(drop=True),
                nav_input, sp_price_idx
            )
            st.dataframe(short_table, use_container_width=True, hide_index=True,
                         height=min(50 + len(shorts)*38, 300))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK
# ══════════════════════════════════════════════════════════════════════════════

with tab3:

    if not RISK_AVAILABLE:
        st.error("Risk manager not available.")
    else:
        st.markdown("<div class='section-header'>Portfolio-Level Limits</div>", unsafe_allow_html=True)

        if ps_final is not None:
            gl  = ps_final.gross_long_nav_pct
            gs  = ps_final.gross_short_nav_pct
            net = ps_final.net_exposure_pct
            grs = ps_final.gross_exposure_pct
            csh = ps_final.cash_pct
        else:
            gl = gs = net = grs = csh = 0.0

        rc1, rc2 = st.columns(2)

        with rc1:
            st.markdown(
                gauge_html(gl,  1.20,      "Gross Long (cap 120%)") +
                gauge_html(gs,  0.20,      "Gross Short (cap 20%)") +
                gauge_html(grs, GROSS_EXPOSURE_CAP_PCT, "Gross Exposure (cap 150%)"),
                unsafe_allow_html=True
            )

        with rc2:
            # Net exposure: show position within [70%, 110%] band
            net_util = (net - NET_EXPOSURE_MIN_PCT) / (NET_EXPOSURE_MAX_PCT - NET_EXPOSURE_MIN_PCT)
            net_util = max(0.0, min(net_util, 1.0))
            st.markdown(
                gauge_html(csh, 1.0, f"Cash (floor {CASH_RESERVE_MIN_PCT:.0%})") +
                f"<div style='margin-bottom:10px'>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:3px'>"
                f"<span style='font-size:0.72rem;color:#8b949e;font-family:monospace'>Net Exposure (range 70%–110%)</span>"
                f"<span style='font-size:0.72rem;color:#58a6ff;font-family:monospace'>{net:.1%}</span>"
                f"</div>"
                f"<div class='gauge-bg'><div class='gauge-fill' "
                f"style='width:{net_util*100:.1f}%;background:#58a6ff'></div></div>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.divider()

        # Sector utilisation
        st.markdown("<div class='section-header'>Sector-Level Limits</div>", unsafe_allow_html=True)

        if ps_final is not None:
            sec_long_pcts  = ps_final.sector_long_pcts
            sec_short_pcts = ps_final.sector_short_pcts
            all_sectors    = sorted(set(list(sec_long_pcts.keys()) + list(sec_short_pcts.keys())))
        else:
            sec_long_pcts = sec_short_pcts = {}
            all_sectors   = []

        if all_sectors:
            sec_rows = []
            for sec in all_sectors:
                long_pct  = sec_long_pcts.get(sec, 0.0)
                short_pct = sec_short_pcts.get(sec, 0.0)
                long_cap  = MAX_SECTOR_GROSS_LONG_FIN_PCT if sec in FINANCIAL_SECTOR_NAMES else MAX_SECTOR_GROSS_LONG_PCT
                sec_rows.append({
                    "Sector":        sec,
                    "Long %NAV":     f"{long_pct:.1%}",
                    "Long Cap":      f"{long_cap:.0%}",
                    "Long Util":     f"{long_pct/long_cap:.0%}",
                    "Short %NAV":    f"{short_pct:.1%}",
                    "Short Cap":     f"{MAX_SECTOR_GROSS_SHORT_PCT:.0%}",
                    "Short Util":    f"{short_pct/MAX_SECTOR_GROSS_SHORT_PCT:.0%}" if short_pct > 0 else "—",
                })

            st.dataframe(
                pd.DataFrame(sec_rows),
                use_container_width=True,
                hide_index=True,
                height=min(50 + len(all_sectors)*38, 400),
            )
        else:
            st.info("No proposed positions to assess sector limits against.")

        st.divider()

        # ── Sector allocation charts (moved from proposed book) ───────────
        import plotly.graph_objects as go
        sa1, sa2 = st.columns(2)
        with sa1:
            st.markdown("<div class='section-header'>Long Book — Sector Allocation</div>",
                        unsafe_allow_html=True)
            if not proposed_df.empty:
                longs_r  = proposed_df[proposed_df["Direction"]=="LONG"]
                if not longs_r.empty:
                    sec_l = longs_r.groupby("Sector")["Size_%NAV"].sum().sort_values(ascending=False).reset_index()
                    fig_sl = go.Figure(go.Bar(
                        x=sec_l["Sector"], y=sec_l["Size_%NAV"],
                        marker_color="#58a6ff",
                        hovertemplate="%{x}<br>%{y:.2f}% NAV<extra></extra>",
                    ))
                    fig_sl.update_layout(
                        height=260, margin=dict(l=10,r=10,t=10,b=60),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="JetBrains Mono, monospace", color="#8b949e", size=11),
                        xaxis=dict(title="Sector", tickfont=dict(size=10,color="#8b949e"),
                                   gridcolor="#1e2d40"),
                        yaxis=dict(title="% of NAV", tickfont=dict(size=10,color="#8b949e"),
                                   gridcolor="#1e2d40"),
                    )
                    st.plotly_chart(fig_sl, use_container_width=True)
        with sa2:
            st.markdown("<div class='section-header'>Short Book — Sector Allocation</div>",
                        unsafe_allow_html=True)
            if not proposed_df.empty:
                shorts_r = proposed_df[proposed_df["Direction"]=="SHORT"]
                if not shorts_r.empty:
                    sec_s = shorts_r.groupby("Sector")["Size_%NAV"].sum().sort_values(ascending=False).reset_index()
                    fig_ss = go.Figure(go.Bar(
                        x=sec_s["Sector"], y=sec_s["Size_%NAV"],
                        marker_color="#f85149",
                        hovertemplate="%{x}<br>%{y:.2f}% NAV<extra></extra>",
                    ))
                    fig_ss.update_layout(
                        height=260, margin=dict(l=10,r=10,t=10,b=60),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="JetBrains Mono, monospace", color="#8b949e", size=11),
                        xaxis=dict(title="Sector", tickfont=dict(size=10,color="#8b949e"),
                                   gridcolor="#1e2d40"),
                        yaxis=dict(title="% of NAV", tickfont=dict(size=10,color="#8b949e"),
                                   gridcolor="#1e2d40"),
                    )
                    st.plotly_chart(fig_ss, use_container_width=True)
                else:
                    st.info("No short positions.")

        # ── Rejected candidates ───────────────────────────────────────────
        if not rejected_df.empty:
            st.divider()
            with st.expander(f"Rejected candidates ({len(rejected_df)})", expanded=False):
                st.dataframe(rejected_df, use_container_width=True, hide_index=True)





# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MACRO & REGIME
# ══════════════════════════════════════════════════════════════════════════════

with tab5:

    st.markdown("<div class='section-header'>Current Regime</div>", unsafe_allow_html=True)

    r1, r2, r3 = st.columns([1, 2, 3])
    with r1:
        st.markdown(
            f"<div style='padding:20px;text-align:center'>"
            f"{regime_pill(regime_int)}"
            f"<div style='font-family:monospace;font-size:0.7rem;color:#8b949e;margin-top:8px'>"
            f"HMM State {regime_int}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with r2:
        regime_descriptions = {
            0: ("Bull",     "Trending up. XGB/LGB momentum-weighted. Full deployment."),
            1: ("Bear",     "Risk-off. Fundamental quality models weighted higher. Net long reduced."),
            2: ("High Vol", "Elevated uncertainty. LSTM sequence patterns weighted up. Tighter position caps."),
            3: ("Sideways", "Range-bound. Tree models preferred. Mean-reversion favoured."),
        }
        label, desc = regime_descriptions.get(regime_int, ("Unknown", "—"))
        st.markdown(
            f"<div style='padding:12px'>"
            f"<div style='font-family:monospace;font-size:0.85rem;color:#c9d1d9;font-weight:600'>{label}</div>"
            f"<div style='font-family:monospace;font-size:0.75rem;color:#8b949e;margin-top:6px;line-height:1.5'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with r3:
        weights = {
            0: {"XGBoost": 0.40, "LightGBM": 0.30, "LSTM": 0.30},
            1: {"XGBoost": 0.35, "LightGBM": 0.35, "LSTM": 0.30},
            2: {"XGBoost": 0.30, "LightGBM": 0.30, "LSTM": 0.40},
            3: {"XGBoost": 0.40, "LightGBM": 0.40, "LSTM": 0.20},
        }.get(regime_int, {"XGBoost": 0.40, "LightGBM": 0.30, "LSTM": 0.30})

        st.markdown(
            "<div style='font-family:monospace;font-size:0.7rem;color:#8b949e;"
            "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px'>Ensemble Weights</div>",
            unsafe_allow_html=True
        )
        for model, w in weights.items():
            st.markdown(
                gauge_html(w, 1.0, f"{model}"),
                unsafe_allow_html=True
            )

    st.divider()

    # Macro snapshot
    st.markdown("<div class='section-header'>Latest Macro Snapshot</div>", unsafe_allow_html=True)

    MACRO_DISPLAY = {
        "India_VIX":         ("India VIX",           ""),
        "USDINR":            ("USD/INR",              "₹"),
        "Crude_Oil":         ("Crude Oil",            "$"),
        "Gold":              ("Gold",                 "$"),
        "Repo_Rate":         ("RBI Repo Rate",        "%"),
        "GDP_India":         ("GDP India (QoQ)",      ""),
        "CPI_India":         ("CPI India",            ""),
        "Fed_Funds_Rate":    ("Fed Funds Rate",       "%"),
        "US_10Y_Bond":       ("US 10Y Yield",         "%"),
        "FII_Daily_Net_Cr":  ("FII Net Inflow (Day)", "₹Cr"),
        "FII_Monthly_Net_Cr":("FII Net Inflow (Mon)", "₹Cr"),
        "DII_Daily_Net_Cr":  ("DII Net Inflow (Day)", "₹Cr"),
        "DII_Monthly_Net_Cr":("DII Net Inflow (Mon)", "₹Cr"),
        "Forex_Reserves_USD":("Forex Reserves",       "$B"),
    }

    macro_cols = st.columns(4)
    col_idx    = 0

    for field, (label, unit) in MACRO_DISPLAY.items():
        if field in macro_row.index and pd.notna(macro_row[field]):
            val = macro_row[field]
            try:
                val_f = float(val)
                if abs(val_f) >= 1_000:
                    display = f"{unit}{val_f:,.1f}"
                elif unit == "%":
                    display = f"{val_f:.2f}%"
                else:
                    display = f"{unit}{val_f:.2f}"
            except (ValueError, TypeError):
                display = str(val)

            macro_cols[col_idx % 4].metric(label, display)
            col_idx += 1

    if col_idx == 0:
        st.info("Macro data not available. Run `python data/macro.py` and `python data/rbi_macro.py`.")

    # Macro date
    if "Date" in macro_row.index and pd.notna(macro_row.get("Date")):
        st.markdown(
            f"<div style='font-family:monospace;font-size:0.7rem;color:#8b949e;margin-top:8px'>"
            f"Data as of: {str(macro_row['Date'])[:10]}</div>",
            unsafe_allow_html=True
        )