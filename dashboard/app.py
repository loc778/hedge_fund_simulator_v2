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

def _wrap_label(s, max_words=2):
    """Split label into lines of max_words words for clean x-axis display."""
    words = str(s).split()
    lines = []
    for i in range(0, len(words), max_words):
        lines.append(" ".join(words[i:i+max_words]))
    return "<br>".join(lines)


def _mk(col, label, value, color="#c9d1d9"):
    """Render a centered colored metric card (shared across all metric rows)."""
    col.markdown(
        f"<div style='background:#0d1117;border:1px solid #1e2d40;border-radius:6px;"
        f"padding:12px 16px;text-align:center'>"
        f"<div style='font-size:0.72rem;color:#8b949e;text-transform:uppercase;"
        f"letter-spacing:0.08em;font-family:monospace'>{label}</div>"
        f"<div style='font-size:1.3rem;font-weight:600;color:{color};"
        f"font-family:monospace'>{value}</div></div>",
        unsafe_allow_html=True,
    )


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

    /* Tab styling — pill capsule navigation */
    [data-testid="stTabs"] { border-bottom: none !important; }
    [data-testid="stTabs"] > div:first-child { border-bottom: none !important; padding: 10px 0 10px 0 !important; }
    [role="tablist"] {
        justify-content: center !important;
        background: #0d1117 !important;
        border: 1px solid #1e2d40 !important;
        border-radius: 50px !important;
        padding: 5px 8px !important;
        gap: 2px !important;
        width: fit-content !important;
        margin: 10px auto !important;
    }
    button[data-baseweb="tab"] {
        color: #8b949e !important;
        font-size: 0.82rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 7px 20px !important;
        background: transparent !important;
        white-space: nowrap !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #c9d1d9 !important;
        background: #1e2d40 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        background: #1e2d40 !important;
        font-weight: 600 !important;
    }
    [data-testid="stTabs"] [role="tablist"] > div[data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [role="tablist"] > div[data-baseweb="tab-border"] {
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
def load_macro_latest(as_of_date: str = None) -> pd.Series:
    """
    Load macro row closest to as_of_date (on or before).
    Falls back to most recent row if as_of_date not provided.
    """
    try:
        engine = get_engine()
        table  = TABLES.get("macro", "macro_indicators")
        with engine.connect() as conn:
            if as_of_date:
                df = pd.read_sql(
                    f"SELECT * FROM {table} "
                    f"WHERE Date <= '{as_of_date}' "
                    f"ORDER BY Date DESC LIMIT 1",
                    conn
                )
            else:
                df = pd.read_sql(
                    f"SELECT * FROM {table} ORDER BY Date DESC LIMIT 1",
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
        "OHLCV":      (TABLES.get("ohlcv",      "nifty500_ohlcv"),      "Date"),
        "Indicators": (TABLES.get("indicators",  "nifty500_indicators"), "Date"),
        "Macro":      (TABLES.get("macro",       "macro_indicators"),    "Date"),
        "Features":   (TABLES.get("features",    "features_master"),     "Date"),
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
def persist_portfolio(
    proposed_df: pd.DataFrame,
    signal_date,
    nav: float,
    engine,
) -> str:
    """
    Persist optimizer output to portfolio_positions.

    - If signal_date already has open rows → skip (idempotent on refresh).
    - New tickers not in the open book → INSERT as new open positions.
    - Open tickers no longer in proposed_df → close them (EXIT_Date, Exit_Price, Exit_Reason).

    Returns a status string for display.
    """
    from sqlalchemy import text as _text
    import pandas as pd

    sig_date_str = pd.Timestamp(signal_date).strftime("%Y-%m-%d")

    try:
        with engine.connect() as conn:
            # ── Idempotency check ─────────────────────────────────────────
            already = conn.execute(_text(
                "SELECT COUNT(*) FROM portfolio_positions "
                "WHERE Signal_Date = :sd AND Status = 'open'"
            ), {"sd": sig_date_str}).scalar()
            if already and already > 0:
                return f"skipped — signal date {sig_date_str} already committed ({already} positions)"

            # ── Current open positions ────────────────────────────────────
            open_rows = pd.read_sql(
                "SELECT id, Ticker, Direction FROM portfolio_positions "
                "WHERE Status = 'open'",
                conn,
            )
            open_tickers = set(open_rows["Ticker"].tolist()) if not open_rows.empty else set()

            # ── Proposed tickers from optimizer ──────────────────────────
            if proposed_df.empty:
                new_tickers = set()
                proposed_tickers = set()
            else:
                proposed_df = proposed_df.copy()
                proposed_df["_Ticker_norm"] = proposed_df["Ticker"].str.replace(
                    r"\.NS$", "", regex=True
                )
                proposed_tickers = set(proposed_df["Ticker"].tolist())

            # Normalise open_tickers to .NS for comparison
            open_tickers_ns = {
                t if t.endswith(".NS") else t + ".NS" for t in open_tickers
            }

            # ── 1. Close exits ────────────────────────────────────────────
            exits = open_tickers_ns - proposed_tickers
            n_closed = 0
            for ticker in exits:
                # Fetch latest price from ohlcv as exit price
                price_row = conn.execute(_text(
                    "SELECT Adj_Close FROM nifty500_ohlcv "
                    "WHERE Ticker = :t AND Date <= :d "
                    "ORDER BY Date DESC LIMIT 1"
                ), {"t": ticker, "d": sig_date_str}).fetchone()
                exit_price = float(price_row[0]) if price_row and price_row[0] else None

                conn.execute(_text(
                    "UPDATE portfolio_positions SET "
                    "Status = 'closed', "
                    "Exit_Date = :ed, "
                    "Exit_Price = :ep, "
                    "Exit_Reason = :er "
                    "WHERE Ticker = :t AND Status = 'open'"
                ), {
                    "ed": sig_date_str,
                    "ep": exit_price,
                    "er": "signal_dropped",
                    "t":  ticker,
                })
                n_closed += 1

            # ── 2. Insert new positions ───────────────────────────────────
            new_tickers = proposed_tickers - open_tickers_ns
            n_inserted  = 0

            if not proposed_df.empty:
                # Load entry prices for new tickers
                new_list = ", ".join(f"'{t}'" for t in new_tickers) if new_tickers else "''"
                price_df = pd.read_sql(
                    f"""
                    SELECT o.Ticker, o.Adj_Close
                    FROM nifty500_ohlcv o
                    INNER JOIN (
                        SELECT Ticker, MAX(Date) AS d
                        FROM nifty500_ohlcv
                        WHERE Ticker IN ({new_list}) AND Date <= '{sig_date_str}'
                        GROUP BY Ticker
                    ) m ON o.Ticker = m.Ticker AND o.Date = m.d
                    """,
                    conn,
                ) if new_tickers else pd.DataFrame(columns=["Ticker", "Adj_Close"])
                price_idx = price_df.set_index("Ticker")["Adj_Close"].to_dict()

                for _, row in proposed_df.iterrows():
                    ticker = row["Ticker"]
                    if ticker not in new_tickers:
                        continue

                    direction  = row["Direction"].lower()
                    size_pct   = float(row["Size_%NAV"]) / 100
                    alloc_rs   = float(row["Alloc_Rs"])
                    sector     = str(row.get("Sector", "Unknown") or "Unknown")
                    is_midcap  = 1 if str(row.get("Tier", "A")) in ("B", "C") else 0
                    entry_price= float(price_idx.get(ticker, 0))
                    shares     = int(alloc_rs / entry_price) if entry_price > 0 else 0
                    pos_class  = (
                        "short"      if direction == "short"
                        else "alpha_long" if is_midcap
                        else "core_long"
                    )
                    stop_loss  = (
                        round(entry_price * (1 - 0.15), 4) if direction == "long" and entry_price > 0
                        else round(entry_price * (1 + 0.10), 4) if direction == "short" and entry_price > 0
                        else None
                    )

                    conn.execute(_text("""
                        INSERT IGNORE INTO portfolio_positions
                        (Ticker, Signal_Date, Entry_Date, Entry_Price, Direction,
                         Position_Class, Sector, Is_Midcap, Stop_Loss_Price,
                         NAV_Weight_At_Entry, Shares, Status)
                        VALUES
                        (:ticker, :sd, :ed, :ep, :dir,
                         :pc, :sector, :imc, :sl,
                         :nav_w, :shares, 'open')
                    """), {
                        "ticker":  ticker,
                        "sd":      sig_date_str,
                        "ed":      sig_date_str,
                        "ep":      entry_price if entry_price > 0 else None,
                        "dir":     direction,
                        "pc":      pos_class,
                        "sector":  sector,
                        "imc":     is_midcap,
                        "sl":      stop_loss,
                        "nav_w":   round(size_pct, 6),
                        "shares":  shares if shares > 0 else None,
                    })
                    n_inserted += 1

            conn.commit()
            return f"committed — {n_inserted} new, {n_closed} closed (signal {sig_date_str})"

    except Exception as e:
        return f"error — {e}"

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
                 help="Runs daily_refresh.py to fetch up-to-date data."):
        with st.spinner("Running daily refresh..."):
            daily_refresh_path = os.path.join(_PROJECT_ROOT, "daily_refresh.py")
            if not os.path.exists(daily_refresh_path):
                st.error("daily_refresh.py not found in project root.")
            else:
                try:
                    result = subprocess.run(
                        [sys.executable, daily_refresh_path],
                        capture_output=True, text=True,
                        cwd=_PROJECT_ROOT, timeout=3600
                    )
                    if result.returncode == 0:
                        st.success("Pipeline refreshed.")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"Refresh failed:\n{(result.stdout + result.stderr)[:500]}")
                except subprocess.TimeoutExpired:
                    st.error("Timed out after 60 minutes.")
                except Exception as e:
                    st.error(f"Refresh error: {e}")




@st.cache_data(ttl=300, show_spinner=False)
def load_nav_series() -> pd.DataFrame:
    """Load backtested NAV series from ensemble output."""
    pattern = os.path.join(SIGNALS_DIR, "nav_series_*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    try:
        df = pd.read_csv(files[-1], parse_dates=["Date"])
        return df.sort_values("Date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_nifty_index(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load Nifty benchmark from macro_indicators table (Nifty500_Close or Nifty50_Close).
    Falls back to yfinance if columns not present.
    """
    try:
        engine = get_engine()
        # Try Nifty500 first, then Nifty50
        for col in ("Nifty500_Close", "Nifty50_Close", "Nifty_500", "Nifty_50",
                    "NIFTY500", "NIFTY50"):
            try:
                with engine.connect() as conn:
                    df = pd.read_sql(
                        f"SELECT Date, `{col}` AS Nifty FROM macro_indicators "
                        f"WHERE Date BETWEEN '{start_date}' AND '{end_date}' "
                        f"AND `{col}` IS NOT NULL ORDER BY Date",
                        conn
                    )
                if not df.empty:
                    df["Date"] = pd.to_datetime(df["Date"])
                    return df[["Date","Nifty"]].dropna()
            except Exception:
                continue
    except Exception:
        pass

    # Fallback: yfinance
    try:
        import yfinance as yf
        for sym in ("^CRSLDX", "^NSEI"):
            df = yf.Ticker(sym).history(start=start_date, end=end_date, auto_adjust=True)
            if not df.empty:
                df = df[["Close"]].rename(columns={"Close": "Nifty"})
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df.reset_index().rename(columns={"Date": "Date"})
    except Exception:
        pass
    return pd.DataFrame()


# ── Tax rates — Union Budget 2024 cutoff (23 Jul 2024) ───────────────────────
_BUDGET_2024_CUTOFF = pd.Timestamp("2024-07-23")
_LTCG_EXEMPTION_INR = 100_000   # ₹1L annual LTCG exemption on equity

def _tax_rates(signal_date_ts):
    """Return (stcg_rate, ltcg_rate) based on signal date."""
    if pd.Timestamp(signal_date_ts) >= _BUDGET_2024_CUTOFF:
        return 0.20, 0.125   # post-July 2024
    return 0.15, 0.10        # pre-July 2024

# Transaction cost rates (from config/architecture)
_BROKERAGE_BPS   = 3     # 0.03%
_STAMP_DUTY_BPS  = 1.5   # 0.015%
_STT_BPS         = 10    # 0.10%
_SLIPPAGE_BPS    = 5     # 0.05%
_EXCHANGE_BPS    = 0.345 # NSE exchange + SEBI
_TOTAL_COST_BPS  = _BROKERAGE_BPS + _STAMP_DUTY_BPS + _STT_BPS + _SLIPPAGE_BPS + _EXCHANGE_BPS


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

signals_df, signals_path = load_signals()
sector_map               = load_sectors()

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

macro_row = load_macro_latest(as_of_date=pd.Timestamp(signal_date).strftime("%Y-%m-%d"))

# ─────────────────────────────────────────────────────────────────────────────
# RUN OPTIMIZER (must run before KPIs so opt_stats is available)
# ─────────────────────────────────────────────────────────────────────────────

opt_result = run_optimization(signals_today, sector_map, nav_input, regime_int)

# ── Auto-persist to portfolio_positions (idempotent on refresh) ──────────────
if opt_result is not None and not opt_result.positions.empty and RISK_AVAILABLE:
    _persist_status = persist_portfolio(
        proposed_df = opt_result.positions,
        signal_date = signal_date,
        nav         = nav_input,
        engine      = get_engine(),
    )
    # Surface in sidebar so it's visible without taking up main area
    with st.sidebar:
        _color = "#3fb950" if "committed" in _persist_status else "#8b949e" if "skipped" in _persist_status else "#f85149"
        st.markdown(
            f"<div style='font-family:monospace;font-size:0.68rem;color:{_color};"
            f"padding:6px 0;border-top:1px solid #1e2d40;margin-top:4px'>"
            f"📒 {_persist_status}</div>",
            unsafe_allow_html=True,
        )

if opt_result is not None:
    proposed_df = opt_result.positions
    # Always re-merge sector + company from sector_map CSV (ground truth)
    if not proposed_df.empty and not sector_map.empty:
        proposed_df = proposed_df.drop(columns=["Sector","Company_Name"], errors="ignore")
        merge_cols = ["Ticker","Sector","Company_Name"] if "Company_Name" in sector_map.columns                      else ["Ticker","Sector"]
        proposed_df = proposed_df.merge(sector_map[merge_cols], on="Ticker", how="left")
        proposed_df["Sector"] = proposed_df["Sector"].fillna("Unknown")
        if "Company_Name" not in proposed_df.columns:
            proposed_df["Company_Name"] = proposed_df["Ticker"].str.replace(".NS","",regex=False)
        proposed_df["Company_Name"] = proposed_df["Company_Name"].fillna(
            proposed_df["Ticker"].str.replace(".NS","",regex=False)
        )
    ps_final    = opt_result.portfolio_state
    opt_stats   = opt_result.stats
    rejected_df = opt_result.rejected
else:
    proposed_df = pd.DataFrame()
    ps_final    = None
    opt_stats   = None
    rejected_df = pd.DataFrame()

# ── Top bar: [Title + Signal Date] | [Macro Strip] | [Regime] ───────────────
_h1, _h2, _h3 = st.columns([3, 4, 2])

with _h1:
    _sig_date_str = pd.Timestamp(signal_date).strftime("%d %b %Y")
    st.markdown(
        # Row 1: Title + "Signal Date" label on same line
        "<div style='display:flex;align-items:baseline;gap:20px'>"
        "<h2 style='font-family:monospace;font-size:1.3rem;color:#c9d1d9;"
        "margin:0;padding:0;font-weight:600;white-space:nowrap'>"
        "AI HEDGE FUND SIMULATOR</h2>"
        "<span style='font-family:monospace;font-size:0.65rem;color:#8b949e;"
        "text-transform:uppercase;letter-spacing:0.1em;white-space:nowrap'>"
        "Signal Date</span>"
        "</div>"
        # Row 2: Subtitle + date value on same line
        "<div style='display:flex;align-items:baseline;gap:16px;margin-top:4px'>"
        "<p style='font-family:monospace;font-size:0.68rem;color:#8b949e;"
        "text-transform:uppercase;letter-spacing:0.1em;margin:0;white-space:nowrap'>"
        "NSE Nifty 500 · Long/Short Equity · v2</p>"
        f"<span style='font-family:monospace;font-size:1.05rem;color:#58a6ff;"
        f"font-weight:700;white-space:nowrap'>{_sig_date_str}</span>"
        "</div>",
        unsafe_allow_html=True
    )

with _h2:
    # Macro strip — larger font, fetch FII from monthly if daily missing
    _macro_fields = {
        "India_VIX":   ("INDIA VIX", ""),
        "USDINR":      ("USD/INR",   "₹"),
        "Crude_Oil":   ("CRUDE",     "$"),
        "Gold":        ("GOLD",      "$"),
        "Repo_Rate":   ("REPO",      ""),
    }
    # FII: try all known column variants, prefer daily
    _fii_val = None
    _fii_label = "FII NET"
    _fii_candidates = [
        ("FII_Daily_Net_Cr",   "FII (D)"),
        ("FII_Monthly_Net_Cr", "FII (M)"),
        ("FII_Net_Buy_Cr",     "FII NET"),
        ("FII_Net",            "FII NET"),
        ("FII",                "FII NET"),
    ]
    for _fii_col, _fii_lbl in _fii_candidates:
        if _fii_col in macro_row.index and pd.notna(macro_row.get(_fii_col)):
            _fii_val   = macro_row[_fii_col]
            _fii_label = _fii_lbl
            break
    # Debug: show available macro columns with data (remove after confirming FII works)
    # st.write({c: macro_row[c] for c in macro_row.index if pd.notna(macro_row.get(c))})

    _mstrip_parts = []
    for _field, (_label, _unit) in _macro_fields.items():
        _raw = macro_row.get(_field) if _field in macro_row.index else None
        if _raw is not None and pd.notna(_raw):
            try:
                _v = float(_raw)
                _display = f"{_unit}{_v:,.2f}" if abs(_v) < 1000 else f"{_unit}{_v:,.0f}"
            except Exception:
                _display = str(_raw)
            if _label == "INDIA VIX":
                _color = "#3fb950" if float(_raw) < 20 else "#f85149" if float(_raw) > 25 else "#e3b341"
            elif _label == "REPO":
                _display = f"{float(_raw):.2f}%"
                _color = "#c9d1d9"
            elif _label == "GOLD":
                _display = f"${float(_raw):,.0f}"
                _color = "#e3b341"
            else:
                _color = "#c9d1d9"
            _mstrip_parts.append(
                f"<div style='text-align:center;padding:0 12px'>"
                f"<div style='font-size:0.62rem;color:#8b949e;text-transform:uppercase;"
                f"letter-spacing:0.1em;margin-bottom:2px'>{_label}</div>"
                f"<div style='font-size:1.0rem;color:{_color};font-weight:700;"
                f"font-family:monospace'>{_display}</div>"
                f"</div>"
            )

    # Add FII
    if _fii_val is not None:
        try:
            _fv = float(_fii_val)
            _fii_display = f"₹{_fv:,.0f} Cr"
            _fii_color = "#3fb950" if _fv > 0 else "#f85149"
        except Exception:
            _fii_display = str(_fii_val)
            _fii_color = "#c9d1d9"
        _mstrip_parts.append(
            f"<div style='text-align:center;padding:0 16px'>"
            f"<div style='font-size:0.65rem;color:#8b949e;text-transform:uppercase;"
            f"letter-spacing:0.1em;margin-bottom:2px'>{_fii_label}</div>"
            f"<div style='font-size:1.1rem;color:{_fii_color};font-weight:700;"
            f"font-family:monospace'>{_fii_display}</div>"
            f"</div>"
        )

    if _mstrip_parts:
        st.markdown(
            "<div style='display:flex;align-items:center;justify-content:center;"
            "padding:12px 0;border-left:1px solid #1e2d40;border-right:1px solid #1e2d40;"
            "height:100%'>" + "".join(_mstrip_parts) + "</div>",
            unsafe_allow_html=True
        )

with _h3:
    _h3_left, _h3_right = st.columns([1, 1])
    with _h3_right:
        _header_regime = st.empty()
        _header_regime.markdown(
            f"<div style='text-align:right;padding-top:6px'>"
            f"<div style='font-family:monospace;font-size:0.62rem;color:#8b949e;"
            f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:2px'>Market Regime</div>"
            f"{regime_pill(regime_int)}"
            f"</div>",
            unsafe_allow_html=True
        )
    with _h3_left:
        _header_exp_ret_placeholder = st.empty()

st.divider()

# Top KPIs
k1, k2, k3, k4, k5, k6 = st.columns(6)
_mk(k1, "Model BUY Signals",   str(n_buy),                                        "#3fb950")
_mk(k2, "Model SELL Signals",  str(n_sell),                                       "#f85149")
_mk(k3, "Neutral Signals",     str(n_hold),                                       "#8b949e")
_mk(k4, "Universe",            "500",                                             "#c9d1d9")
_mk(k5, "Portfolio Positions", str(opt_stats.n_total) if opt_stats else "—",      "#58a6ff")
_mk(k6, "Simulation NAV",      f"₹{nav_input/1e7:.2f} Cr",                        "#c9d1d9")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab4, tab6 = st.tabs([
    "SIGNALS", "PORTFOLIO", "BACKTEST GRAPHS", "RETURNS & TAXES"
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

        # Merge sector + company name — always use sector_map CSV as ground truth
        # Drop any Sector/Company_Name from signals CSV (may be Unknown from ensemble)
        df_view = signals_today.copy()
        df_view["Ticker_Display"] = df_view["Ticker"].str.replace(".NS","",regex=False)
        df_view = df_view.drop(columns=["Sector","Company_Name"], errors="ignore")
        df_view = df_view.merge(sector_map, on="Ticker", how="left")
        df_view["Sector"] = df_view["Sector"].fillna("Unknown")
        if "Company_Name" not in df_view.columns:
            df_view["Company_Name"] = df_view["Ticker_Display"]
        df_view["Company_Name"] = df_view["Company_Name"].fillna(df_view["Ticker_Display"])
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
            "Signal":       df_show["Signal"].map({1:"BUY", -1:"SELL", 0:"NEUTRAL"}),
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
                    x=[_wrap_label(s) for s in buys_sector["Sector"]],
                    y=buys_sector["Count"],
                    text=buys_sector["Count"],
                    textposition="outside",
                    textfont=dict(color="#3fb950", size=10),
                    marker_color="#3fb950",
                    hovertemplate="%{x}<br>BUY signals: %{y}<extra></extra>",
                    customdata=buys_sector["Sector"],
                    hovertext=buys_sector["Sector"],
                ))
                fig.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=30, b=160),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="JetBrains Mono, monospace", color="#8b949e", size=11),
                    xaxis=dict(
                        title=dict(text="Sector", font=dict(color="#8b949e", size=11)),
                        tickfont=dict(color="#8b949e", size=9),
                        tickangle=-35,
                        gridcolor="#1e2d40", linecolor="#1e2d40",
                    ),
                    yaxis=dict(
                        title=dict(text="Signal Count", font=dict(color="#8b949e", size=11)),
                        tickfont=dict(color="#8b949e", size=10),
                        gridcolor="#1e2d40", linecolor="#1e2d40",
                        showticklabels=False,
                        showgrid=False,
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
                    x=[_wrap_label(s) for s in sells_sector["Sector"]],
                    y=sells_sector["Count"],
                    text=sells_sector["Count"],
                    textposition="outside",
                    textfont=dict(color="#f85149", size=10),
                    marker_color="#f85149",
                    hovertemplate="%{x}<br>SELL signals: %{y}<extra></extra>",
                    customdata=sells_sector["Sector"],
                ))
                fig_sell.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=30, b=160),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="JetBrains Mono, monospace", color="#8b949e", size=11),
                    xaxis=dict(
                        title=dict(text="Sector", font=dict(color="#8b949e", size=11)),
                        tickfont=dict(color="#8b949e", size=9),
                        tickangle=-35,
                        gridcolor="#1e2d40", linecolor="#1e2d40",
                    ),
                    yaxis=dict(
                        title=dict(text="Signal Count", font=dict(color="#8b949e", size=11)),
                        tickfont=dict(color="#8b949e", size=10),
                        gridcolor="#1e2d40", linecolor="#1e2d40",
                        showgrid=False,
                        showticklabels=False,
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
        _mk(m1, "Total Positions", str(opt_stats.n_total),                    "#c9d1d9")
        _mk(m2, "Longs",           str(opt_stats.n_longs),                    "#3fb950")
        _mk(m3, "Shorts",          str(opt_stats.n_shorts),                   "#f85149")
        _mk(m4, "Gross Long",      f"{opt_stats.gross_long_pct:.1%}",         "#58a6ff")
        _mk(m5, "Gross Short",     f"{opt_stats.gross_short_pct:.1%}",        "#58a6ff")
        _mk(m6, "Net Exposure",    f"{opt_stats.net_exposure_pct:.1%}",       "#ffffff")
        _mk(m7, "Cash",            f"{opt_stats.cash_pct:.1%}",               "#ffffff")

        st.divider()

        # ── Pie charts ────────────────────────────────────────────────────
        import plotly.graph_objects as go

        pc1, pc2 = st.columns(2)

        with pc1:
            st.markdown("<div class='section-header'>Portfolio Distribution</div>",
                        unsafe_allow_html=True)
            if not proposed_df.empty:
                # Use ticker as short label inside chart, full company name on hover
                _ticker_short = proposed_df["Ticker"].str.replace(".NS","",regex=False)
                _raw_name     = proposed_df.get("Company_Name", _ticker_short).fillna(_ticker_short)
                # Company_Name sometimes contains "Name,Direction,Sector" from CSV — strip after first comma
                _pie_name     = _raw_name.astype(str).str.split(",").str[0].str.strip()
                # Sector: get directly from proposed_df (not from the fused Company_Name field)
                _pie_sector   = proposed_df["Sector"].fillna("—") if "Sector" in proposed_df.columns else pd.Series(["—"]*len(proposed_df))
                pie_values    = proposed_df["Size_%NAV"]

                # High-contrast alternating greens for longs, reds for shorts
                # Use shade variation so adjacent slices are distinguishable
                _long_palette  = ["#2ea043","#3fb950","#56d364","#26a641","#1a7f37",
                                   "#4ac26b","#6fdd8b","#34d058","#28a745","#218838"]
                _short_palette = ["#da3633","#f85149","#ff6b6b","#c0392b","#e74c3c",
                                   "#ff4444","#d63031","#b71c1c","#ff5252","#e53935"]
                _long_idx = _short_idx = 0
                pie_colors = []
                for d in proposed_df["Direction"]:
                    if d == "LONG":
                        pie_colors.append(_long_palette[_long_idx % len(_long_palette)])
                        _long_idx += 1
                    else:
                        pie_colors.append(_short_palette[_short_idx % len(_short_palette)])
                        _short_idx += 1

                # Pre-format hover text as single string per slice (avoids Plotly customdata[n] issues)
                _hover_texts = [
                    f"<b>{name}</b><br>{sector}<br>{val:.2f}% NAV"
                    for name, sector, val in zip(_pie_name, _pie_sector, pie_values)
                ]

                fig_pie1 = go.Figure(go.Pie(
                    labels=_ticker_short,           # short ticker inside chart
                    values=pie_values,
                    customdata=_hover_texts,
                    marker=dict(
                        colors=pie_colors,
                        line=dict(color="#0a0e1a", width=2)
                    ),
                    hovertemplate="%{customdata}<extra></extra>",
                    # Show only percent inside slice, ticker as pull-out label
                    textinfo="label+percent",
                    textposition="inside",
                    textfont=dict(size=9, color="#000000", family="JetBrains Mono, monospace"),
                    insidetextorientation="radial",
                    hole=0.38,
                    pull=[0.03 if d=="SHORT" else 0 for d in proposed_df["Direction"]],
                ))
                fig_pie1.update_layout(
                    height=480,
                    margin=dict(l=80, r=80, t=30, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    font=dict(family="JetBrains Mono, monospace", color="#c9d1d9", size=10),
                    uniformtext=dict(minsize=7, mode="hide"),
                )
                st.plotly_chart(fig_pie1, use_container_width=True)

        with pc2:
            st.markdown("<div class='section-header'>Sector Allocation (% NAV)</div>",
                        unsafe_allow_html=True)
            if not proposed_df.empty:
                sec_nav    = proposed_df.groupby("Sector")["Size_%NAV"].sum().reset_index()
                sec_counts = proposed_df.groupby("Sector").size().reindex(sec_nav["Sector"]).values

                # Sector pie — use distinct color palette
                _sec_palette = [
                    "#58a6ff","#3fb950","#e3b341","#f85149","#bc8cff",
                    "#ff9800","#00bcd4","#e91e63","#8bc34a","#ff5722",
                    "#9c27b0","#03a9f4","#4caf50","#ffc107","#f44336",
                ]
                fig_pie2 = go.Figure(go.Pie(
                    labels=sec_nav["Sector"],
                    values=sec_nav["Size_%NAV"].round(2),
                    customdata=sec_counts,
                    marker=dict(
                        colors=_sec_palette[:len(sec_nav)],
                        line=dict(color="#0a0e1a", width=2)
                    ),
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "NAV: %{value:.2f}%<br>"
                        "Stocks: %{customdata}"
                        "<extra></extra>"
                    ),
                    textinfo="label+percent",
                    textposition="inside",
                    textfont=dict(size=11, color="#000000", family="JetBrains Mono, monospace"),
                    insidetextorientation="radial",
                    hole=0.38,
                ))
                fig_pie2.update_layout(
                    height=480,
                    margin=dict(l=80, r=80, t=30, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=True,
                    legend=dict(
                        font=dict(color="#c9d1d9", size=10,
                                  family="JetBrains Mono, monospace"),
                        bgcolor="rgba(0,0,0,0)",
                        orientation="v",
                        x=1.02, y=0.5,
                    ),
                    font=dict(family="JetBrains Mono, monospace", color="#c9d1d9", size=10),
                    uniformtext=dict(minsize=7, mode="hide"),
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

        # ── Rejected candidates ───────────────────────────────────────────
        if not rejected_df.empty:
            st.divider()
            with st.expander(f"Rejected candidates ({len(rejected_df)})", expanded=False):
                st.dataframe(rejected_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — GRAPHS
# ══════════════════════════════════════════════════════════════════════════════

with tab4:

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    nav_df = load_nav_series()

    if nav_df.empty:
        st.warning(
            "No NAV series file found. Download `nav_series_*.csv` from "
            "Google Drive and place it in `exports/model_output/`."
        )
    else:
        # Identify date and NAV columns
        date_col = next((c for c in nav_df.columns if "date" in c.lower()), nav_df.columns[0])
        nav_col  = next((c for c in nav_df.columns
                         if c != date_col and any(k in c.lower() for k in ("nav","value","portfolio","return"))),
                        [c for c in nav_df.columns if c != date_col][0])

        nav_df[date_col] = pd.to_datetime(nav_df[date_col])
        start_str = nav_df[date_col].min().strftime("%Y-%m-%d")
        end_str   = nav_df[date_col].max().strftime("%Y-%m-%d")

        # Normalise portfolio to base 100
        base          = nav_df[nav_col].iloc[0]
        nav_df["Portfolio_Idx"] = nav_df[nav_col] / base * 100

        # Load Nifty benchmark
        with st.spinner("Fetching Nifty benchmark from yfinance…"):
            nifty_df = load_nifty_index(start_str, end_str)

        if not nifty_df.empty:
            nifty_df["Nifty_Idx"] = nifty_df["Nifty"] / nifty_df["Nifty"].iloc[0] * 100
            merged = nav_df[[date_col, "Portfolio_Idx"]].merge(
                nifty_df[["Date", "Nifty_Idx"]], left_on=date_col, right_on="Date", how="left"
            )
        else:
            merged = nav_df[[date_col, "Portfolio_Idx"]].copy()
            merged["Nifty_Idx"] = float("nan")

        merged = merged.rename(columns={date_col: "Date"})

        # ── Summary return metrics ────────────────────────────────────────
        port_total_ret = (merged["Portfolio_Idx"].iloc[-1] / 100 - 1) * 100
        nifty_total_ret = (merged["Nifty_Idx"].iloc[-1] / 100 - 1) * 100 if merged["Nifty_Idx"].notna().any() else None
        n_days = (merged["Date"].iloc[-1] - merged["Date"].iloc[0]).days
        n_years = max(n_days / 365.25, 0.001)
        port_cagr = ((merged["Portfolio_Idx"].iloc[-1] / 100) ** (1 / n_years) - 1) * 100
        nifty_cagr = ((merged["Nifty_Idx"].iloc[-1] / 100) ** (1 / n_years) - 1) * 100 if nifty_total_ret is not None else None

        # ── Returns chart ─────────────────────────────────────────────────
        st.markdown("<div class='section-header'>Portfolio vs Nifty 500</div>",
                    unsafe_allow_html=True)

        gm1, gm2, gm3, gm4 = st.columns(4)
        _mk(gm1, "Portfolio Total Return", f"{port_total_ret:+.1f}%", "#3fb950")
        _mk(gm2, "Portfolio CAGR",         f"{port_cagr:+.1f}%",      "#3fb950")
        if nifty_total_ret is not None and not pd.isna(nifty_total_ret):
            _mk(gm3, "Nifty 500 Total Return", f"{nifty_total_ret:+.1f}%", "#e3b341")
            _nifty_cagr_val = f"{nifty_cagr:+.1f}%" if nifty_cagr and not pd.isna(nifty_cagr) else "—"
            _mk(gm4, "Nifty 500 CAGR", _nifty_cagr_val, "#e3b341")
        else:
            _mk(gm3, "Nifty 500 Total Return", "—", "#e3b341")
            _mk(gm4, "Nifty 500 CAGR", "—", "#e3b341")

        st.markdown("<div style='margin-top: 20px'></div>", unsafe_allow_html=True)

        fig_ret = go.Figure()
        fig_ret.add_trace(go.Scatter(
            x=merged["Date"], y=merged["Portfolio_Idx"].round(2),
            name="Portfolio", line=dict(color="#3fb950", width=2.5, dash="dash"),
            hovertemplate="Date: %{x|%d %b %Y}<br>Portfolio: %{y:.2f}<extra></extra>",
            fill="tozeroy", fillcolor="rgba(63,185,80,0.06)",
        ))
        if merged["Nifty_Idx"].notna().any():
            fig_ret.add_trace(go.Scatter(
                x=merged["Date"], y=merged["Nifty_Idx"].round(2),
                name="Nifty 500", line=dict(color="#e3b341", width=1.5, dash="dot"),
                hovertemplate="Date: %{x|%d %b %Y}<br>Nifty 500: %{y:.2f}<extra></extra>",
            ))
        _ret_max = max(
            merged["Portfolio_Idx"].max() if not merged["Portfolio_Idx"].empty else 100,
            merged["Nifty_Idx"].max() if merged["Nifty_Idx"].notna().any() else 100,
        ) * 1.05
        fig_ret.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color="#8b949e", size=11),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9")),
            hovermode="x unified",
            xaxis=dict(gridcolor="#1e2d40", linecolor="#1e2d40",
                       tickfont=dict(color="#8b949e")),
            yaxis=dict(title="Indexed Return (Base = 100)", gridcolor="#1e2d40",
                       linecolor="#1e2d40", tickfont=dict(color="#8b949e"),
                       range=[80, _ret_max]),
        )
        st.plotly_chart(fig_ret, use_container_width=True)

        st.divider()

        # ── Drawdown chart ────────────────────────────────────────────────
        st.markdown("<div class='section-header'>Portfolio Drawdown</div>",
                    unsafe_allow_html=True)

        port_series = merged["Portfolio_Idx"]
        rolling_max = port_series.cummax()
        drawdown    = ((port_series - rolling_max) / rolling_max * 100).round(2)
        max_dd      = drawdown.min()
        max_dd_date = merged["Date"].iloc[drawdown.idxmin()]

        _dd_l, dd_metric1, dd_metric2, _dd_r = st.columns([1, 1, 1, 1])
        _mk(dd_metric1, "Max Drawdown", f"{max_dd:.2f}%", "#f85149")
        _mk(dd_metric2, "Max DD Date",  max_dd_date.strftime("%d %b %Y"), "#c9d1d9")

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=merged["Date"], y=drawdown,
            name="Drawdown", line=dict(color="#f85149", width=1.5),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.12)",
            hovertemplate="Date: %{x|%d %b %Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
        ))
        fig_dd.add_hline(y=0, line=dict(color="#484f58", width=1))
        fig_dd.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color="#8b949e", size=11),
            hovermode="x unified",
            xaxis=dict(gridcolor="#1e2d40", linecolor="#1e2d40",
                       tickfont=dict(color="#8b949e")),
            yaxis=dict(title="Drawdown (%)", gridcolor="#1e2d40",
                       linecolor="#1e2d40", tickfont=dict(color="#8b949e")),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        st.divider()

        # ── Rolling Sharpe chart ──────────────────────────────────────────
        st.markdown("<div class='section-header'>Rolling Sharpe Ratio (63-Day)</div>",
                    unsafe_allow_html=True)

        port_daily_rets = merged["Portfolio_Idx"].pct_change().dropna()
        roll_window     = 63
        rolling_sharpe  = (
            port_daily_rets.rolling(roll_window).mean() /
            port_daily_rets.rolling(roll_window).std()
        ) * np.sqrt(252)
        rolling_sharpe  = rolling_sharpe.reindex(merged.index)

        # Drop leading NaN so chart starts flush with first valid Sharpe value
        _first_valid    = rolling_sharpe.first_valid_index()
        _rs_dates       = merged["Date"]
        if _first_valid is not None:
            _rs_mask    = merged.index >= _first_valid
            _rs_dates   = merged.loc[_rs_mask, "Date"]
            rolling_sharpe = rolling_sharpe.loc[_rs_mask]

        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(
            x=_rs_dates, y=rolling_sharpe.round(3),
            name="Rolling Sharpe", line=dict(color="#bc8cff", width=1.5),
            hovertemplate="Date: %{x|%d %b %Y}<br>Sharpe: %{y:.3f}<extra></extra>",
            fill="tozeroy", fillcolor="rgba(188,140,255,0.08)",
        ))
        fig_rs.add_hline(y=0,   line=dict(color="#484f58", width=1))
        fig_rs.add_hline(y=1.0, line=dict(color="#3fb950", width=1, dash="dot"),
                         annotation_text="Sharpe=1",
                         annotation_font=dict(color="#3fb950", size=10))
        fig_rs.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color="#8b949e", size=11),
            hovermode="x unified",
            xaxis=dict(gridcolor="#1e2d40", linecolor="#1e2d40",
                       tickfont=dict(color="#8b949e")),
            yaxis=dict(title="Sharpe Ratio", gridcolor="#1e2d40",
                       linecolor="#1e2d40", tickfont=dict(color="#8b949e"),
                       zeroline=True, zerolinecolor="#484f58"),
        )
        st.plotly_chart(fig_rs, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RETURNS & FEES
# ══════════════════════════════════════════════════════════════════════════════

with tab6:

    import plotly.graph_objects as go

    if proposed_df.empty or opt_stats is None:
        st.warning("No portfolio positions available. Run optimizer first.")
    else:
        stcg_rate, ltcg_rate = _tax_rates(signal_date)
        budget_label = "Post-Budget 2024" if pd.Timestamp(signal_date) >= _BUDGET_2024_CUTOFF else "Pre-Budget 2024"

        # ── Capital deployment summary ────────────────────────────────────
        longs_rf  = proposed_df[proposed_df["Direction"] == "LONG"]
        shorts_rf = proposed_df[proposed_df["Direction"] == "SHORT"]

        total_deployed_rs  = proposed_df["Alloc_Rs"].sum()
        long_deployed_rs   = longs_rf["Alloc_Rs"].sum()
        short_deployed_rs  = shorts_rf["Alloc_Rs"].sum()
        cash_remaining_rs  = nav_input - total_deployed_rs

        # Transaction costs on entry (buy-side for longs, sell-side for shorts)
        cost_rate = _TOTAL_COST_BPS / 10000
        entry_costs_rs = total_deployed_rs * cost_rate
        # Exit costs assumed same rate on exit (round-trip = 2×)
        exit_costs_rs  = total_deployed_rs * cost_rate
        total_costs_rs = entry_costs_rs + exit_costs_rs

        st.markdown("<div class='section-header'>Capital Summary</div>", unsafe_allow_html=True)
        cs1, cs2, cs3, cs4, cs5 = st.columns(5)
        _mk(cs1, "Total NAV",      f"₹{nav_input:,.0f}",         "#c9d1d9")
        _mk(cs2, "Total Deployed", f"₹{total_deployed_rs:,.0f}", "#c9d1d9")
        _mk(cs3, "Long Book",      f"₹{long_deployed_rs:,.0f}",  "#3fb950")
        _mk(cs4, "Short Book",     f"₹{short_deployed_rs:,.0f}", "#f85149")
        _mk(cs5, "Cash Remaining", f"₹{cash_remaining_rs:,.0f}", "#ffffff")

        st.divider()

        # ── Expected returns per position ─────────────────────────────────
        # _has_calib must be defined before the description markdown references it
        _has_calib = "Projected_Return_21d" in signals_today.columns

        st.markdown("<div class='section-header'>Expected Returns & Tax per Position</div>",
                    unsafe_allow_html=True)

        # Merge calibrated returns onto proposed_df via bare ticker (no .NS)
        # Drop any stale calib columns first to prevent _x/_y suffix on re-renders
        _calib_data_cols = ["Projected_Return_21d", "Projected_Return_252d",
                            "Band_Low_21d", "Band_High_21d"]
        proposed_df = proposed_df.drop(
            columns=[c for c in _calib_data_cols if c in proposed_df.columns],
            errors="ignore"
        )
        if _has_calib:
            _calib_cols = ["Ticker"] + [c for c in _calib_data_cols if c in signals_today.columns]
            _sig_latest_calib = signals_today[
                signals_today["Date"] == signals_today["Date"].max()
            ][_calib_cols].copy()
            # Normalise to bare ticker on both sides for reliable join
            _sig_latest_calib["_Ticker_bare"] = _sig_latest_calib["Ticker"].str.replace(
                r"\.NS$", "", regex=True
            )
            _sig_latest_calib = (
                _sig_latest_calib
                .drop(columns=["Ticker"])
                .drop_duplicates("_Ticker_bare")
            )
            proposed_df["_Ticker_bare"] = proposed_df["Ticker"].str.replace(
                r"\.NS$", "", regex=True
            )
            proposed_df = proposed_df.merge(_sig_latest_calib, on="_Ticker_bare", how="left")
            proposed_df = proposed_df.drop(columns=["_Ticker_bare"], errors="ignore")

        # Always (re-)merge Final_Rank from signals_today to ensure distinct per-ticker values.
        # The optimizer may store rounded ranks; signals_today has the raw per-ticker values.
        # NOTE: Ticker format must be normalised before merging (.NS suffix may differ between
        #       proposed_df (always .NS) and signals_today (may or may not have .NS).
        if "Final_Rank" in signals_today.columns:
            _sig_latest = signals_today[signals_today["Date"] == signals_today["Date"].max()].copy()
            # Normalise both sides to bare ticker (no .NS) for reliable join
            _sig_latest["_Ticker_bare"] = _sig_latest["Ticker"].str.replace(r"\.NS$", "", regex=True)
            _sig_latest_ranks = (
                _sig_latest[["_Ticker_bare", "Final_Rank"]]
                .drop_duplicates("_Ticker_bare")
            )
            # Keep original Final_Rank from optimizer as fallback (already distinct per ticker)
            _orig_ranks = proposed_df[["Ticker", "Final_Rank"]].copy() if "Final_Rank" in proposed_df.columns else None
            proposed_df["_Ticker_bare"] = proposed_df["Ticker"].str.replace(r"\.NS$", "", regex=True)
            proposed_df = proposed_df.drop(columns=["Final_Rank"], errors="ignore")
            proposed_df = proposed_df.merge(_sig_latest_ranks, on="_Ticker_bare", how="left")
            proposed_df = proposed_df.drop(columns=["_Ticker_bare"], errors="ignore")
            # If re-merge failed (all NaN), fall back to optimizer's own per-ticker ranks
            if proposed_df["Final_Rank"].isna().all() and _orig_ranks is not None:
                proposed_df = proposed_df.merge(_orig_ranks, on="Ticker", how="left")
            proposed_df["Final_Rank"] = proposed_df["Final_Rank"].fillna(0.5)

        ret_rows = []
        for _, row in proposed_df.iterrows():
            direction  = row["Direction"]
            alloc_rs   = row["Alloc_Rs"]
            rank       = float(row.get("Final_Rank", 0.5))
            is_midcap  = bool(row.get("is_midcap", False))

            # Hold period and tax classification
            if direction == "LONG":
                hold_months  = 9 if is_midcap else 12
                is_long_term = not is_midcap
                hold_label   = "12 mo (LTCG)" if is_long_term else "9 mo (STCG)"
            else:
                hold_months  = 4
                is_long_term = False
                hold_label   = "4 mo (STCG)"

            # Use calibrated return if available, else fall back to formula
            proj_21d = row.get("Projected_Return_21d", float("nan"))
            if _has_calib and pd.notna(proj_21d):
                # Calibrated: proj_21d is a 21-trading-day fractional return
                # Annualise: 252 / 21 = 12 periods per year
                ann_ret        = float(proj_21d) * (252 / 21) * 100
                # Period return: scale annual down to hold period (months out of 12)
                exp_ret_period = ann_ret * hold_months / 12
                # Cap at reasonable bounds
                ann_ret        = max(-50.0, min(ann_ret, 100.0))
                exp_ret_period = max(-50.0, min(exp_ret_period, 100.0))
                source_label   = "calibrated"
            else:
                # Formula fallback (pre-calibration): Final_Rank × 30% max annual return proxy
                if direction == "LONG":
                    ann_ret = rank * 30.0   # rank 1.0 → 30%, rank 0.5 → 15%, rank 0.0 → 0%
                else:
                    ann_ret = (1.0 - rank) * 30.0   # inverse for shorts
                exp_ret_period = ann_ret / 12 * hold_months
                source_label   = "formula"

            gross_gain_rs   = alloc_rs * exp_ret_period / 100
            applicable_rate = ltcg_rate if is_long_term else stcg_rate
            taxable_gain    = max(0, gross_gain_rs - (_LTCG_EXEMPTION_INR if is_long_term else 0))
            tax_rs          = taxable_gain * applicable_rate
            net_gain_rs     = gross_gain_rs - tax_rs   # Net = Gross - Tax

            ret_rows.append({
                "Ticker":              row["Ticker"].replace(".NS",""),
                "Direction":           direction,
                "Sector":              row.get("Sector","—"),
                "Hold Period":         hold_label,
                "Deployed (₹)":       f"₹{alloc_rs:,.0f}",
                "Gross Expected Gain (₹)": f"₹{gross_gain_rs:,.0f}",
                "Tax Rate":            f"{applicable_rate:.0%}",
                "Tax (₹)":             f"₹{tax_rs:,.0f}",
                "Net Gain (₹)":        f"₹{net_gain_rs:,.0f}",
            })

        ret_df = pd.DataFrame(ret_rows)

        # ── Fill projected performance block (defined above capital summary) ──
        if total_deployed_rs > 0 and ret_rows:
            _gross_gain_sum = sum(
                float(r["Gross Expected Gain (₹)"].replace("₹","").replace(",","")) for r in ret_rows
            )
            _tax_sum = sum(
                float(r["Tax (₹)"].replace("₹","").replace(",","")) for r in ret_rows
            )
            _tc_sum = total_deployed_rs * 2 * (_TOTAL_COST_BPS / 10000)
            _exp_ret_pct = (_gross_gain_sum / total_deployed_rs) * 100
            _abs_ret_pct = ((_gross_gain_sum - _tax_sum - _tc_sum) / total_deployed_rs) * 100
            _exp_color   = "#3fb950" if _exp_ret_pct >= 0 else "#f85149"
            _header_exp_ret_placeholder.markdown(
                f"<div style='text-align:center;padding:0 12px;padding-top:6px'>"
                f"<div style='font-size:0.62rem;color:#8b949e;text-transform:uppercase;"
                f"letter-spacing:0.1em;margin-bottom:2px'>Expected Return</div>"
                f"<div style='font-size:1.0rem;color:{_exp_color};font-weight:700;"
                f"font-family:monospace'>{_exp_ret_pct:+.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        def colour_direction(val):
            if val == "LONG":  return "color: #3fb950; font-weight:600"
            if val == "SHORT": return "color: #f85149; font-weight:600"
            return ""

        st.dataframe(
            ret_df.style.map(colour_direction, subset=["Direction"]),
            use_container_width=True, hide_index=True,
            height=min(50 + len(ret_df)*38, 500),
        )

        st.divider()

        # ── Portfolio-level summary ────────────────────────────────────────
        # Compute total_fees_rs for summary and waterfall (no display section)
        total_fees_rs = sum([
            total_deployed_rs * 2 * (_STT_BPS / 10000),
            total_deployed_rs * 2 * (_BROKERAGE_BPS / 10000),
            total_deployed_rs * (_STAMP_DUTY_BPS / 10000),
            total_deployed_rs * 2 * (_SLIPPAGE_BPS / 10000),
            total_deployed_rs * 2 * (_EXCHANGE_BPS / 10000),
        ])
        st.markdown("<div class='section-header'>Portfolio-Level Summary</div>",
                    unsafe_allow_html=True)

        total_gross_gain = sum(
            float(r["Gross Expected Gain (₹)"].replace("₹","").replace(",","")) for r in ret_rows
        )
        total_tax        = sum(
            float(r["Tax (₹)"].replace("₹","").replace(",","")) for r in ret_rows
        )
        total_net        = sum(
            float(r["Net Gain (₹)"].replace("₹","").replace(",","")) for r in ret_rows
        )

        def _cm(col, label, value, color):
            _mk(col, label, value, color)

        ps1, ps2, ps3, ps4, ps5 = st.columns(5)
        _mk(ps1, "Total Deployed",         f"₹{total_deployed_rs:,.0f}", "#c9d1d9")
        _mk(ps2, "Gross Expected Gain",    f"₹{total_gross_gain:,.0f}",  "#3fb950")
        _mk(ps3, "Total Transaction Costs",f"₹{total_fees_rs:,.0f}",     "#f85149")
        _mk(ps4, "Total Tax (Projected)",  f"₹{total_tax:,.0f}",         "#f85149")
        _mk(ps5, "Net Expected Gain",      f"₹{total_net:,.0f}",         "#3fb950")

        # ── Waterfall chart: NAV → Deployed → Costs → Tax → Net ──────────
        st.markdown("<div class='section-header'>Capital Waterfall</div>",
                    unsafe_allow_html=True)

        # Waterfall: Starting NAV → Cash Held → Gross Gain → Costs → Tax → Final NAV
        # cash_held = undeployed cash sitting aside (not put to work)
        cash_held = cash_remaining_rs
        final_nav = nav_input - cash_held + total_gross_gain - total_fees_rs - total_tax + cash_held
        # Simplified: final_nav = nav_input + net_gain
        final_nav = nav_input + total_net

        wf_labels  = ["Starting NAV", "Cash Held", "Gross Gain",
                      "Transaction Costs", "Tax", "Final NAV"]
        wf_measures= ["absolute", "relative", "relative", "relative", "relative", "total"]
        wf_values  = [
            nav_input,           # absolute start
            -cash_remaining_rs,  # cash held aside (reduces deployable)
            total_gross_gain,    # expected gains from deployed capital
            -total_fees_rs,      # transaction costs
            -total_tax,          # projected tax
            final_nav,           # total bar
        ]

        # Format y-axis tick labels in Cr/L for readability
        def _fmt_inr(v):
            a = abs(v)
            if a >= 1e7: return f"₹{v/1e7:.1f}Cr"
            if a >= 1e5: return f"₹{v/1e5:.1f}L"
            return f"₹{v:,.0f}"

        fig_wf = go.Figure(go.Waterfall(
            name="Capital", orientation="v",
            measure=wf_measures,
            x=wf_labels, y=wf_values,
            base=0,
            connector=dict(
                line=dict(color="#484f58", width=1, dash="dot"),
                visible=True,
            ),
            increasing=dict(marker=dict(color="#3fb950", line=dict(width=0))),
            decreasing=dict(marker=dict(color="#f85149", line=dict(width=0))),
            totals=dict(marker=dict(color="#58a6ff", line=dict(width=0))),
            text=[_fmt_inr(v) for v in wf_values],
            textposition="outside",
            textfont=dict(color="#c9d1d9", size=10),
        ))
        fig_wf.update_layout(
            height=420, margin=dict(l=10, r=10, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,14,26,0.6)",
            font=dict(family="JetBrains Mono, monospace", color="#8b949e", size=11),
            xaxis=dict(tickfont=dict(color="#8b949e", size=11), gridcolor="#1e2d40",
                       linecolor="#30363d"),
            yaxis=dict(
                title="Amount (₹)",
                tickfont=dict(color="#8b949e"),
                gridcolor="#1e2d40",
                linecolor="#30363d",
                rangemode="tozero",
                tickformat=",",
            ),
            showlegend=False,
        )
        # Add zero baseline
        fig_wf.add_hline(y=0, line=dict(color="#30363d", width=1))
        st.plotly_chart(fig_wf, use_container_width=True)

        st.markdown(
            f"<div style='font-family:monospace;font-size:0.72rem;color:#8b949e;margin-top:8px'>"
            f"Tax regime: <span style='color:#c9d1d9'>{budget_label}</span> · "
            f"STCG: <span style='color:#e3b341'>{stcg_rate:.0%}</span> · "
            f"LTCG: <span style='color:#e3b341'>{ltcg_rate:.0%}</span> "
            f"(₹1L annual exemption applies) · "
            f"Transaction costs: <span style='color:#e3b341'>{_TOTAL_COST_BPS:.2f}bps</span>"
            f"</div>",
            unsafe_allow_html=True
        )