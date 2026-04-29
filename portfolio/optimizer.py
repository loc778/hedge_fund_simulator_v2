from __future__ import annotations
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJECT_ROOT)

from risk.risk_manager import (
    PortfolioState,
    OpenPosition,
    check_position,
    MAX_SINGLE_CORE_LONG_PCT,
    MAX_SINGLE_MIDCAP_LONG_PCT,
    MAX_SINGLE_SHORT_PCT,
    CASH_RESERVE_MIN_PCT,
    MIN_POSITION_SIZE_PCT,
    FINANCIAL_SECTOR_NAMES,
)

# ── Optimizer constants ───────────────────────────────────────────────────────

MAX_POSITIONS         = 45      # hard cap: longs + shorts combined (fewer, higher conviction)
MIN_POSITIONS         = 25      # floor (warning only, not enforced)
LONG_RANK_THRESHOLD   = 0.82    # Final_Rank >= this → BUY candidate (slightly wider to fill slots)
SHORT_RANK_THRESHOLD  = 0.08    # Final_Rank <= this → SELL candidate (tighter = fewer shorts)

LONG_TARGET_PCT       = 1.30    # ↑ target gross long as fraction of NAV (was 1.15)
SHORT_TARGET_PCT      = 0.10    # ↓ target gross short (was 0.15) — less drag, more net long
LONG_HARD_CAP_PCT     = 1.35    # hard cap gross long
SHORT_HARD_CAP_PCT    = 0.15    # hard cap gross short

# Regime-aware maximum deployable NAV fraction (cash floor is the complement)
# Raised across the board — the old values left too much idle cash suppressing returns
REGIME_DEPLOY_CAP = {
    0: 0.97,   # Bull       — nearly fully deployed
    1: 0.88,   # Bear       — more cautious but still active
    2: 0.90,   # High Vol   — was 0.85, slight raise; ATR sizing auto-reduces anyway
    3: 0.94,   # Sideways   — was 0.90
}

# Tier → is_midcap mapping (Tier 2 = mid-cap proxy)
TIER_IS_MIDCAP = {1: False, 2: True, 3: True}

# ATR fallback when indicator data is missing
ATR_PCT_FALLBACK = 0.02   # 2% daily vol assumed


# ─────────────────────────────────────────────────────────────────────────────
# Return types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioStats:
    """
    Book-level summary statistics for the optimized portfolio.
    All pct fields are fractions of NAV (e.g. 0.85 = 85%).
    """
    nav                : float
    regime             : int
    regime_label       : str
    deploy_cap         : float          # regime-aware max deployable
    gross_long_pct     : float
    gross_short_pct    : float
    net_exposure_pct   : float
    gross_exposure_pct : float
    cash_pct           : float
    n_longs            : int
    n_shorts           : int
    n_total            : int
    n_rejected         : int
    long_sectors       : int            # distinct sectors in long book
    below_min_warning  : bool           # True if n_total < MIN_POSITIONS


@dataclass
class OptimizedPortfolio:
    """
    Complete output of optimize_portfolio().

    positions       : approved book — one row per position
    rejected        : candidates that did not pass risk checks
    stats           : book-level metrics
    portfolio_state : PortfolioState snapshot after all positions added
                      (pass this to risk_manager for subsequent checks)
    """
    positions       : pd.DataFrame
    rejected        : pd.DataFrame
    stats           : PortfolioStats
    portfolio_state : PortfolioState


# ─────────────────────────────────────────────────────────────────────────────
# ATR loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_atr(tickers: list[str], engine) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["Ticker", "ATR_14", "Adj_Close", "ATR_pct"])

    try:
        import config as cfg
        table = cfg.TABLES.get("indicators", "nifty500_indicators")
    except ImportError:
        table = "nifty500_indicators"

    # Build both .NS and bare variants — indicators table may use either format
    tickers_bare = [t.replace(".NS", "") for t in tickers]
    tickers_all  = list(set(tickers) | set(tickers_bare))
    ticker_list  = ", ".join(f"'{t}'" for t in tickers_all)

    # nifty500_indicators has ATR_14 but NOT Adj_Close
    # nifty500_ohlcv has Adj_Close — join the two tables
    try:
        import config as cfg_inner
        ohlcv_table = cfg_inner.TABLES.get("ohlcv", "nifty500_ohlcv")
    except ImportError:
        ohlcv_table = "nifty500_ohlcv"

    query = f"""
        SELECT i.Ticker, i.Date, i.ATR_14, o.Adj_Close
        FROM {table} i
        INNER JOIN (
            SELECT Ticker, MAX(Date) AS MaxDate
            FROM {table}
            WHERE Ticker IN ({ticker_list})
              AND ATR_14 IS NOT NULL
              AND ATR_14 > 0
            GROUP BY Ticker
        ) m ON i.Ticker = m.Ticker AND i.Date = m.MaxDate
        LEFT JOIN {ohlcv_table} o
            ON o.Ticker = i.Ticker AND o.Date = i.Date
        WHERE o.Adj_Close IS NOT NULL AND o.Adj_Close > 0
    """

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
    except Exception as e:
        warnings.warn(f"[optimizer] ATR/price load failed: {e}. Using fallback values.")
        df = pd.DataFrame(columns=["Ticker", "ATR_14", "Adj_Close"])

    if df.empty:
        warnings.warn("[optimizer] ATR query returned no rows — check ticker format in DB. Using fallback.")
        df = pd.DataFrame({"Ticker": tickers})
        df["ATR_14"]   = float("nan")
        df["Adj_Close"]= float("nan")

    df["ATR_14"]    = pd.to_numeric(df["ATR_14"],    errors="coerce")
    df["Adj_Close"] = pd.to_numeric(df["Adj_Close"], errors="coerce")

    # Normalise Ticker to .NS format so merge with signals (which use .NS) succeeds
    df["Ticker"] = df["Ticker"].apply(
        lambda x: x if str(x).endswith(".NS") else str(x) + ".NS"
    )
    # Drop duplicates after normalisation (keep highest ATR_14 row per ticker)
    df = df.sort_values("ATR_14", ascending=False).drop_duplicates("Ticker")

    # ATR_pct = ATR_14 / Adj_Close, fallback if either is missing/zero
    df["ATR_pct"] = df["ATR_14"] / df["Adj_Close"].replace(0, float("nan"))
    df["ATR_pct"] = df["ATR_pct"].fillna(ATR_PCT_FALLBACK).clip(lower=0.005)

    return df[["Ticker", "ATR_14", "Adj_Close", "ATR_pct"]]


# ─────────────────────────────────────────────────────────────────────────────
# Sizing functions
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sizes(
    candidates: pd.DataFrame,
    direction: str,
    nav: float,
    book_target_pct: float,
    deploy_cap: float,
) -> pd.Series:
    if candidates.empty:
        return pd.Series(dtype=float)

    # Rank-conviction-weighted inverse-vol sizing:
    # Combine inverse ATR (risk parity) with Final_Rank signal strength.
    # This ensures rank-0.99 names are sized up vs rank-0.85 names.
    inv_vol = 1.0 / candidates["ATR_pct"].clip(lower=0.005)

    if "Final_Rank" in candidates.columns:
        if direction == "long":
            # For longs: rank above threshold carries conviction weight
            conviction = candidates["Final_Rank"].clip(lower=0.0, upper=1.0)
        else:
            # For shorts: inverted rank (lower rank = more conviction short)
            conviction = (1.0 - candidates["Final_Rank"]).clip(lower=0.0, upper=1.0)
        # Blend: 60% risk-parity, 40% conviction — prevents pure vol dominance
        combined_w = 0.60 * inv_vol + 0.40 * (conviction / conviction.sum() * len(conviction))
    else:
        combined_w = inv_vol

    raw_w   = combined_w / combined_w.sum()

    # Per-position cap depends on direction and tier
    if direction == "long":
        caps = candidates["Data_Tier"].map(TIER_IS_MIDCAP).map(
            {True: MAX_SINGLE_MIDCAP_LONG_PCT, False: MAX_SINGLE_CORE_LONG_PCT}
        ).fillna(MAX_SINGLE_CORE_LONG_PCT)
    else:
        caps = pd.Series(MAX_SINGLE_SHORT_PCT, index=candidates.index)

    # Scale to book target, but respect deploy cap
    effective_target = min(book_target_pct, deploy_cap)
    raw_sizes = raw_w * effective_target

    # Clip to per-position cap
    clipped = raw_sizes.clip(upper=caps.values)

    # Re-normalise so total = effective_target after clipping
    # (iterative: clipping changes total, so renormalise once)
    total_after_clip = clipped.sum()
    if total_after_clip > 0:
        scale   = min(effective_target / total_after_clip, 1.0)
        clipped = (clipped * scale).clip(upper=caps.values)

    # Drop anything below minimum
    clipped = clipped.where(clipped >= MIN_POSITION_SIZE_PCT, other=float("nan"))

    result = clipped.copy()
    result.index = candidates["Ticker"].values
    return result.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Main optimizer
# ─────────────────────────────────────────────────────────────────────────────

def optimize_portfolio(
    signals_df: pd.DataFrame,
    nav: float,
    regime_int: int,
    engine,
    sector_map_df: pd.DataFrame,
    signal_date: Optional[date] = None,
) -> OptimizedPortfolio:

    REGIME_LABELS = {0: "Bull", 1: "Bear", 2: "High Vol", 3: "Sideways"}
    regime_label  = REGIME_LABELS.get(regime_int, "Unknown")
    deploy_cap    = REGIME_DEPLOY_CAP.get(regime_int, 0.90)
    as_of         = signal_date or date.today()

    # ── 1. Candidate selection ─────────────────────────────────────────────

    df = signals_df.copy()
    # Always use sector_map_df as ground truth — signals CSV sector may be "Unknown"
    df = df.drop(columns=["Sector"], errors="ignore")
    df = df.merge(sector_map_df[["Ticker","Sector"]], on="Ticker", how="left")
    df["Sector"] = df["Sector"].fillna("Unknown")
    df["is_midcap"] = df["Data_Tier"].map(TIER_IS_MIDCAP).fillna(True)

    long_cands  = df[df["Final_Rank"] >= LONG_RANK_THRESHOLD].sort_values(
        "Final_Rank", ascending=False
    ).reset_index(drop=True)

    short_cands = df[df["Final_Rank"] <= SHORT_RANK_THRESHOLD].sort_values(
        "Final_Rank", ascending=True   # worst rank first (most conviction shorts)
    ).reset_index(drop=True)

    all_cands = pd.concat([long_cands, short_cands], ignore_index=True)

    if all_cands.empty:
        empty_stats = PortfolioStats(
            nav=nav, regime=regime_int, regime_label=regime_label,
            deploy_cap=deploy_cap, gross_long_pct=0, gross_short_pct=0,
            net_exposure_pct=0, gross_exposure_pct=0,
            cash_pct=1.0, n_longs=0, n_shorts=0, n_total=0, n_rejected=0,
            long_sectors=0, below_min_warning=True,
        )
        empty_ps = PortfolioState(
            as_of_date=as_of, nav=nav, peak_nav=nav, cash=nav, positions={},
            regime=regime_int,
        )
        return OptimizedPortfolio(
            positions=pd.DataFrame(), rejected=pd.DataFrame(),
            stats=empty_stats, portfolio_state=empty_ps,
        )

    # ── 2. Load ATR data ───────────────────────────────────────────────────

    all_tickers = all_cands["Ticker"].tolist()
    atr_df      = _load_atr(all_tickers, engine)

    long_cands  = long_cands.merge(atr_df[["Ticker","ATR_pct","Adj_Close"]], on="Ticker", how="left")
    short_cands = short_cands.merge(atr_df[["Ticker","ATR_pct","Adj_Close"]], on="Ticker", how="left")
    long_cands["ATR_pct"]  = long_cands["ATR_pct"].fillna(ATR_PCT_FALLBACK)
    short_cands["ATR_pct"] = short_cands["ATR_pct"].fillna(ATR_PCT_FALLBACK)

    # ── 3. Compute proposed sizes ──────────────────────────────────────────

    # Allocate MAX_POSITIONS slots proportionally to candidate counts
    n_long_cands  = len(long_cands)
    n_short_cands = len(short_cands)
    total_cands   = n_long_cands + n_short_cands

    if total_cands == 0:
        long_slots  = MAX_POSITIONS
        short_slots = 0
    else:
        # Bias heavily toward longs: shorts capped at 20% of MAX_POSITIONS slots
        # This prevents a balanced signal count from wasting slots on shorts
        max_short_slots = max(3, round(MAX_POSITIONS * 0.20))  # ≤ 20% slots to shorts
        short_slots = min(max_short_slots, n_short_cands)
        long_slots  = min(MAX_POSITIONS - short_slots, n_long_cands)
        long_slots  = max(long_slots, min(5, n_long_cands))  # at least 5 longs

    # Take top candidates by rank within each slot budget
    long_pool  = long_cands.head(long_slots).reset_index(drop=True)
    short_pool = short_cands.head(short_slots).reset_index(drop=True)

    long_sizes  = _compute_sizes(long_pool,  "long",  nav, LONG_TARGET_PCT,  deploy_cap)
    short_sizes = _compute_sizes(short_pool, "short", nav, SHORT_TARGET_PCT, deploy_cap)

    # ── 4. Risk-check loop ────────────────────────────────────────────────
    # Process longs first (higher conviction, larger book), then shorts.
    # Each iteration updates the running portfolio state.

    approved_positions: dict[str, dict] = {}
    rejected_records:   list[dict]      = []

    def _current_ps() -> PortfolioState:
        """Build PortfolioState from approved_positions so far."""
        pos_map = {
            t: OpenPosition(
                ticker        = t,
                direction     = d["direction"],
                size_nav_pct  = d["size_nav_pct"],
                entry_price   = 0.0,
                current_price = 0.0,
                entry_date    = as_of,
                sector        = d["sector"],
                is_midcap     = d["is_midcap"],
            )
            for t, d in approved_positions.items()
        }
        cash_used = sum(d["size_nav_pct"] * nav for d in approved_positions.values())
        return PortfolioState(
            as_of_date          = as_of,
            nav                 = nav,
            peak_nav            = nav,
            cash                = max(nav - cash_used, 0.0),
            positions           = pos_map,
            fii_consecutive_neg = 0,
            fii_net_today_cr    = None,
            nav_return_21d      = None,
            is_budget_period    = False,
            is_fo_expiry_week   = False,
            regime              = regime_int,
        )

    def _try_add(row: pd.Series, direction: str, proposed_size: float) -> bool:
        """
        Run risk check, add to approved if passes.
        Returns True if added (APPROVED or REDUCED).
        """
        if len(approved_positions) >= MAX_POSITIONS:
            rejected_records.append(_rej_row(row, direction, proposed_size,
                                             "REJECTED", "CAP", "MAX_POSITIONS reached"))
            return False

        ticker    = row["Ticker"]
        sector    = row["Sector"]
        is_midcap = bool(row.get("is_midcap", False))

        decision = check_position(
            ticker                 = ticker,
            direction              = direction,
            size_nav_pct           = proposed_size,
            sector                 = sector,
            portfolio_state        = _current_ps(),
            is_midcap              = is_midcap,
            market_cap_cr          = None,
            lower_circuit_hits_90d = 0,
        )

        if decision.status == "REJECTED":
            rejected_records.append(_rej_row(
                row, direction, proposed_size,
                decision.status, decision.limit or "—", decision.reason
            ))
            return False

        # APPROVED or REDUCED
        approved_positions[ticker] = {
            "direction":   direction,
            "size_nav_pct": decision.approved_size,
            "sector":      sector,
            "is_midcap":   is_midcap,
        }
        return True

    def _rej_row(row, direction, size, status, limit, reason) -> dict:
        return {
            "Ticker":    row["Ticker"],
            "Direction": "LONG" if direction == "long" else "SHORT",
            "Sector":    row.get("Sector", "Unknown"),
            "Final_Rank":round(float(row["Final_Rank"]), 3),
            "Proposed_Size_%NAV": round(size * 100, 2),
            "Risk_Status": status,
            "Risk_Limit":  limit,
            "Reason":      reason[:120],
        }

    # Process longs
    for _, row in long_pool.iterrows():
        ticker = row["Ticker"]
        size   = float(long_sizes.get(ticker, MIN_POSITION_SIZE_PCT))
        _try_add(row, "long", size)

    # Process shorts
    for _, row in short_pool.iterrows():
        ticker = row["Ticker"]
        size   = float(short_sizes.get(ticker, MIN_POSITION_SIZE_PCT))
        _try_add(row, "short", size)

    # ── 5. Build output DataFrames ────────────────────────────────────────

    if not approved_positions:
        position_rows = []
    else:
        position_rows = []
        # Merge back signal metadata
        all_sig = pd.concat([long_pool, short_pool], ignore_index=True)
        sig_idx  = all_sig.set_index("Ticker")

        for ticker, pos in approved_positions.items():
            sig_row = sig_idx.loc[ticker] if ticker in sig_idx.index else pd.Series()
            adj_close = float(sig_row.get("Adj_Close", float("nan")))
            atr_pct   = float(sig_row.get("ATR_pct",   ATR_PCT_FALLBACK))
            rank      = float(sig_row.get("Final_Rank", 0.5))
            lstm_vol  = float(sig_row.get("LSTM_Vol",  float("nan")))
            tier      = int(sig_row.get("Data_Tier", 1)) if "Data_Tier" in sig_row.index else 1

            position_rows.append({
                "Ticker":       ticker,
                "Direction":    "LONG" if pos["direction"] == "long" else "SHORT",
                "Sector":       pos["sector"],
                "Tier":         {1:"A", 2:"B", 3:"C"}.get(tier, "?"),
                "Final_Rank":   round(rank, 3),
                "ATR_pct":      round(atr_pct, 4),
                "LSTM_Vol":     round(lstm_vol, 3) if not pd.isna(lstm_vol) else None,
                "Last_Price":   round(adj_close, 2) if not pd.isna(adj_close) else None,
                "Size_%NAV":    round(pos["size_nav_pct"] * 100, 2),
                "Alloc_Rs":     round(pos["size_nav_pct"] * nav),
            })

    positions_df = pd.DataFrame(position_rows)
    rejected_df  = pd.DataFrame(rejected_records)

    # ── 6. Compute stats ──────────────────────────────────────────────────

    ps_final = _current_ps()

    if not positions_df.empty:
        longs_df  = positions_df[positions_df["Direction"] == "LONG"]
        shorts_df = positions_df[positions_df["Direction"] == "SHORT"]
        n_longs   = len(longs_df)
        n_shorts  = len(shorts_df)
        gl        = longs_df["Size_%NAV"].sum() / 100
        gs        = shorts_df["Size_%NAV"].sum() / 100
        long_secs = longs_df["Sector"].nunique() if not longs_df.empty else 0
    else:
        n_longs = n_shorts = 0
        gl = gs = 0.0
        long_secs = 0

    net  = gl - gs
    grs  = gl + gs
    csh  = 1 - gl - gs
    n_tot = n_longs + n_shorts

    stats = PortfolioStats(
        nav                = nav,
        regime             = regime_int,
        regime_label       = regime_label,
        deploy_cap         = deploy_cap,
        gross_long_pct     = gl,
        gross_short_pct    = gs,
        net_exposure_pct   = net,
        gross_exposure_pct = grs,
        cash_pct           = csh,
        n_longs            = n_longs,
        n_shorts           = n_shorts,
        n_total            = n_tot,
        n_rejected         = len(rejected_records),
        long_sectors       = long_secs,
        below_min_warning  = n_tot < MIN_POSITIONS,
    )

    return OptimizedPortfolio(
        positions       = positions_df,
        rejected        = rejected_df,
        stats           = stats,
        portfolio_state = ps_final,
    )