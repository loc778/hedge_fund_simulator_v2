# risk/risk_manager.py
"""
AI Hedge Fund Simulator v2 — Risk Manager
==========================================
Pure function module. No DB writes.
Caller is responsible for persisting positions after APPROVED is returned.

Entry point
-----------
    decision = check_position(ticker, direction, size_nav_pct, portfolio_state)

Return type
-----------
    RiskDecision.status  : "APPROVED" | "REJECTED" | "REDUCED"
    RiskDecision.limit   : limit code that triggered (e.g. "P1", "S1", "PT2")
    RiskDecision.reason  : human-readable explanation
    RiskDecision.approved_size : final allowed size_nav_pct (< input if REDUCED)

Check order (first failure returns immediately)
-----------------------------------------------
    1. Position-level limits   (P1–P6)
    2. Sector-level limits     (S1–S3)
    3. Portfolio-level limits  (PT1–PT6)
    4. Seasonal / India rules  (IR1–IR5)

Stubs
-----
    IR3 (earnings blackout)  : always passes — no results calendar table built yet
    IR5 (circuit breaker)    : always passes — no lower-circuit history table built yet
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Config path resolution
# ---------------------------------------------------------------------------
# Assumes risk_manager.py lives at  <project_root>/risk/risk_manager.py
# and fo_list.csv lives at          <project_root>/files/fo_list.csv
_HERE        = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

try:
    from config import FO_LIST_CSV  # type: ignore
    _FO_LIST_PATH = FO_LIST_CSV
except ImportError:
    _FO_LIST_PATH = os.path.join(_PROJECT_ROOT, "files", "fo_list.csv")


# ---------------------------------------------------------------------------
# Hard limit constants  (change values here — never inline)
# ---------------------------------------------------------------------------

# Position-level
MAX_SINGLE_CORE_LONG_PCT   = 0.08    # P1 — raised from 5% → 8% for large-cap conviction
MAX_SINGLE_MIDCAP_LONG_PCT = 0.05    # P2 — raised from 3% → 5% for mid-cap alpha
MAX_SINGLE_SHORT_PCT       = 0.025   # P3 — unchanged (shorts stay small)
LONG_STOP_LOSS_PCT         = -0.15   # P4 — unchanged
SHORT_STOP_LOSS_PCT        = 0.10    # P5 — unchanged
MIN_POSITION_SIZE_PCT      = 0.005   # P6 — unchanged

# Sector-level
MAX_SECTOR_GROSS_LONG_PCT     = 0.30  # S1 — raised from 25%; allows meaningful sector tilts
MAX_SECTOR_GROSS_LONG_FIN_PCT = 0.35  # S1 exception — Financials sector (raised from 30%)
MAX_SECTOR_GROSS_SHORT_PCT    = 0.15  # S2 — unchanged
MIN_SECTOR_COUNT_LONG         = 5     # S3 — unchanged

# Portfolio-level
NET_EXPOSURE_MIN_PCT   = 0.80   # PT1 — raised from 70% — enforce minimum long bias
NET_EXPOSURE_MAX_PCT   = 1.25   # PT1 — raised from 110% to match LONG_TARGET_PCT
GROSS_EXPOSURE_CAP_PCT = 1.60   # PT2 — raised from 150% (longs 135% + shorts 15% = 150%)
CASH_RESERVE_MIN_PCT   = 0.05   # PT3 — reduced from 8% → 5%; was leaving too much idle
CASH_RESERVE_TEMP_PCT  = 0.03   # PT3 — temporary floor reduced proportionally
MIDCAP_LONG_BOOK_MAX   = 0.40   # PT4 — raised from 35%; mid-caps drive alpha
DRAWDOWN_REVIEW_PCT    = -0.05  # PT5 — unchanged
PEAK_TROUGH_STOP_PCT   = -0.10  # PT6 — unchanged

# Seasonal
FII_SINGLE_SESSION_SELL_CR = 5_000   # IR4 — single session net sell threshold (Crores)
FII_CONSECUTIVE_NEG_DAYS   = 5       # IR4 — consecutive negative sessions threshold
FII_SELL_NET_LONG_MAX_PCT  = 0.85    # IR4 — reduce net long to 85% during FII sell-off
CIRCUIT_LOOKBACK_DAYS      = 90      # IR5 — lookback window for lower circuits
CIRCUIT_MIN_OCCURRENCES    = 2       # IR5 — min lower-circuit hits to block short
MIN_MKTCAP_FOR_SHORT_CR    = 3_000   # IR5 — min market cap to short (Crores)
BUDGET_PERIOD_REDUCTION    = 0.25    # IR2 — reduce all positions 25% during budget window
FNO_EXPIRY_SHORT_REDUCTION = 0.25    # IR1 — reduce short book 20-30% before expiry

# Financial sector aliases (any sector string matching these is treated as Financials)
FINANCIAL_SECTOR_NAMES = {
    "Financial Services", "Financials", "Banking", "BFSI",
    "Banks", "Insurance", "NBFC"
}


# ---------------------------------------------------------------------------
# F&O eligible set  (loaded once at module import)
# ---------------------------------------------------------------------------

def _load_fo_set(path: str) -> frozenset:
    """
    Load F&O eligible tickers from CSV.
    Expected format: one column named 'Symbol' (with or without .NS suffix).
    Falls back to empty set with a warning if file is missing.
    """
    if not os.path.exists(path):
        import warnings
        warnings.warn(
            f"[risk_manager] fo_list.csv not found at {path}. "
            "F&O check will REJECT all short attempts.",
            RuntimeWarning
        )
        return frozenset()

    tickers = set()
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        # Strip trailing spaces from field names (NSE CSVs often have them)
        raw_fields = reader.fieldnames or []
        field_map  = {f.strip(): f for f in raw_fields}  # stripped -> original
        col = None
        for candidate in ("SYMBOL", "Symbol", "symbol", "Ticker", "ticker", "UNDERLYING"):
            if candidate in field_map:
                col = field_map[candidate]  # use original key for row lookup
                break
        if col is None:
            raise ValueError(
                f"[risk_manager] fo_list.csv must have a 'SYMBOL' or 'Symbol' column. "
                f"Found: {raw_fields}"
            )
        for row in reader:
            raw = row[col].strip()
            # Normalise: store both with and without .NS suffix
            tickers.add(raw)
            tickers.add(raw.replace(".NS", "").replace(".ns", ""))

    return frozenset(tickers)


_FO_SET: frozenset = _load_fo_set(_FO_LIST_PATH)


def is_fo_eligible(ticker: str) -> bool:
    """Return True if ticker is F&O eligible."""
    return (ticker in _FO_SET) or (ticker.replace(".NS", "") in _FO_SET)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    """
    Represents a single open position in the portfolio.

    Fields
    ------
    ticker          : NSE ticker string (e.g. "RELIANCE.NS")
    direction       : "long" or "short"
    size_nav_pct    : current allocation as fraction of NAV (e.g. 0.04 = 4%)
    entry_price     : price at entry (used for stop-loss calculation)
    current_price   : latest mark-to-market price
    entry_date      : date position was opened
    sector          : sector string (must match SECTOR_MAP values in config.py)
    is_midcap       : True if stock is ranked 101–500 by market cap (Nifty 101-500)
    lower_circuit_hits_90d : number of lower-circuit events in last 90 trading days
                             (set to 0 if data unavailable — IR5 stub)
    market_cap_cr   : market capitalisation in Crores (set to None if unknown)
    """
    ticker                  : str
    direction               : str           # "long" | "short"
    size_nav_pct            : float
    entry_price             : float
    current_price           : float
    entry_date              : date
    sector                  : str
    is_midcap               : bool = False
    lower_circuit_hits_90d  : int  = 0      # IR5 — stub; set from DB when available
    market_cap_cr           : Optional[float] = None  # IR5 — stub


@dataclass
class PortfolioState:
    """
    Complete snapshot of portfolio state at a single point in time.
    Pass this to check_position() on every pre-trade call.

    Fields
    ------
    as_of_date          : date of this snapshot
    nav                 : current total NAV in Rupees
    peak_nav            : highest NAV recorded (for PT6 drawdown check)
    cash                : uninvested cash in Rupees
    positions           : dict of ticker → OpenPosition for all open positions
    fii_net_today_cr    : FII net equity flow today in Crores (negative = selling)
                          Use None if data not yet available for today
    fii_consecutive_neg : number of consecutive sessions of negative FII net flow
    nav_return_21d      : portfolio return over last 21 trading days (fraction)
                          e.g. -0.04 means -4% over 21 days. None if < 21 days history.
    is_budget_period    : True between Jan 15 and Union Budget day (typically Feb 1)
    is_fo_expiry_week   : True during the F&O expiry week (last week of month,
                          specifically Mon–Wed before Thursday expiry)
    regime              : current HMM regime int (0=Bull,1=Bear,2=HighVol,3=Sideways)
                          Used for informational context; not a hard limit gate.

    Helper properties (computed, do not set manually)
    --------------------------------------------------
    gross_long_nav_pct  : sum of all long position sizes as % of NAV
    gross_short_nav_pct : sum of all short position sizes as % of NAV
    net_exposure_pct    : (gross_long - gross_short) / NAV
    gross_exposure_pct  : (gross_long + gross_short) / NAV
    cash_pct            : cash / NAV
    midcap_long_pct     : mid-cap allocation as % of total long book
    sector_long_pcts    : dict of sector → gross long % of NAV
    sector_short_pcts   : dict of sector → gross short % of NAV
    long_sector_count   : number of distinct sectors in long book

    Class method
    ------------
    PortfolioState.from_db(engine, nav, peak_nav, fii_today, fii_consec,
                            nav_21d, is_budget, is_expiry_week, regime)
        Reads portfolio_positions table and constructs the object.
        Stub implementation included — fill in SQL query when positions go live.
    """
    as_of_date          : date
    nav                 : float
    peak_nav            : float
    cash                : float
    positions           : Dict[str, OpenPosition] = field(default_factory=dict)
    fii_net_today_cr    : Optional[float] = None
    fii_consecutive_neg : int  = 0
    nav_return_21d      : Optional[float] = None
    is_budget_period    : bool = False
    is_fo_expiry_week   : bool = False
    regime              : int  = 3  # default Sideways

    # ── computed properties ──────────────────────────────────────────────────

    @property
    def gross_long_nav_pct(self) -> float:
        return sum(
            p.size_nav_pct for p in self.positions.values() if p.direction == "long"
        )

    @property
    def gross_short_nav_pct(self) -> float:
        return sum(
            p.size_nav_pct for p in self.positions.values() if p.direction == "short"
        )

    @property
    def net_exposure_pct(self) -> float:
        return self.gross_long_nav_pct - self.gross_short_nav_pct

    @property
    def gross_exposure_pct(self) -> float:
        return self.gross_long_nav_pct + self.gross_short_nav_pct

    @property
    def cash_pct(self) -> float:
        return self.cash / self.nav if self.nav > 0 else 0.0

    @property
    def midcap_long_pct(self) -> float:
        """Mid-cap allocation as fraction of total long book (not NAV)."""
        total_long = self.gross_long_nav_pct
        if total_long == 0:
            return 0.0
        midcap = sum(
            p.size_nav_pct
            for p in self.positions.values()
            if p.direction == "long" and p.is_midcap
        )
        return midcap / total_long

    @property
    def sector_long_pcts(self) -> Dict[str, float]:
        """Gross long allocation per sector as fraction of NAV."""
        d: Dict[str, float] = {}
        for p in self.positions.values():
            if p.direction == "long":
                d[p.sector] = d.get(p.sector, 0.0) + p.size_nav_pct
        return d

    @property
    def sector_short_pcts(self) -> Dict[str, float]:
        """Gross short allocation per sector as fraction of NAV."""
        d: Dict[str, float] = {}
        for p in self.positions.values():
            if p.direction == "short":
                d[p.sector] = d.get(p.sector, 0.0) + p.size_nav_pct
        return d

    @property
    def long_sector_count(self) -> int:
        return len(self.sector_long_pcts)

    # ── from_db stub ─────────────────────────────────────────────────────────

    @classmethod
    def from_db(
        cls,
        engine,
        nav: float,
        peak_nav: float,
        cash: float,
        fii_net_today_cr: Optional[float],
        fii_consecutive_neg: int,
        nav_return_21d: Optional[float],
        is_budget_period: bool,
        is_fo_expiry_week: bool,
        regime: int,
        as_of_date: Optional[date] = None,
    ) -> "PortfolioState":
        """
        Build PortfolioState by reading open positions from portfolio_positions table.

        TODO: implement SQL query when portfolio_positions is being written.
        Currently returns an empty positions dict (safe — all size checks pass
        against zero-filled book).

        Expected query:
            SELECT Ticker, Entry_Date, Entry_Price, NAV_Weight_At_Entry,
                   Position_Class, Sector
            FROM portfolio_positions
            WHERE Exit_Date IS NULL
        Then for each row, construct OpenPosition with:
            - current_price from latest nifty500_ohlcv Adj_Close
            - is_midcap from nifty500_sectors or config SECTOR_MAP
            - lower_circuit_hits_90d from nifty500_ohlcv circuit analysis (TODO)
            - market_cap_cr from nifty500_fundamentals or external source (TODO)
        """
        positions: Dict[str, OpenPosition] = {}
        # ── STUB: no active positions yet ────────────────────────────────────
        # Uncomment and implement when portfolio_positions is live:
        #
        # import pandas as pd
        # with engine.connect() as conn:
        #     rows = pd.read_sql(
        #         "SELECT * FROM portfolio_positions WHERE Exit_Date IS NULL",
        #         conn
        #     )
        # for _, row in rows.iterrows():
        #     positions[row['Ticker']] = OpenPosition(
        #         ticker        = row['Ticker'],
        #         direction     = 'long' if 'long' in row['Position_Class'] else 'short',
        #         size_nav_pct  = row['NAV_Weight_At_Entry'],
        #         entry_price   = row['Entry_Price'],
        #         current_price = _fetch_latest_price(engine, row['Ticker']),
        #         entry_date    = row['Entry_Date'],
        #         sector        = row.get('Sector', 'Unknown'),
        #         is_midcap     = row.get('Is_Midcap', False),
        #     )

        return cls(
            as_of_date          = as_of_date or date.today(),
            nav                 = nav,
            peak_nav            = peak_nav,
            cash                = cash,
            positions           = positions,
            fii_net_today_cr    = fii_net_today_cr,
            fii_consecutive_neg = fii_consecutive_neg,
            nav_return_21d      = nav_return_21d,
            is_budget_period    = is_budget_period,
            is_fo_expiry_week   = is_fo_expiry_week,
            regime              = regime,
        )


@dataclass
class RiskDecision:
    """
    Return type from check_position().

    status        : "APPROVED" | "REJECTED" | "REDUCED"
    limit         : limit code that triggered the decision (e.g. "P1", "S2", "IR4")
                    None if APPROVED without any reduction
    reason        : human-readable explanation
    approved_size : final allowed size_nav_pct
                    = original size if APPROVED
                    = reduced size if REDUCED
                    = 0.0 if REJECTED
    """
    status        : str
    limit         : Optional[str]
    reason        : str
    approved_size : float


# ---------------------------------------------------------------------------
# Internal limit checks
# ---------------------------------------------------------------------------

def _check_position_limits(
    ticker: str,
    direction: str,
    size: float,
    sector: str,
    is_midcap: bool,
    ps: PortfolioState,
) -> Optional[RiskDecision]:
    """
    P1–P6 position-level checks.
    Returns a RiskDecision if a limit is triggered, else None (pass).
    """

    # P6 — minimum size
    if size < MIN_POSITION_SIZE_PCT:
        return RiskDecision(
            status="REJECTED",
            limit="P6",
            reason=f"Proposed size {size:.2%} is below minimum {MIN_POSITION_SIZE_PCT:.2%}.",
            approved_size=0.0,
        )

    if direction == "long":

        # P1 / P2 — max single position
        if is_midcap:
            cap = MAX_SINGLE_MIDCAP_LONG_PCT  # P2
            limit_code = "P2"
        else:
            cap = MAX_SINGLE_CORE_LONG_PCT    # P1
            limit_code = "P1"

        if size > cap:
            # REDUCED — trim to cap rather than full reject
            return RiskDecision(
                status="REDUCED",
                limit=limit_code,
                reason=(
                    f"Proposed long size {size:.2%} exceeds "
                    f"{'mid-cap' if is_midcap else 'core'} cap {cap:.2%}. "
                    f"Reduced to {cap:.2%}."
                ),
                approved_size=cap,
            )

        # P4 — check if existing position already at stop-loss (informational guard)
        if ticker in ps.positions:
            pos = ps.positions[ticker]
            if pos.entry_price > 0:
                pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price
                if pnl_pct <= LONG_STOP_LOSS_PCT:
                    return RiskDecision(
                        status="REJECTED",
                        limit="P4",
                        reason=(
                            f"{ticker} is at or below long stop-loss "
                            f"({pnl_pct:.2%} from entry). Cannot add to position."
                        ),
                        approved_size=0.0,
                    )

    elif direction == "short":

        # P3 — max single short
        if size > MAX_SINGLE_SHORT_PCT:
            return RiskDecision(
                status="REDUCED",
                limit="P3",
                reason=(
                    f"Proposed short size {size:.2%} exceeds cap {MAX_SINGLE_SHORT_PCT:.2%}. "
                    f"Reduced to {MAX_SINGLE_SHORT_PCT:.2%}."
                ),
                approved_size=MAX_SINGLE_SHORT_PCT,
            )

        # P5 — check if existing short position is at stop-loss
        if ticker in ps.positions:
            pos = ps.positions[ticker]
            if pos.entry_price > 0:
                adverse_move = (pos.current_price - pos.entry_price) / pos.entry_price
                if adverse_move >= SHORT_STOP_LOSS_PCT:
                    return RiskDecision(
                        status="REJECTED",
                        limit="P5",
                        reason=(
                            f"{ticker} short position has moved +{adverse_move:.2%} against "
                            f"entry (threshold {SHORT_STOP_LOSS_PCT:.2%}). Cannot add."
                        ),
                        approved_size=0.0,
                    )

    return None  # all position checks passed


def _check_sector_limits(
    direction: str,
    size: float,
    sector: str,
    ps: PortfolioState,
) -> Optional[RiskDecision]:
    """
    S1–S3 sector-level checks.
    """

    if direction == "long":
        current_sector_long = ps.sector_long_pcts.get(sector, 0.0)
        projected = current_sector_long + size

        # S1 — sector long cap (Financials has higher limit)
        is_financial = sector in FINANCIAL_SECTOR_NAMES
        cap = MAX_SECTOR_GROSS_LONG_FIN_PCT if is_financial else MAX_SECTOR_GROSS_LONG_PCT

        if projected > cap:
            # Can we fit a reduced position?
            headroom = cap - current_sector_long
            if headroom < MIN_POSITION_SIZE_PCT:
                return RiskDecision(
                    status="REJECTED",
                    limit="S1",
                    reason=(
                        f"Sector '{sector}' long at {current_sector_long:.2%}; "
                        f"cap {cap:.2%}. No headroom for minimum position."
                    ),
                    approved_size=0.0,
                )
            return RiskDecision(
                status="REDUCED",
                limit="S1",
                reason=(
                    f"Sector '{sector}' long would reach {projected:.2%}; "
                    f"cap {cap:.2%}. Reduced to headroom {headroom:.2%}."
                ),
                approved_size=headroom,
            )

        # S3 — if this is a new sector for the long book, check minimum diversity
        # (S3 is a minimum — adding a new sector only improves diversity, never rejects)
        # S3 blocks positions that would *reduce* sector count (i.e. exiting last
        # position in a sector when already at minimum). Not applicable here (entry).

    elif direction == "short":
        current_sector_short = ps.sector_short_pcts.get(sector, 0.0)
        projected = current_sector_short + size

        # S2 — sector short cap
        if projected > MAX_SECTOR_GROSS_SHORT_PCT:
            headroom = MAX_SECTOR_GROSS_SHORT_PCT - current_sector_short
            if headroom < MIN_POSITION_SIZE_PCT:
                return RiskDecision(
                    status="REJECTED",
                    limit="S2",
                    reason=(
                        f"Sector '{sector}' short at {current_sector_short:.2%}; "
                        f"cap {MAX_SECTOR_GROSS_SHORT_PCT:.2%}. No headroom."
                    ),
                    approved_size=0.0,
                )
            return RiskDecision(
                status="REDUCED",
                limit="S2",
                reason=(
                    f"Sector '{sector}' short would reach {projected:.2%}; "
                    f"cap {MAX_SECTOR_GROSS_SHORT_PCT:.2%}. Reduced to {headroom:.2%}."
                ),
                approved_size=headroom,
            )

    return None


def _check_portfolio_limits(
    direction: str,
    size: float,
    ps: PortfolioState,
) -> Optional[RiskDecision]:
    """
    PT1–PT6 portfolio-level checks.
    """

    # PT6 — peak-to-trough hard stop (highest priority portfolio check)
    if ps.peak_nav > 0:
        drawdown = (ps.nav - ps.peak_nav) / ps.peak_nav
        if drawdown <= PEAK_TROUGH_STOP_PCT:
            return RiskDecision(
                status="REJECTED",
                limit="PT6",
                reason=(
                    f"Portfolio is {drawdown:.2%} from NAV peak "
                    f"(hard stop {PEAK_TROUGH_STOP_PCT:.2%}). "
                    "Reduce book by 30% before opening new positions."
                ),
                approved_size=0.0,
            )

    # PT5 — 21-day rolling drawdown review
    if ps.nav_return_21d is not None and ps.nav_return_21d <= DRAWDOWN_REVIEW_PCT:
        return RiskDecision(
            status="REJECTED",
            limit="PT5",
            reason=(
                f"21-day portfolio return {ps.nav_return_21d:.2%} triggers "
                f"drawdown review threshold ({DRAWDOWN_REVIEW_PCT:.2%}). "
                "No new positions during review."
            ),
            approved_size=0.0,
        )

    # PT3 — cash reserve
    # Project cash after this trade
    projected_cash_pct = ps.cash_pct - size
    if projected_cash_pct < CASH_RESERVE_MIN_PCT:
        headroom = ps.cash_pct - CASH_RESERVE_MIN_PCT
        if headroom < MIN_POSITION_SIZE_PCT:
            return RiskDecision(
                status="REJECTED",
                limit="PT3",
                reason=(
                    f"Opening {size:.2%} position would push cash to "
                    f"{projected_cash_pct:.2%}, below minimum {CASH_RESERVE_MIN_PCT:.2%}."
                ),
                approved_size=0.0,
            )
        return RiskDecision(
            status="REDUCED",
            limit="PT3",
            reason=(
                f"Cash constrained. Maximum deployable without breaching "
                f"{CASH_RESERVE_MIN_PCT:.2%} reserve: {headroom:.2%}."
            ),
            approved_size=headroom,
        )

    # PT2 — gross exposure cap
    projected_gross = ps.gross_exposure_pct + size
    if projected_gross > GROSS_EXPOSURE_CAP_PCT:
        headroom = GROSS_EXPOSURE_CAP_PCT - ps.gross_exposure_pct
        if headroom < MIN_POSITION_SIZE_PCT:
            return RiskDecision(
                status="REJECTED",
                limit="PT2",
                reason=(
                    f"Gross exposure {ps.gross_exposure_pct:.2%}; "
                    f"cap {GROSS_EXPOSURE_CAP_PCT:.2%}. No headroom."
                ),
                approved_size=0.0,
            )
        return RiskDecision(
            status="REDUCED",
            limit="PT2",
            reason=(
                f"Gross exposure would reach {projected_gross:.2%}; "
                f"cap {GROSS_EXPOSURE_CAP_PCT:.2%}. Reduced to {headroom:.2%}."
            ),
            approved_size=headroom,
        )

    # PT1 — net exposure range
    if direction == "long":
        projected_net = ps.net_exposure_pct + size
    else:
        projected_net = ps.net_exposure_pct - size

    if projected_net > NET_EXPOSURE_MAX_PCT:
        return RiskDecision(
            status="REJECTED",
            limit="PT1",
            reason=(
                f"Net exposure would reach {projected_net:.2%}; "
                f"max {NET_EXPOSURE_MAX_PCT:.2%}."
            ),
            approved_size=0.0,
        )
    # Lower bound only enforced when portfolio is already deployed (not during book construction)
    # and only for shorts — a long always increases net exposure toward the valid range
    if projected_net < NET_EXPOSURE_MIN_PCT and ps.gross_long_nav_pct > 0 and direction == "short":
        return RiskDecision(
            status="REJECTED",
            limit="PT1",
            reason=(
                f"Net exposure would fall to {projected_net:.2%}; "
                f"min {NET_EXPOSURE_MIN_PCT:.2%}."
            ),
            approved_size=0.0,
        )

    # PT4 — mid-cap concentration (long only)
    if direction == "long":
        # Compute what mid-cap % of long book would be after this trade
        # (approximation: assumes new position is at proposed size)
        total_long_after = ps.gross_long_nav_pct + size
        midcap_long_nav = ps.midcap_long_pct * ps.gross_long_nav_pct
        # Caller sets is_midcap via OpenPosition; we can't know it here.
        # PT4 check is therefore done post-factor in check_position().
        pass

    return None


def _check_seasonal_rules(
    ticker: str,
    direction: str,
    size: float,
    is_midcap: bool,
    market_cap_cr: Optional[float],
    lower_circuit_hits_90d: int,
    ps: PortfolioState,
) -> Optional[RiskDecision]:
    """
    IR1–IR5 India-specific seasonal rules.
    """

    # IR1 — F&O expiry week: block NEW short positions (existing shorts managed by caller)
    if ps.is_fo_expiry_week and direction == "short":
        return RiskDecision(
            status="REJECTED",
            limit="IR1",
            reason=(
                "F&O expiry week. No new short positions. "
                "Reduce existing short book 20–30% before Thursday."
            ),
            approved_size=0.0,
        )

    # IR2 — Budget period: reduce all new positions by 25%
    if ps.is_budget_period:
        reduced = size * (1 - BUDGET_PERIOD_REDUCTION)
        if reduced < MIN_POSITION_SIZE_PCT:
            return RiskDecision(
                status="REJECTED",
                limit="IR2",
                reason=(
                    f"Budget period. 25% size reduction leaves {reduced:.2%}, "
                    f"below minimum {MIN_POSITION_SIZE_PCT:.2%}."
                ),
                approved_size=0.0,
            )
        return RiskDecision(
            status="REDUCED",
            limit="IR2",
            reason=(
                f"Budget period. All new positions reduced 25%. "
                f"Size adjusted from {size:.2%} to {reduced:.2%}."
            ),
            approved_size=reduced,
        )

    # IR3 — Earnings blackout: STUB — no results calendar table yet
    # TODO: Query earnings_calendar table for ticker's next result date.
    #       If (next_result_date - as_of_date) <= 21 calendar days AND direction == "short":
    #           return REJECTED with limit="IR3"
    #       For existing shorts: reduce to 50% of current size in the 21-day window.
    #       Implement once earnings_calendar table is built.

    # IR4 — FII sell-off protocol (long positions only affected)
    if direction == "long":
        fii_trigger = False
        if (
            ps.fii_net_today_cr is not None
            and ps.fii_net_today_cr < -FII_SINGLE_SESSION_SELL_CR
        ):
            fii_trigger = True
        if ps.fii_consecutive_neg >= FII_CONSECUTIVE_NEG_DAYS:
            fii_trigger = True

        if fii_trigger:
            # Net long must not exceed 85% — check if adding this position would breach
            projected_net = ps.net_exposure_pct + size
            if projected_net > FII_SELL_NET_LONG_MAX_PCT:
                headroom = max(FII_SELL_NET_LONG_MAX_PCT - ps.net_exposure_pct, 0.0)
                if headroom < MIN_POSITION_SIZE_PCT:
                    return RiskDecision(
                        status="REJECTED",
                        limit="IR4",
                        reason=(
                            "FII sell-off protocol active. Net long already at/above "
                            f"{FII_SELL_NET_LONG_MAX_PCT:.0%} limit. No new longs."
                        ),
                        approved_size=0.0,
                    )
                return RiskDecision(
                    status="REDUCED",
                    limit="IR4",
                    reason=(
                        f"FII sell-off protocol active. Net long capped at "
                        f"{FII_SELL_NET_LONG_MAX_PCT:.0%}. Size reduced to {headroom:.2%}."
                    ),
                    approved_size=headroom,
                )

    # IR5 — Circuit breaker trap prevention (short only)
    # STUB: lower_circuit_hits_90d sourced from OpenPosition (caller sets it).
    # Full implementation requires querying nifty500_ohlcv for circuit-limit days.
    if direction == "short":
        if lower_circuit_hits_90d >= CIRCUIT_MIN_OCCURRENCES:
            return RiskDecision(
                status="REJECTED",
                limit="IR5",
                reason=(
                    f"{ticker} hit lower circuit {lower_circuit_hits_90d} times "
                    f"in last {CIRCUIT_LOOKBACK_DAYS} days. Short blocked (trap risk)."
                ),
                approved_size=0.0,
            )
        if market_cap_cr is not None and market_cap_cr < MIN_MKTCAP_FOR_SHORT_CR:
            return RiskDecision(
                status="REJECTED",
                limit="IR5",
                reason=(
                    f"{ticker} market cap ₹{market_cap_cr:.0f} Cr is below "
                    f"minimum ₹{MIN_MKTCAP_FOR_SHORT_CR:,} Cr for short positions."
                ),
                approved_size=0.0,
            )

    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def check_position(
    ticker: str,
    direction: str,
    size_nav_pct: float,
    sector: str,
    portfolio_state: PortfolioState,
    is_midcap: bool = False,
    market_cap_cr: Optional[float] = None,
    lower_circuit_hits_90d: int = 0,
) -> RiskDecision:
    """
    Pre-trade risk check. Call before every new position or position increase.

    Parameters
    ----------
    ticker               : NSE ticker string, e.g. "RELIANCE.NS"
    direction            : "long" or "short"
    size_nav_pct         : proposed allocation as fraction of NAV, e.g. 0.03 = 3%
    sector               : sector string matching config.py SECTOR_MAP values
    portfolio_state      : PortfolioState snapshot as of today
    is_midcap            : True if Nifty 101–500 stock
    market_cap_cr        : market cap in Crores (used for IR5 short check)
    lower_circuit_hits_90d : lower-circuit events in last 90 days (IR5)

    Returns
    -------
    RiskDecision with status APPROVED / REDUCED / REJECTED.
    On REDUCED, approved_size is the maximum allowable size.
    On REJECTED, approved_size is 0.0.

    Check order (first failure returns immediately)
    -----------------------------------------------
    P6 → P1/P2/P3 → P4/P5 → F&O eligibility → S1/S2 → PT6 → PT5 → PT3 → PT2 → PT1 → PT4 → IR1 → IR2 → IR3 → IR4 → IR5
    """

    # ── Validate inputs ──────────────────────────────────────────────────────
    if direction not in ("long", "short"):
        return RiskDecision(
            status="REJECTED",
            limit=None,
            reason=f"Invalid direction '{direction}'. Must be 'long' or 'short'.",
            approved_size=0.0,
        )

    if size_nav_pct <= 0:
        return RiskDecision(
            status="REJECTED",
            limit=None,
            reason="size_nav_pct must be > 0.",
            approved_size=0.0,
        )

    # Track effective size as it may be reduced through checks
    effective_size = size_nav_pct
    first_reduction_limit: Optional[str] = None  # first limit that triggered a reduction

    # ── F&O eligibility (short only) ─────────────────────────────────────────
    if direction == "short" and not is_fo_eligible(ticker):
        return RiskDecision(
            status="REJECTED",
            limit="FO",
            reason=(
                f"{ticker} is not F&O eligible. "
                "Short positions restricted to F&O-listed stocks only."
            ),
            approved_size=0.0,
        )

    # ── Position limits (P1–P6) ───────────────────────────────────────────────
    decision = _check_position_limits(
        ticker, direction, effective_size, sector, is_midcap, portfolio_state
    )
    if decision is not None:
        if decision.status == "REJECTED":
            return decision
        effective_size = decision.approved_size  # REDUCED — continue with smaller size
        if first_reduction_limit is None:
            first_reduction_limit = decision.limit

    # ── Sector limits (S1–S3) ─────────────────────────────────────────────────
    decision = _check_sector_limits(direction, effective_size, sector, portfolio_state)
    if decision is not None:
        if decision.status == "REJECTED":
            return decision
        effective_size = decision.approved_size
        if first_reduction_limit is None:
            first_reduction_limit = decision.limit

    # ── Portfolio limits (PT1–PT6) ────────────────────────────────────────────
    decision = _check_portfolio_limits(direction, effective_size, portfolio_state)
    if decision is not None:
        if decision.status == "REJECTED":
            return decision
        effective_size = decision.approved_size
        if first_reduction_limit is None:
            first_reduction_limit = decision.limit

    # ── PT4 — mid-cap concentration (needs effective_size post-reduction) ─────
    if direction == "long" and is_midcap:
        total_long_after = portfolio_state.gross_long_nav_pct + effective_size
        if total_long_after > 0:
            midcap_nav_current = (
                portfolio_state.midcap_long_pct * portfolio_state.gross_long_nav_pct
            )
            midcap_pct_after = (midcap_nav_current + effective_size) / total_long_after
            if midcap_pct_after > MIDCAP_LONG_BOOK_MAX:
                # How much mid-cap can we add without breaching the cap?
                # midcap_pct_after = (midcap_nav_current + x) / (long_book + x) = MIDCAP_MAX
                # Solving: x = (MIDCAP_MAX * long_book - midcap_nav_current) / (1 - MIDCAP_MAX)
                long_book = portfolio_state.gross_long_nav_pct
                headroom = (
                    MIDCAP_LONG_BOOK_MAX * long_book - midcap_nav_current
                ) / (1 - MIDCAP_LONG_BOOK_MAX)
                headroom = max(headroom, 0.0)
                if headroom < MIN_POSITION_SIZE_PCT:
                    return RiskDecision(
                        status="REJECTED",
                        limit="PT4",
                        reason=(
                            f"Mid-cap concentration would reach {midcap_pct_after:.2%} "
                            f"of long book; cap {MIDCAP_LONG_BOOK_MAX:.0%}. No headroom."
                        ),
                        approved_size=0.0,
                    )
                return RiskDecision(
                    status="REDUCED",
                    limit="PT4",
                    reason=(
                        f"Mid-cap concentration cap {MIDCAP_LONG_BOOK_MAX:.0%}. "
                        f"Size reduced from {effective_size:.2%} to {headroom:.2%}."
                    ),
                    approved_size=headroom,
                )

    # ── Seasonal rules (IR1–IR5) ──────────────────────────────────────────────
    decision = _check_seasonal_rules(
        ticker, direction, effective_size, is_midcap,
        market_cap_cr, lower_circuit_hits_90d, portfolio_state
    )
    if decision is not None:
        if decision.status == "REJECTED":
            return decision
        effective_size = decision.approved_size
        if first_reduction_limit is None:
            first_reduction_limit = decision.limit

    # ── All checks passed ─────────────────────────────────────────────────────
    if effective_size < size_nav_pct:
        return RiskDecision(
            status="REDUCED",
            limit=first_reduction_limit,
            reason=f"Size reduced from {size_nav_pct:.2%} to {effective_size:.2%} by risk limits.",
            approved_size=effective_size,
        )

    return RiskDecision(
        status="APPROVED",
        limit=None,
        reason="All risk checks passed.",
        approved_size=effective_size,
    )


# ---------------------------------------------------------------------------
# Utility: bulk signal screening
# ---------------------------------------------------------------------------

def screen_signals(
    signals: list[dict],
    portfolio_state: PortfolioState,
) -> list[dict]:
    """
    Run check_position() over a list of candidate signals and return only
    APPROVED or REDUCED ones, with their approved sizes attached.

    Each signal dict must contain:
        ticker, direction, size_nav_pct, sector
    Optional:
        is_midcap, market_cap_cr, lower_circuit_hits_90d

    Returns list of approved signals with 'approved_size' and 'risk_decision' added.
    Signals are returned in the same order; rejected signals are excluded.
    """
    approved = []
    for sig in signals:
        decision = check_position(
            ticker                 = sig["ticker"],
            direction              = sig["direction"],
            size_nav_pct           = sig["size_nav_pct"],
            sector                 = sig["sector"],
            portfolio_state        = portfolio_state,
            is_midcap              = sig.get("is_midcap", False),
            market_cap_cr          = sig.get("market_cap_cr"),
            lower_circuit_hits_90d = sig.get("lower_circuit_hits_90d", 0),
        )
        if decision.status in ("APPROVED", "REDUCED"):
            out = dict(sig)
            out["approved_size"]  = decision.approved_size
            out["risk_decision"]  = decision
            approved.append(out)
    return approved