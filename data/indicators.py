import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import ta

from config import TABLES, TIER_X_EXCLUDED
from data.db import get_engine, save_to_db

engine = get_engine()


# ═══════════════════════════════════════════════════════════
# SECTION 1 — RESUME SUPPORT
# ═══════════════════════════════════════════════════════════

def get_already_computed() -> set:
    """Returns set of tickers already present in nifty500_indicators."""
    try:
        result = pd.read_sql(
            f"SELECT DISTINCT Ticker FROM {TABLES['indicators']}",
            con=engine
        )
        return set(result["Ticker"].tolist())
    except Exception:
        return set()


# ═══════════════════════════════════════════════════════════
# SECTION 2 — INDICATOR CALCULATION
# ═══════════════════════════════════════════════════════════

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)

    price = df["Adj_Close"]   # C1 fix — never use raw Close for price indicators

    # ── Trend: SMA ────────────────────────────────────────────────────
    df["SMA_20"]  = price.rolling(window=20,  min_periods=20).mean()
    df["SMA_50"]  = price.rolling(window=50,  min_periods=50).mean()
    df["SMA_200"] = price.rolling(window=200, min_periods=200).mean()

    # ── Trend: EMA ────────────────────────────────────────────────────
    df["EMA_9"]  = price.ewm(span=9,  adjust=False).mean()
    df["EMA_21"] = price.ewm(span=21, adjust=False).mean()

    # ── Momentum: MACD ────────────────────────────────────────────────
    ema_fast         = price.ewm(span=12, adjust=False).mean()
    ema_slow         = price.ewm(span=26, adjust=False).mean()
    df["MACD"]       = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]  = df["MACD"] - df["MACD_Signal"]

    # ── Momentum: RSI ─────────────────────────────────────────────────
    df["RSI_14"] = ta.momentum.rsi(price, window=14)

    # ── Volatility: Bollinger Bands ───────────────────────────────────
    bb = ta.volatility.BollingerBands(price, window=20, window_dev=2)
    df["BB_Upper"]  = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"]  = bb.bollinger_lband()

    # ── Volatility: ATR ───────────────────────────────────────────────
    # Raw OHLC used — ATR measures price range, not price level.
    # Using Adj_Close here would be wrong (range compression on split dates).
    df["ATR_14"] = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=14
    )

    # ── Momentum: Stochastic ──────────────────────────────────────────
    # Raw OHLC used — %K/%D are range-relative measures.
    stoch        = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"], window=14, smooth_window=3
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # ── Trend: ADX ────────────────────────────────────────────────────
    # Raw OHLC used — directional movement is range-based.
    df["ADX_14"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)

    # ── Volume: OBV ───────────────────────────────────────────────────
    # OBV accumulates across splits creating discontinuities but this
    # is a medium-severity issue, not a blocker for initial training.
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    # ── Price vs VWAP ─────────────────────────────────────────────────
    df["VWAP_Dev"] = (
        (price - df["VWAP_Daily"]) / df["VWAP_Daily"].replace(0, np.nan) * 100
    )

    return df


def process_ticker(df: pd.DataFrame, ticker: str):
    if len(df) < 30:
        return None, f"only {len(df)} rows — skipping"

    df = df.copy().reset_index(drop=True)
    df = calculate_indicators(df)

    cols = [
        "Date", "Ticker",
        "SMA_20", "SMA_50", "SMA_200",
        "EMA_9", "EMA_21",
        "MACD", "MACD_Signal", "MACD_Hist",
        "RSI_14",
        "BB_Upper", "BB_Middle", "BB_Lower",
        "ATR_14",
        "Stoch_K", "Stoch_D",
        "ADX_14",
        "OBV",
        "VWAP_Dev",
    ]

    result = df[cols].copy()

    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].round(4)

    # Strict warmup filter — drop rows until all core indicators are valid.
    # SMA_200 requires 200 days minimum, so this is the binding constraint.
    result.dropna(
        subset=["SMA_20", "SMA_50", "SMA_200", "EMA_9", "EMA_21",
                "MACD", "RSI_14"],
        inplace=True
    )

    if result.empty:
        return None, "all rows dropped after warmup filter (insufficient history)"

    return result, None


# ═══════════════════════════════════════════════════════════
# SECTION 3 — MAIN LOOP
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("TECHNICAL INDICATORS — hedge_v2")
    print("=" * 60)

    # ── Load OHLCV ────────────────────────────────────────────────────
    print("\nLoading OHLCV from MySQL...")
    ohlcv = pd.read_sql(
        f"SELECT Date, Ticker, Open, High, Low, Close, Adj_Close, Volume, VWAP_Daily "
        f"FROM {TABLES['ohlcv']}",
        con=engine
    )
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])
    ohlcv = ohlcv.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Exclude Tier X
    ohlcv = ohlcv[~ohlcv["Ticker"].isin(TIER_X_EXCLUDED)]

    all_tickers = sorted(ohlcv["Ticker"].unique())
    print(f"Loaded {len(ohlcv):,} rows | {len(all_tickers)} tickers after Tier X exclusion")

    # ── Resume support ────────────────────────────────────────────────
    done    = get_already_computed()
    pending = [t for t in all_tickers if t not in done]

    print(f"\nTotal   : {len(all_tickers)}")
    print(f"Done    : {len(done)}")
    print(f"Pending : {len(pending)}\n")

    if not pending:
        print("All tickers already computed.")
        return

    # ── Process ───────────────────────────────────────────────────────
    BATCH_SIZE = 50   # save to DB every N tickers
    batch      = []
    failed     = []
    skipped    = []
    success    = 0

    for idx, ticker in enumerate(pending, 1):
        df     = ohlcv[ohlcv["Ticker"] == ticker].copy()
        result, error = process_ticker(df, ticker)

        if error:
            skipped.append(ticker)
        else:
            batch.append(result)
            success += 1

        # Batch save
        if len(batch) >= BATCH_SIZE:
            combined = pd.concat(batch).reset_index(drop=True)
            save_to_db(combined, TABLES["indicators"], engine)
            batch = []

        # Progress every 50 tickers
        if idx % 50 == 0 or idx == len(pending):
            print(f"  [{idx}/{len(pending)}] Done: {success} | Skipped: {len(skipped)}")

    # Save remainder
    if batch:
        combined = pd.concat(batch).reset_index(drop=True)
        save_to_db(combined, TABLES["indicators"], engine)

    print(f"""
{'='*60}
COMPLETE
  Computed : {success}
  Skipped  : {len(skipped)}
  Failed   : {len(failed)}
{'='*60}""")

    if skipped:
        print(f"Skipped : {skipped}")

    print("\nNext: python data/screener_fundamentals.py")


if __name__ == "__main__":
    main()