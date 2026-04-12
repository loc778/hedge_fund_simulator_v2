# data/indicators.py — hedge_v2
# FIXED VERSION (EMA + SMA + Adj_Close consistency)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import ta
import numpy as np

from config import TABLES, TIER_X_EXCLUDED
from data.db import get_engine, save_to_db

engine = get_engine()

# ── Load OHLCV ────────────────────────────────────────────────────────
print("📥 Loading OHLCV from MySQL...")

ohlcv = pd.read_sql(
    f"SELECT Date, Ticker, Open, High, Low, Close, Adj_Close, Volume, VWAP_Daily "
    f"FROM {TABLES['ohlcv']}",
    con=engine
)

ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])
ohlcv = ohlcv.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# Drop Tier X tickers
before = ohlcv["Ticker"].nunique()
ohlcv  = ohlcv[~ohlcv["Ticker"].isin(TIER_X_EXCLUDED)]
after  = ohlcv["Ticker"].nunique()

print(f"✅ Loaded {len(ohlcv):,} rows | {after} tickers ({before - after} Tier X excluded)\n")


# ── Indicator Calculation ─────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators for a single stock DataFrame.
    """

    # Ensure ordering
    df = df.sort_values("Date").reset_index(drop=True)

    # 🔥 Use Adj_Close everywhere
    price = df["Adj_Close"]

    # ── SMA (strict warmup) ──
    df["SMA_20"]  = price.rolling(window=20, min_periods=20).mean()
    df["SMA_50"]  = price.rolling(window=50, min_periods=50).mean()
    df["SMA_200"] = price.rolling(window=200, min_periods=200).mean()

    # ── EMA (stable pandas version) ──
    df["EMA_9"]  = price.ewm(span=9, adjust=False).mean()
    df["EMA_21"] = price.ewm(span=21, adjust=False).mean()

    # ── MACD (recomputed using Adj_Close) ──
    ema_fast = price.ewm(span=12, adjust=False).mean()
    ema_slow = price.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # ── RSI ──
    df["RSI_14"] = ta.momentum.rsi(price, window=14)

    # ── Bollinger Bands ──
    bb = ta.volatility.BollingerBands(price, window=20, window_dev=2)
    df["BB_Upper"]  = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"]  = bb.bollinger_lband()

    # ── ATR (still uses OHLC — correct) ──
    df["ATR_14"] = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=14
    )

    # ── Stochastic ──
    stoch = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"], window=14, smooth_window=3
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # ── ADX ──
    df["ADX_14"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)

    # ── OBV (volume-based, keep Close) ──
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    # ── VWAP Deviation ──
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
        "SMA_20", "SMA_50", "SMA_200", "EMA_9", "EMA_21",
        "MACD", "MACD_Signal", "MACD_Hist", "RSI_14",
        "BB_Upper", "BB_Middle", "BB_Lower", "ATR_14",
        "Stoch_K", "Stoch_D", "ADX_14",
        "OBV", "VWAP_Dev",
    ]

    result = df[cols].copy()

    # Round numeric columns
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].round(4)

    # 🔥 STRICT WARMUP FILTER (NO HARD CODING)
    result.dropna(
        subset=[
            "SMA_20",
            "SMA_50",
            "SMA_200",
            "EMA_9",
            "EMA_21",
            "MACD",
            "RSI_14"
        ],
        inplace=True
    )

    if result.empty:
        return None, "all rows dropped after warmup removal"

    return result, None


# ── Main Loop ─────────────────────────────────────────────────────────

tickers_in_db = ohlcv["Ticker"].unique()
print(f"⚙️  Computing indicators for {len(tickers_in_db)} tickers...\n")

all_data = []
failed   = []

for ticker in tickers_in_db:
    df = ohlcv[ohlcv["Ticker"] == ticker].copy()
    result, error = process_ticker(df, ticker)

    if error:
        print(f"  ⚠️  {ticker} — {error}")
        failed.append(ticker)
        continue

    all_data.append(result)
    print(f"  ✅ {ticker} — {len(result)} rows")


# ── Save ──────────────────────────────────────────────────────────────

if all_data:
    combined = pd.concat(all_data).reset_index(drop=True)
    save_to_db(combined, TABLES["indicators"], engine)
    print(f"\n✅ Total rows saved: {len(combined):,}")
else:
    print("\n❌ No indicator data to save.")

if failed:
    print(f"\n⚠️  {len(failed)} tickers failed or skipped: {failed}")

print("\n📋 Next step: python data/screener_fundamentals.py")