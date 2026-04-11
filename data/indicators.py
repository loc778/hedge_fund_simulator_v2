# data/indicators.py — hedge_v2
# ═══════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# Reads nifty500_ohlcv → computes indicators → saves to nifty500_indicators
#
# Skips TIER_X_EXCLUDED tickers (< 252 days, recent IPOs).
# Drops warmup rows where RSI/MACD are still NaN (first ~33 rows per stock).
# ═══════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import ta
import time

from config import TABLES, TIER_X_EXCLUDED
from data.db import get_engine, save_to_db

engine = get_engine()

# ── Load OHLCV ────────────────────────────────────────────────────────
print("📥 Loading OHLCV from MySQL...")

ohlcv = pd.read_sql(
    f"SELECT Date, Ticker, Open, High, Low, Close, Volume, VWAP_Daily "
    f"FROM {TABLES['ohlcv']}",
    con=engine
)
ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])
ohlcv = ohlcv.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# Drop Tier X tickers — insufficient history, excluded from all training
before = ohlcv["Ticker"].nunique()
ohlcv  = ohlcv[~ohlcv["Ticker"].isin(TIER_X_EXCLUDED)]
after  = ohlcv["Ticker"].nunique()

print(f"✅ Loaded {len(ohlcv):,} rows | {after} tickers ({before - after} Tier X excluded)\n")


# ── Indicator Calculation ─────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators for a single stock DataFrame.
    df must be sorted by Date ascending and contain:
      Open, High, Low, Close, Volume, VWAP_Daily
    """
    # Moving Averages
    df["SMA_20"]  = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"]  = ta.trend.sma_indicator(df["Close"], window=50)
    df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)
    df["EMA_9"]   = ta.trend.ema_indicator(df["Close"], window=9)
    df["EMA_21"]  = ta.trend.ema_indicator(df["Close"], window=21)

    # MACD
    macd              = ta.trend.MACD(df["Close"], window_fast=12,
                                       window_slow=26, window_sign=9)
    df["MACD"]        = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"]   = macd.macd_diff()

    # RSI
    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)

    # Bollinger Bands
    bb              = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_Upper"]  = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"]  = bb.bollinger_lband()

    # ATR
    df["ATR_14"] = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=14
    )

    # Stochastic
    stoch         = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"], window=14, smooth_window=3
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # ADX
    df["ADX_14"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)

    # OBV
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    # VWAP Deviation (% deviation from daily VWAP)
    df["VWAP_Dev"] = (
        (df["Close"] - df["VWAP_Daily"]) / df["VWAP_Daily"].replace(0, float("nan")) * 100
    )

    return df


def process_ticker(df: pd.DataFrame, ticker: str):
    """
    Process one ticker. Returns (result_df, error_string).
    Returns (None, reason) if the stock should be skipped.
    """
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

    # Drop warmup rows (RSI and MACD not yet valid)
    result.dropna(subset=["RSI_14", "MACD"], inplace=True)

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