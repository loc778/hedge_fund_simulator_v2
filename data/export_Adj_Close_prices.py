"""
data/export_prices.py
---------------------
Exports Adj_Close prices from nifty500_ohlcv for the holdout backtest period.
Run locally after ensuring MySQL is running.

Output : exports/backtest_prices.csv
Columns: Ticker, Date, Adj_Close
"""

import sys
import os
import pandas as pd
from pathlib import Path
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine

# Holdout period: 2024-01-01 → today
# Buffer: no end date cap — export everything available so calibration
# can compute 21d/63d/252d forward returns even for recent signals.
START_DATE  = '2024-01-01'
OUTPUT_DIR  = Path(__file__).parent.parent / 'exports'
OUTPUT_FILE = OUTPUT_DIR / 'backtest_prices.csv'

OUTPUT_DIR.mkdir(exist_ok=True)

engine = get_engine()
print(f'Fetching prices from {START_DATE} → latest available ...')

query = text("""
    SELECT Ticker, Date, Adj_Close
    FROM nifty500_ohlcv
    WHERE Date >= :start
    ORDER BY Ticker, Date
""")

with engine.connect() as conn:
    df = pd.read_sql(query, conn, params={'start': START_DATE})

print(f'Rows fetched : {len(df):,}')
print(f'Tickers      : {df["Ticker"].nunique()}')
print(f'Date range   : {df["Date"].min()} → {df["Date"].max()}')

df.to_csv(OUTPUT_FILE, index=False)
print(f'Saved        : {OUTPUT_FILE}')