"""
data/export_prices.py
---------------------
Exports Adj_Close prices from nifty500_ohlcv for the backtest period.
Run locally after ensuring MySQL is running.

Output: exports/backtest_prices.csv
Columns: Date, Ticker, Adj_Close
"""

import sys
import os
import pandas as pd
from pathlib import Path
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine

START_DATE  = '2010-01-04 '  # buffer before backtest start for T+1 logic
END_DATE    = '2018-04-14'
OUTPUT_DIR  = Path(__file__).parent.parent / 'exports'
OUTPUT_FILE = OUTPUT_DIR / 'backtest_prices.csv'

OUTPUT_DIR.mkdir(exist_ok=True)

engine = get_engine()
print(f'Fetching prices {START_DATE} -> {END_DATE} ...')

query = text("""
    SELECT Date, Ticker, Adj_Close
    FROM nifty500_ohlcv
    WHERE Date >= :start
      AND Date <= :end
      AND Adj_Close IS NOT NULL
    ORDER BY Ticker, Date
""")

with engine.connect() as conn:
    df = pd.read_sql(query, conn, params={'start': START_DATE, 'end': END_DATE})

print(f'Rows fetched   : {len(df):,}')
print(f'Tickers        : {df["Ticker"].nunique()}')
print(f'Date range     : {df["Date"].min()} -> {df["Date"].max()}')

df.to_csv(OUTPUT_FILE, index=False)
print(f'Saved: {OUTPUT_FILE}')