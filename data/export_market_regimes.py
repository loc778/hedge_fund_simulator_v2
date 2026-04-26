# data/export_market_regimes.py
import sys, os, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from sqlalchemy import text

engine = get_engine()
with engine.connect() as conn:
    df = pd.read_sql(text(
        "SELECT Date, Regime_Int, Regime_Label FROM market_regimes ORDER BY Date"
    ), conn)

df['Date'] = pd.to_datetime(df['Date'])

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'exports', 'market_regimes.csv')
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows → {out}")
print(df[df['Date'] >= '2024-01-01']['Regime_Label'].value_counts())