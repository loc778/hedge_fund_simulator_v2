# Run locally: python data/export_hmm_features.py
from sqlalchemy import text
import pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine

engine = get_engine()
query = text("""
    SELECT Date,
           Nifty50_Close,
           India_VIX,
           FII_Monthly_Net_Cr,
           FII_Daily_Net_Cr,
           FII_Source_Flag
    FROM macro_indicators
    WHERE Date >= '2023-09-01'
    ORDER BY Date
""")
with engine.connect() as conn:
    df = pd.read_sql(query, conn)

# Compute HMM features
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['Nifty_Return']    = df['Nifty50_Close'].pct_change()
df['FII_Flow']        = df['FII_Daily_Net_Cr'].fillna(df['FII_Monthly_Net_Cr'])
df['FII_Flow_zscore'] = (df['FII_Flow'] - df['FII_Flow'].mean()) / df['FII_Flow'].std()

df.to_csv('exports/hmm_features.csv', index=False)
print(f"Saved {len(df)} rows")