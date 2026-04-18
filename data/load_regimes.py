"""
data/load_regimes.py
--------------------
Reads market_regimes_latest.csv (downloaded from Google Drive after HMM Colab run)
and writes to market_regimes table in hedge_v2_db.

Run after every HMM Colab run:
    python data/load_regimes.py

Expected CSV columns:
    Date, Regime_Label, Regime_Int,
    Prob_Bull, Prob_Bear, Prob_HighVol, Prob_Sideways
"""

import sys
import os
import pandas as pd
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TABLES, MODEL_VERSION
from data.db import get_engine, save_to_db

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH   = Path(__file__).parent.parent / 'models' / 'market_regimes_latest.csv'
TABLE_NAME = TABLES['market_regimes']

# ── Load CSV ──────────────────────────────────────────────────────────────────
if not CSV_PATH.exists():
    print(f'ERROR: CSV not found at {CSV_PATH}')
    print('Download market_regimes_latest.csv from Google Drive into models/ folder.')
    sys.exit(1)

df = pd.read_csv(CSV_PATH, parse_dates=['Date'])
df['Date'] = df['Date'].dt.date.astype(str)

# ── Validate columns ──────────────────────────────────────────────────────────
required = ['Date', 'Regime_Label', 'Regime_Int',
            'Prob_Bull', 'Prob_Bear', 'Prob_HighVol', 'Prob_Sideways']
missing = [c for c in required if c not in df.columns]
if missing:
    print(f'ERROR: Missing columns in CSV: {missing}')
    sys.exit(1)

# ── Add model version ─────────────────────────────────────────────────────────
df['Model_Version'] = MODEL_VERSION

print(f'Rows to load   : {len(df):,}')
print(f'Date range     : {df.Date.min()} → {df.Date.max()}')
print(f'Regime dist    :')
print(df['Regime_Label'].value_counts().to_string())

# ── Truncate + reload (regimes are fully recomputed on every HMM run) ─────────
engine = get_engine()
with engine.connect() as conn:
    conn.execute(__import__('sqlalchemy').text(f'TRUNCATE TABLE {TABLE_NAME}'))
    conn.commit()

save_to_db(df[required + ['Model_Version']], TABLE_NAME, engine)

# ── Verify ────────────────────────────────────────────────────────────────────
with engine.connect() as conn:
    result = conn.execute(
        __import__('sqlalchemy').text(f'SELECT COUNT(*) FROM {TABLE_NAME}')
    ).fetchone()
    print(f'\nRows in {TABLE_NAME}: {result[0]:,}')

print('Done. market_regimes table updated.')