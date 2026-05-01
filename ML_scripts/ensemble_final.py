# =============================================================================
# BLOCK A — ENVIRONMENT SETUP
# =============================================================================

# Run manually if needed: pip install xgboost lightgbm pyarrow tensorflow hmmlearn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("GPU disabled — CPU only")

import gc
import glob
import itertools
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
print(f"TF: {tf.__version__}  |  NumPy: {np.__version__}  |  Pandas: {pd.__version__}")


class CrossSectionalNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_cols = None
        self.min_stocks   = 10
        self.stats_       = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, dates=None):
        X = np.array(X, dtype=np.float32)
        if self.stats_ is None:
            return X
        if dates is not None:
            dates = pd.to_datetime(dates)
            out   = np.zeros_like(X)
            for i, d in enumerate(dates):
                d_ts = pd.Timestamp(d).normalize()
                if d_ts in self.stats_.index:
                    row = self.stats_.loc[d_ts]
                else:
                    avail = self.stats_.index[self.stats_.index <= d_ts]
                    if len(avail) == 0:
                        out[i] = X[i]
                        continue
                    row = self.stats_.loc[avail[-1]]
                means = np.array([row.get(f'mean_{f}', 0.0) for f in self.feature_cols], dtype=np.float32)
                stds  = np.array([row.get(f'std_{f}',  1.0) for f in self.feature_cols], dtype=np.float32)
                stds[stds == 0] = 1.0
                out[i] = (X[i] - means) / stds
            return out
        else:
            means = np.array([self.stats_[f'mean_{f}'].mean() for f in self.feature_cols], dtype=np.float32)
            stds  = np.array([self.stats_[f'std_{f}'].mean()  for f in self.feature_cols], dtype=np.float32)
            stds[stds == 0] = 1.0
            return (X - means) / stds

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

print("CrossSectionalNormalizer defined.")

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(BASE_DIR, 'ML_models')
RESULTS_DIR  = os.path.join(BASE_DIR, 'exports', 'model_output')
PARQUET_PATH = os.path.join(BASE_DIR, 'exports', 'features_master_latest.parquet')
PRICES_PATH  = os.path.join(BASE_DIR, 'exports', 'backtest_prices.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)
DATE_STAMP = datetime.today().strftime('%Y%m%d')
print(f"Date stamp     : {DATE_STAMP}")
print(f"Parquet exists : {os.path.exists(PARQUET_PATH)}")
print(f"Prices exists  : {os.path.exists(PRICES_PATH)}")


# =============================================================================
# BLOCK B — CONFIGURATION
# =============================================================================

# ── Shared time-range config ──────────────────────────────────────────────────
HOLDOUT_START = '2024-01-01'
TRAIN_END     = '2023-12-31'
BACKTEST_END  = datetime.today().strftime('%Y-%m-%d')

# ── Model file patterns ───────────────────────────────────────────────────────
XGB_PATTERN          = 'xgboost_v*.pkl'
LGB_PATTERN          = 'lightgbm_v*.pkl'
LSTM_KERAS_PATTERN   = 'lstm_v*.keras'
LSTM_NORM_PATTERN    = 'lstm_norm_v*.pkl'
HMM_MODEL_PATTERN    = 'hmm_model_v*.pkl'
HMM_SCALER_PATTERN   = 'hmm_scaler_v*.pkl'
HMM_STATEMAP_PATTERN = 'hmm_statemap_v*.pkl'

# ── Ensemble score weights by regime ─────────────────────────────────────────
REGIME_WEIGHTS = {
    0: {'xgb': 0.571, 'lgb': 0.429},
    1: {'xgb': 0.500, 'lgb': 0.500},
    2: {'xgb': 0.500, 'lgb': 0.500},
    3: {'xgb': 0.500, 'lgb': 0.500},
}

# ── Tier multiplier (used in ensemble score calc) ─────────────────────────────
TIER_MULT = {1: 1.00, 2: 1.00, 3: 1.00}

# ── Signal thresholds (used in signal generation for df_signals) ─────────────
# NOTE: BUY_THRESHOLD / SELL_THRESHOLD here are only for the signal-generation
#       step (df_signals['Signal'] column).  The backtest engine uses
#       BUY_THRESHOLD_ENH / SELL_THRESHOLD_ENH for trade decisions.
BUY_THRESHOLD  = 0.80
SELL_THRESHOLD = 0.20

# ── Shared column names ───────────────────────────────────────────────────────
DATE_COL   = 'Date'
TICKER_COL = 'Ticker'
TIER_COL   = 'Data_Tier'
TARGET_COL = 'Target_Rank_21d'

# ── LSTM sequence length (default; may be overridden from model meta) ─────────
SEQ_LEN = 60

# ── Starting NAV ──────────────────────────────────────────────────────────────
STARTING_NAV = 10_000_000

# ── Backtest engine parameters ────────────────────────────────────────────────
LONG_TOP_K          = 20
SHORT_TOP_K_ENH     = 6
MAX_POSITIONS_ENH   = 36
BUY_THRESHOLD_ENH   = 0.85
SELL_THRESHOLD_ENH  = 0.05
POS_CAP_LARGE       = 0.10
POS_CAP_MID         = 0.05
POS_CAP_SHORT       = 0.025
CASH_FLOOR_ENH      = 0.02
REGIME_DEPLOY_ENH   = {0: 0.99, 1: 0.88, 2: 0.90, 3: 0.94}
STOP_LOSS_ENH       = -0.12
SHORT_STOP_LOSS_ENH =  0.10
MAX_SECTOR_PCT      = 0.30
ATR_FALLBACK        = 0.02
INV_VOL_WEIGHT      = 0.60
CONVICTION_WEIGHT   = 0.40
DD_DEFENCE_THRESHOLD = -0.12
DD_DEFENCE_REDUCE    =  0.20
MOMENTUM_GAIN_THRESHOLD = 0.05
MOMENTUM_TOPUP_FACTOR   = 1.25
TARGET_VOL_ANN_ENH  = 0.15
LSTM_VOL_SCALING    = True
SHORT_REGIMES       = {1, 2}
MAX_SHORT_BOOK_ENH  = 0.10
COST_BPS            = 19.5
COST_RATE_ENH       = COST_BPS / 10_000

# ── normalise any Timestamp to tz-naive date Timestamp ──────
def _ts(t):
    """Strip time and timezone — makes rank_pivot.index comparable to daily dates."""
    return pd.Timestamp(t).normalize().tz_localize(None)

print("Configuration loaded.")
print(f"  BUY threshold   : {BUY_THRESHOLD_ENH}")
print(f"  Cash floor      : {CASH_FLOOR_ENH*100:.0f}%")
print(f"  Position cap L  : {POS_CAP_LARGE*100:.0f}%")
print(f"  Stop-loss       : {STOP_LOSS_ENH*100:.0f}%")
print(f"  Bull deploy     : {REGIME_DEPLOY_ENH[0]*100:.0f}%")


# =============================================================================
# BLOCK C — MODELS & DATA
# =============================================================================

def load_latest(pattern, base=MODELS_DIR):
    files = sorted(glob.glob(os.path.join(base, pattern)))
    if not files:
        raise FileNotFoundError(f"No file matching: {os.path.join(base, pattern)}")
    path = files[-1]
    print(f"  Loaded : {os.path.basename(path)}")
    return path

print("Loading models ...")

xgb_path = load_latest(XGB_PATTERN)
with open(xgb_path, 'rb') as f: xgb_data = pickle.load(f)
xgb_model    = xgb_data['model']
XGB_FEATURES = xgb_data['feature_cols']
print(f"    XGB features  : {len(XGB_FEATURES)}")
print(f"    XGB holdout IC: {xgb_data.get('holdout_ic', xgb_data.get('mean_ic', 'N/A'))}")

lgb_path = load_latest(LGB_PATTERN)
with open(lgb_path, 'rb') as f: lgb_data = pickle.load(f)
lgb_model    = lgb_data['model']
LGB_FEATURES = lgb_data['feature_cols']
print(f"    LGB features  : {len(LGB_FEATURES)}")
print(f"    LGB holdout IC: {lgb_data.get('holdout_ic', lgb_data.get('mean_ic', 'N/A'))}")

lstm_keras_path = load_latest(LSTM_KERAS_PATTERN)
lstm_norm_path  = load_latest(LSTM_NORM_PATTERN)
lstm_model      = keras.models.load_model(lstm_keras_path)
with open(lstm_norm_path, 'rb') as f: lstm_meta = pickle.load(f)

print(f"    lstm_norm type : {type(lstm_meta)}")
if isinstance(lstm_meta, dict):
    LSTM_FEATURES   = lstm_meta['feature_cols']
    lstm_scalers    = lstm_meta.get('scalers', lstm_meta.get('normalizer', {}))
    SEQ_LEN         = lstm_meta.get('seq_len', SEQ_LEN)
    lstm_normalizer = lstm_meta.get('normalizer', None)
else:
    lstm_normalizer = lstm_meta
    LSTM_FEATURES   = getattr(lstm_meta, 'feature_cols',
                        getattr(lstm_meta, 'feature_names_',
                        getattr(lstm_meta, 'features_', None)))
    lstm_scalers    = {}
    SEQ_LEN         = getattr(lstm_meta, 'seq_len', SEQ_LEN)
    if LSTM_FEATURES is None:
        raise ValueError("Cannot find feature_cols on CrossSectionalNormalizer.")

print(f"    LSTM features : {len(LSTM_FEATURES)}  SEQ_LEN={SEQ_LEN}")

hmm_model_path    = load_latest(HMM_MODEL_PATTERN)
hmm_scaler_path   = load_latest(HMM_SCALER_PATTERN)
hmm_statemap_path = load_latest(HMM_STATEMAP_PATTERN)
with open(hmm_model_path,    'rb') as f: hmm_model    = pickle.load(f)
with open(hmm_scaler_path,   'rb') as f: hmm_scaler   = pickle.load(f)
with open(hmm_statemap_path, 'rb') as f: hmm_statemap = pickle.load(f)
HMM_FEATURES     = hmm_statemap.get('feature_cols',
    ['Nifty_Return', 'India_VIX', 'Volatility_20d', 'Return_5d', 'FII_Flow_zscore'])
REGIME_LABEL_MAP = hmm_statemap.get('label_map', {0:'Bull',1:'Bear',2:'HighVol',3:'Sideways'})
print(f"    HMM features  : {HMM_FEATURES}")
print("\nAll models loaded successfully.")

# ── Parquet ───────────────────────────────────────────────────────────────────
print(f"\nLoading parquet: {PARQUET_PATH}")
df_raw = pd.read_parquet(PARQUET_PATH)
df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL])

non_feat = {DATE_COL, TICKER_COL, TIER_COL, 'FII_Source_Flag', 'id',
            'Target_Return_21d', 'Target_Rank_21d', 'Target_Direction_Median',
            'Target_Direction_Tertile', 'Target_Vol_5d', 'Sector'}
for col in df_raw.columns:
    if col not in non_feat and df_raw[col].dtype == object:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

holdout_start_dt = pd.Timestamp(HOLDOUT_START)
pre_holdout = df_raw[DATE_COL] < holdout_start_dt
if pre_holdout.sum() > 0:
    print(f"⚠️  Dropping {pre_holdout.sum():,} rows before {HOLDOUT_START}")
    df_raw = df_raw[~pre_holdout].copy()

seq_buffer_start = holdout_start_dt - pd.tseries.offsets.BDay(SEQ_LEN + 10)
df_full = pd.read_parquet(PARQUET_PATH)
df_full[DATE_COL] = pd.to_datetime(df_full[DATE_COL])
for col in df_full.columns:
    if col not in non_feat and df_full[col].dtype == object:
        df_full[col] = pd.to_numeric(df_full[col], errors='coerce')
df_full = df_full[df_full[DATE_COL] >= seq_buffer_start].copy()
df_full = df_full.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)

sector_map = (df_raw[[TICKER_COL, 'Sector']].drop_duplicates()
              if 'Sector' in df_raw.columns
              else df_raw[[TICKER_COL]].drop_duplicates().assign(Sector='Unknown'))
del df_raw; gc.collect()

print(f"Holdout rows : {len(df_full[df_full[DATE_COL] >= holdout_start_dt]):,}")
print(f"Tickers      : {df_full[TICKER_COL].nunique()}")
print(f"Date range   : {df_full[DATE_COL].min().date()} → {df_full[DATE_COL].max().date()}")

# ── HMM Regimes ───────────────────────────────────────────────────────────────
print("\nLoading pre-computed HMM regimes ...")
engine = get_engine()
regimes_all = pd.read_sql(
    "SELECT Date, Regime_Label FROM market_regimes ORDER BY Date",
    engine
)
regimes_all['Date'] = pd.to_datetime(regimes_all['Date'])
regimes_df  = regimes_all[regimes_all['Date'] >= holdout_start_dt].copy()
regimes_df  = regimes_df.sort_values('Date').reset_index(drop=True)

LABEL_TO_INT     = hmm_statemap.get('LABEL_TO_INT', {'Bull':0,'Bear':1,'HighVol':2,'Sideways':3})
REGIME_LABEL_MAP = {v: k for k, v in LABEL_TO_INT.items()}
regimes_df['Regime_Int'] = regimes_df['Regime_Label'].map(LABEL_TO_INT).fillna(3).astype(int)
regimes_df = regimes_df.rename(columns={'Date': DATE_COL})

print(f"Holdout regimes : {len(regimes_df)} days")
for label, cnt in regimes_df['Regime_Label'].value_counts().items():
    ri = LABEL_TO_INT.get(label, '?')
    print(f"  {label:<12}: {cnt:4d} days ({cnt/len(regimes_df)*100:.1f}%)  [int={ri}]")

# ── XGB + LGB Predictions ─────────────────────────────────────────────────────
df_bt = df_full[
    (df_full[DATE_COL] >= holdout_start_dt) &
    (df_full[TIER_COL].isin([1, 2, 3]))
].copy()
print(f"\nInference rows : {len(df_bt):,}  |  Dates: {df_bt[DATE_COL].nunique()}")

df_bt['XGB_Score'] = xgb_model.predict(df_bt[XGB_FEATURES].copy())
df_bt['LGB_Score'] = lgb_model.predict(df_bt[LGB_FEATURES].copy())
df_bt['XGB_Rank']  = df_bt.groupby(DATE_COL)['XGB_Score'].rank(pct=True)
df_bt['LGB_Rank']  = df_bt.groupby(DATE_COL)['LGB_Score'].rank(pct=True)
print("XGB + LGB predictions + cross-sectional ranking done.")

# ── LSTM Vol Predictions ───────────────────────────────────────────────────────
print("\nGenerating LSTM vol predictions (Head 2) ...")
lstm_records = []
tickers_all  = df_bt[TICKER_COL].unique()
n_ok = n_skip = 0

for i, ticker in enumerate(tickers_all):
    t_hist = df_full[df_full[TICKER_COL] == ticker].sort_values(DATE_COL).reset_index(drop=True)
    if len(t_hist) < SEQ_LEN + 1: n_skip += 1; continue
    missing = [f for f in LSTM_FEATURES if f not in t_hist.columns]
    if missing:                    n_skip += 1; continue

    feat_df  = t_hist[LSTM_FEATURES].ffill().bfill().fillna(0.0)
    try:
        feat_arr  = feat_df.values.astype(np.float32)
        feat_vals = lstm_normalizer.transform(feat_arr, dates=t_hist[DATE_COL].values)
    except Exception: n_skip += 1; continue

    holdout_pos = t_hist.index[t_hist[DATE_COL] >= holdout_start_dt].tolist()
    seqs, dates = [], []
    for pos in holdout_pos:
        if pos < SEQ_LEN: continue
        seqs.append(feat_vals[pos - SEQ_LEN: pos])
        dates.append(t_hist.loc[pos, DATE_COL])

    if len(seqs) == 0: n_skip += 1; continue

    try:
        preds = lstm_model.predict(np.array(seqs, dtype=np.float32), verbose=0, batch_size=256)
        vol_preds = preds[1].flatten() if isinstance(preds, (list,tuple)) and len(preds)==2 else preds.flatten()
    except Exception: n_skip += 1; continue

    for d, v in zip(dates, vol_preds):
        lstm_records.append({TICKER_COL: ticker, DATE_COL: d, 'LSTM_Vol': float(v)})
    n_ok += 1
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(tickers_all)}  ok={n_ok}  skip={n_skip}")

lstm_vol_df = (pd.DataFrame(lstm_records) if lstm_records
               else pd.DataFrame(columns=[TICKER_COL, DATE_COL, 'LSTM_Vol']))
print(f"LSTM vol done : {len(lstm_vol_df):,} rows  ok={n_ok}  skip={n_skip}")
del lstm_records; gc.collect()

# ── Combine Ensemble Signals ───────────────────────────────────────────────────
df_signals = df_bt.merge(lstm_vol_df[[TICKER_COL, DATE_COL, 'LSTM_Vol']],
                         on=[TICKER_COL, DATE_COL], how='left')

reg_merge = regimes_df.drop_duplicates(subset=[DATE_COL]).copy()
for c in ['Regime_Int', 'Regime_Label']:
    if c in df_signals.columns: df_signals = df_signals.drop(columns=c)
df_signals = df_signals.merge(reg_merge[[DATE_COL,'Regime_Int','Regime_Label']],
                               on=DATE_COL, how='left')
df_signals['Regime_Int'] = df_signals['Regime_Int'].fillna(3).astype(int)

def compute_ensemble_score(row):
    w = REGIME_WEIGHTS.get(row['Regime_Int'], REGIME_WEIGHTS[3])
    return w['xgb'] * row['XGB_Rank'] + w['lgb'] * row['LGB_Rank']

df_signals['Raw_Score']      = df_signals.apply(compute_ensemble_score, axis=1)
df_signals['Tier_Mult']      = df_signals[TIER_COL].map(TIER_MULT).fillna(0.50)
df_signals['Ensemble_Score'] = 0.5 + (df_signals['Raw_Score'] - 0.5) * df_signals['Tier_Mult']

df_signals = df_signals.merge(sector_map, on=TICKER_COL, how='left')
df_signals['Sector'] = df_signals['Sector'].fillna('Unknown')

tier_a_mean = (df_signals[df_signals[TIER_COL]==1]
               .groupby([DATE_COL,'Sector'])['Raw_Score'].mean().reset_index()
               .rename(columns={'Raw_Score':'Sector_Score'}))
df_signals   = df_signals.merge(tier_a_mean, on=[DATE_COL,'Sector'], how='left')
tier_c       = df_signals[TIER_COL] == 3
df_signals.loc[tier_c, 'Ensemble_Score'] = (
    0.5 + (df_signals.loc[tier_c,'Sector_Score'].fillna(0.5) - 0.5) * TIER_MULT[3])

df_signals['Final_Rank'] = df_signals.groupby(DATE_COL)['Ensemble_Score'].rank(pct=True)



# Signal column: used for output CSV only; backtest engine uses Final_Rank directly.
df_signals['Signal'] = 0
df_signals.loc[df_signals['Final_Rank'] >= BUY_THRESHOLD,  'Signal'] =  1
df_signals.loc[df_signals['Final_Rank'] <= SELL_THRESHOLD, 'Signal'] = -1
df_signals.loc[(df_signals[TIER_COL]==3) & (df_signals['Signal']==-1), 'Signal'] = 0

df_signals['LSTM_Vol'] = df_signals['LSTM_Vol'].replace([np.inf,-np.inf], np.nan)
df_signals['LSTM_Vol'] = df_signals['LSTM_Vol'].fillna(df_signals['LSTM_Vol'].median())

try:
    calib_path = os.path.join(MODELS_DIR, 'rank_calibration.pkl')
    with open(calib_path,'rb') as f: calib_payload = pickle.load(f)
    calib_df  = calib_payload['calibration_df'].sort_values('rank_lo').reset_index(drop=True)
    bin_lows  = calib_df['rank_lo'].values
    bin_highs = calib_df['rank_hi'].values
    def get_calib(rank, col):
        for i in range(len(calib_df)):
            if bin_lows[i] <= rank <= bin_highs[i]:
                return calib_df.iloc[i].get(col, np.nan)
        return calib_df.iloc[(calib_df['rank_mid']-rank).abs().argmin()].get(col, np.nan)
    for col, src in [('Projected_Return_21d','mean_return_21d'),
                     ('Projected_Return_63d','mean_return_63d'),
                     ('Projected_Return_252d','mean_return_252d'),
                     ('Band_Low_21d','p05_return_21d'),
                     ('Band_High_21d','p95_return_21d')]:
        df_signals[col] = df_signals['Final_Rank'].apply(lambda r: get_calib(r, src))
    print("Calibration applied.")
except Exception as e:
    print(f"⚠️  Calibration not applied: {e}")
    for col in ['Projected_Return_21d','Projected_Return_63d','Projected_Return_252d',
                'Band_Low_21d','Band_High_21d']:
        df_signals[col] = np.nan

print(f"\nEnsemble signals : {len(df_signals):,} rows")
print(f"Date range       : {df_signals[DATE_COL].min().date()} → {df_signals[DATE_COL].max().date()}")

OUTPUT_COLS = [c for c in [TICKER_COL,DATE_COL,'Final_Rank','Signal','Regime_Int',
               TIER_COL,'Sector','LSTM_Vol','XGB_Rank','LGB_Rank',
               'Projected_Return_21d','Projected_Return_63d','Projected_Return_252d',
               'Band_Low_21d','Band_High_21d'] if c in df_signals.columns]
df_signals[OUTPUT_COLS].to_csv(os.path.join(RESULTS_DIR, f'signals_{DATE_STAMP}.csv'), index=False)

# ── Snapshot BEFORE any mutation ──────────────────────────────────────────────
df_signals_clean = df_signals.copy()
print(f"df_signals_clean snapshot: {df_signals_clean.shape}")

# ── DIAGNOSTIC: rank distribution ────────────────────────────────────────────
rank_sample = df_signals_clean['Final_Rank']
print(f"\nFinal_Rank distribution:")
print(f"  min={rank_sample.min():.4f}  p10={rank_sample.quantile(0.10):.4f}  "
      f"p50={rank_sample.quantile(0.50):.4f}  p82={rank_sample.quantile(0.82):.4f}  "
      f"max={rank_sample.max():.4f}")
n_above_threshold = (rank_sample >= BUY_THRESHOLD_ENH).sum()
print(f"  Rows with Final_Rank >= {BUY_THRESHOLD_ENH}: {n_above_threshold:,} "
      f"({n_above_threshold/len(rank_sample)*100:.1f}%)")
print(f"  Unique signal dates: {df_signals_clean[DATE_COL].nunique()}")


# =============================================================================
# BLOCK D — PRICE DATA
# =============================================================================

print("\nLoading backtest prices ...")
prices = pd.read_csv(PRICES_PATH, parse_dates=[DATE_COL])
prices[DATE_COL]    = pd.to_datetime(prices[DATE_COL])
prices['Adj_Close'] = pd.to_numeric(prices['Adj_Close'], errors='coerce')
prices = prices.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
prices['Daily_Ret'] = prices.groupby(TICKER_COL)['Adj_Close'].pct_change().clip(-0.5, 0.5)
daily_ret_pivot     = prices.pivot(index=DATE_COL, columns=TICKER_COL, values='Daily_Ret')

print(f"Price rows  : {len(prices):,}  |  Tickers: {prices[TICKER_COL].nunique()}")
print(f"Price range : {prices[DATE_COL].min().date()} → {prices[DATE_COL].max().date()}")
dr = prices['Daily_Ret'].dropna()
print(f"Daily ret   : mean={dr.mean()*100:.3f}%  std={dr.std()*100:.3f}%")

fo_df = pd.read_csv(os.path.join(BASE_DIR, 'files', 'fo_list.csv'))
fo_df.columns   = fo_df.columns.str.strip()
fo_df['SYMBOL'] = fo_df['SYMBOL'].str.strip()
FO_TICKERS      = set(fo_df['SYMBOL'].dropna().str.strip())
print(f"F&O tickers : {len(FO_TICKERS)}")

# ── DIAGNOSTIC: date alignment check ─────────────────────────────────────────
sig_dates   = set(df_signals_clean[DATE_COL].dt.normalize())
price_dates = set(daily_ret_pivot.index.normalize())
overlap_d   = sig_dates & price_dates
print(f"\nDate alignment: {len(sig_dates)} signal dates, {len(price_dates)} price dates, "
      f"{len(overlap_d)} overlap")
if len(overlap_d) == 0:
    print("⚠️  CRITICAL: zero date overlap between signals and prices!")
    print(f"   Signal dates sample : {sorted(sig_dates)[:3]}")
    print(f"   Price dates sample  : {sorted(price_dates)[:3]}")
else:
    print(f"   Signal date sample  : {sorted(sig_dates)[:3]}")
    print(f"   Price date sample   : {sorted(price_dates)[:3]}")


# =============================================================================
# BLOCK E — HELPER FUNCTIONS
# =============================================================================

def compute_metrics(nav_df, rf_rate=0.065):
    nav  = nav_df['NAV']
    rets = nav.pct_change().dropna()
    n    = len(rets)
    yrs  = n / 252
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    cagr      = (1 + total_ret) ** (1 / max(yrs, 0.001)) - 1
    vol_ann   = rets.std() * np.sqrt(252)
    sharpe    = (cagr - rf_rate) / vol_ann if vol_ann > 0 else 0
    roll_max  = nav.cummax()
    dd        = (nav - roll_max) / roll_max
    max_dd    = dd.min()
    calmar    = cagr / abs(max_dd) if max_dd != 0 else 0
    win_rate  = (rets > 0).mean()
    return {
        'Period'        : f"{nav_df.index[0].date()} → {nav_df.index[-1].date()}",
        'Total Return'  : f"{total_ret*100:.2f}%",
        'CAGR'          : f"{cagr*100:.2f}%",
        'Ann Volatility': f"{vol_ann*100:.2f}%",
        'Sharpe Ratio'  : f"{sharpe:.3f}",
        'Max Drawdown'  : f"{max_dd*100:.2f}%",
        'Calmar Ratio'  : f"{calmar:.3f}",
        'Win Rate'      : f"{win_rate*100:.2f}%",
        'Trading Days'  : n,
    }, rets, dd

print("compute_metrics() defined.")


def compute_atr_from_prices(ret_pivot: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR-proxy: rolling std of daily returns per ticker."""
    recent  = ret_pivot.tail(window + 1)
    atr_pct = recent.std(axis=0).fillna(ATR_FALLBACK).clip(lower=0.005)
    return atr_pct


def compute_position_sizes(
    candidates: pd.DataFrame,
    direction: str,
    nav: float,
    book_target_pct: float,
    deploy_cap: float,
    atr_series: pd.Series,
    lstm_vol_col: str = 'LSTM_Vol',
) -> pd.Series:
    if candidates.empty:
        return pd.Series(dtype=float)

    # Reset index so numpy ops stay aligned
    cands = candidates.reset_index(drop=True)
    tickers = cands[TICKER_COL].values

    atr_pct = cands[TICKER_COL].map(atr_series).fillna(ATR_FALLBACK).clip(lower=0.005)
    inv_vol = 1.0 / atr_pct.values

    if 'Final_Rank' in cands.columns:
        conviction = (cands['Final_Rank'].clip(0.0, 1.0).values if direction == 'long'
                      else (1.0 - cands['Final_Rank']).clip(0.0, 1.0).values)
        conv_norm  = conviction / (conviction.sum() + 1e-9) * len(conviction)
        combined_w = INV_VOL_WEIGHT * inv_vol + CONVICTION_WEIGHT * conv_norm
    else:
        combined_w = inv_vol

    raw_w            = combined_w / (combined_w.sum() + 1e-9)
    effective_target = min(book_target_pct, deploy_cap)
    raw_sizes        = raw_w * effective_target

    # BUG 3 FIX: explicit column access + reset index guarantees alignment
    if direction == 'long':
        if TIER_COL in cands.columns:
            is_midcap = cands[TIER_COL].isin([2, 3]).values
        else:
            is_midcap = np.zeros(len(cands), dtype=bool)
        caps = np.where(is_midcap, POS_CAP_MID, POS_CAP_LARGE)
    else:
        caps = np.full(len(cands), POS_CAP_SHORT)

    clipped = np.minimum(raw_sizes, caps)

    # BUG 4 FIX: compare annualised vol; cap downscale at 0.5× not 0.0×
    if LSTM_VOL_SCALING and lstm_vol_col in cands.columns:
        lstm_vol_raw = cands[lstm_vol_col].fillna(TARGET_VOL_ANN_ENH / np.sqrt(252)).values
        # Head 2 predicts daily vol → annualise for comparison
        lstm_vol_ann = lstm_vol_raw * np.sqrt(252)
        vol_scale    = (TARGET_VOL_ANN_ENH / lstm_vol_ann.clip(min=0.01)).clip(0.5, 1.5)
        clipped      = np.minimum(clipped * vol_scale, caps)

    total = clipped.sum()
    if total > 0:
        scale   = min(effective_target / total, 1.0)
        clipped = np.minimum(clipped * scale, caps)

    # Floor: only drop truly negligible positions (< 0.3% NAV)
    clipped = np.where(clipped >= 0.003, clipped, np.nan)
    return pd.Series(clipped, index=tickers).dropna()


def apply_sector_cap(candidates: pd.DataFrame, sizes: pd.Series,
                     max_sector_pct: float = MAX_SECTOR_PCT) -> pd.Series:
    """Trim positions so no sector > max_sector_pct of long book."""
    if candidates.empty or sizes.empty:
        return sizes
    df         = candidates.reset_index(drop=True).copy()
    df['size'] = df[TICKER_COL].map(sizes).fillna(0.0)
    total_size = df['size'].sum()
    if total_size == 0:
        return sizes
    sector_totals = df.groupby('Sector')['size'].sum()
    adjusted      = sizes.copy()
    for sector, sector_sum in sector_totals.items():
        if sector_sum / total_size > max_sector_pct:
            scale = (max_sector_pct * total_size) / sector_sum
            for t in df[df['Sector'] == sector][TICKER_COL]:
                if t in adjusted.index:
                    adjusted[t] *= scale
    return adjusted

print("All helper functions defined.")


# =============================================================================
# BLOCK F — BACKTEST ENGINE
# =============================================================================

def run_backtest(starting_nav: float, daily_ret_pivot: pd.DataFrame) -> pd.DataFrame:
    signals_df = df_signals_clean.copy().sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    signals_df[DATE_COL] = signals_df[DATE_COL].dt.normalize().dt.tz_localize(None)

    holdout_dates = sorted([d for d in daily_ret_pivot.index
                            if _ts(d) >= _ts(HOLDOUT_START)])

    rank_pivot     = signals_df.pivot_table(
        index=DATE_COL, columns=TICKER_COL, values='Final_Rank', aggfunc='first')
    regime_by_date = signals_df.groupby(DATE_COL)['Regime_Int'].first()

    signal_meta_cols = [c for c in [DATE_COL, TICKER_COL, 'Final_Rank', 'Data_Tier',
                                    'LSTM_Vol', 'Sector'] if c in signals_df.columns]
    signal_meta = signals_df[signal_meta_cols].copy()

    # Pre-normalise rank_pivot index once
    rp_idx_norm = rank_pivot.index.normalize().tz_localize(None)

    cash       = float(starting_nav)
    positions  = {}
    nav_series = []
    # Bi-weekly trigger: initialise far enough back so first holdout day rebalances
    last_rebalance_date = _ts(HOLDOUT_START) - pd.tseries.offsets.BDay(10)
    peak_nav   = float(starting_nav)

    rebalance_log = []

    for dt in holdout_dates:
        ts        = _ts(dt)
        rebalance = (ts - last_rebalance_date).days >= 14

        # ── Daily MTM ────────────────────────────────────────────────────────
        long_val = short_val = 0.0
        if dt in daily_ret_pivot.index:
            day_rets = daily_ret_pivot.loc[dt]
            for ticker, pos in list(positions.items()):
                # Skip MTM on entry day and already-stopped positions
                if pos['stopped'] or pos['entry_date'] == dt:
                    if pos['direction'] == 'long':
                        long_val  += pos['alloc'] + pos['pnl']
                    else:
                        short_val += pos['alloc'] - pos['pnl']
                    continue

                ret = day_rets.get(ticker, np.nan)
                if pd.isna(ret):
                    if pos['direction'] == 'long':
                        long_val  += pos['alloc'] + pos['pnl']
                    else:
                        short_val += pos['alloc'] - pos['pnl']
                    continue

                if pos['direction'] == 'long':
                    pos['pnl']       += pos['alloc'] * ret
                    pos['high_water'] = max(pos['high_water'], pos['alloc'] + pos['pnl'])
                    if (pos['alloc'] + pos['pnl']) / pos['high_water'] - 1 <= STOP_LOSS_ENH:
                        cash += (pos['alloc'] + pos['pnl']) * (1 - COST_RATE_ENH)
                        pos['stopped'] = True
                    else:
                        long_val += pos['alloc'] + pos['pnl']

                else:  # short
                    pos['pnl'] += pos['alloc'] * (-ret)
                    if pos['pnl'] / pos['alloc'] <= -SHORT_STOP_LOSS_ENH:
                        # AUDIT FIX 3: shorts close at alloc - pnl (profit reduces liability)
                        cash += (pos['alloc'] - pos['pnl']) * (1 - COST_RATE_ENH)
                        pos['stopped'] = True
                    else:
                        short_val += pos['alloc'] - pos['pnl']

        nav_now  = cash + long_val + short_val
        peak_nav = max(peak_nav, nav_now)
        in_defence = (nav_now / peak_nav) - 1 < DD_DEFENCE_THRESHOLD

        # ── Bi-weekly partial rebalance ───────────────────────────────────────
        if rebalance:
            last_rebalance_date = ts

            #  strict < so today's signal is never used today
            available = rank_pivot.index[rp_idx_norm < ts]     # was <=
            if len(available) == 0:
                rebalance_log.append({'date': dt, 'reason': 'no_signal_date',
                                      'n_long': 0, 'n_closed': 0, 'n_opened': 0})
                #  no append+continue — fall through to single append below
            else:
                signal_dt = available[-1]

                regime_int = int(regime_by_date.get(
                    signal_dt,
                    regime_by_date.get(rp_idx_norm[rank_pivot.index.get_loc(signal_dt)], 3)
                ))
                deploy_cap = REGIME_DEPLOY_ENH.get(regime_int, 0.90)
                if in_defence:
                    deploy_cap *= (1 - DD_DEFENCE_REDUCE)

                meta_today = signal_meta[signal_meta[DATE_COL] == signal_dt].copy()
                if meta_today.empty:
                    rebalance_log.append({'date': dt, 'reason': 'empty_meta',
                                          'n_long': 0, 'n_closed': 0, 'n_opened': 0})
                    #  fall through to single append below
                else:
                    ranks = rank_pivot.loc[signal_dt].dropna()

                    #  threshold fallback
                    all_long_ranks = ranks.nlargest(LONG_TOP_K)
                    above_thresh   = ranks[ranks >= BUY_THRESHOLD_ENH].nlargest(LONG_TOP_K)

                    if len(above_thresh) >= 3:
                        long_ranks = above_thresh
                    elif len(all_long_ranks) >= 3:
                        long_ranks = all_long_ranks
                    else:
                        rebalance_log.append({'date': dt, 'reason': 'too_few_candidates',
                                              'n_above_thresh': len(above_thresh),
                                              'n_long': 0, 'n_closed': 0, 'n_opened': 0})
                        #  fall through to single append below
                        long_ranks = None

                    if long_ranks is not None:
                        long_candidates = meta_today[
                            meta_today[TICKER_COL].isin(long_ranks.index)].copy()

                        if long_candidates.empty:
                            rebalance_log.append({'date': dt, 'reason': 'no_meta_match',
                                                  'n_long': 0, 'n_closed': 0, 'n_opened': 0})
                            #  fall through to single append below
                        else:
                            # Short candidates (Bear/HighVol regimes, F&O only)
                            short_candidates = pd.DataFrame()
                            if regime_int in SHORT_REGIMES:
                                short_ranks      = ranks[ranks <= SELL_THRESHOLD_ENH].nsmallest(SHORT_TOP_K_ENH)
                                short_candidates = meta_today[
                                    meta_today[TICKER_COL].isin(short_ranks.index) &
                                    meta_today[TICKER_COL].str.replace('.NS', '', regex=False).isin(FO_TICKERS)
                                ].copy()

                            # ATR from recent prices
                            recent_prices = daily_ret_pivot.loc[:dt]
                            atr_series    = compute_atr_from_prices(recent_prices.tail(20), window=14)

                            NAV             = cash + long_val + short_val
                            long_target_pct = max(deploy_cap - CASH_FLOOR_ENH, 0.0)

                            long_sizes = compute_position_sizes(
                                candidates=long_candidates, direction='long', nav=NAV,
                                book_target_pct=long_target_pct, deploy_cap=deploy_cap,
                                atr_series=atr_series,
                            )
                            long_sizes = apply_sector_cap(long_candidates, long_sizes)

                            short_sizes = pd.Series(dtype=float)
                            if not short_candidates.empty:
                                short_book_cap = min(MAX_SHORT_BOOK_ENH, deploy_cap * 0.10)
                                short_sizes    = compute_position_sizes(
                                    candidates=short_candidates, direction='short', nav=NAV,
                                    book_target_pct=short_book_cap, deploy_cap=deploy_cap,
                                    atr_series=atr_series,
                                )

                            # ──  partial rebalance ───────────────
                            # Step A: close positions no longer in the signal set
                            target_long_tickers  = set(long_sizes.index)
                            target_short_tickers = set(short_sizes.index)
                            n_closed = 0

                            for ticker in list(positions.keys()):
                                pos = positions[ticker]
                                if pos['stopped']:
                                    del positions[ticker]
                                    continue
                                if pos['direction'] == 'long' and ticker not in target_long_tickers:
                                    # Signal gone — exit long
                                    cash += (pos['alloc'] + pos['pnl']) * (1 - COST_RATE_ENH)
                                    del positions[ticker]
                                    n_closed += 1
                                elif pos['direction'] == 'short' and ticker not in target_short_tickers:
                                    # Signal gone — exit short
                                    # AUDIT FIX 3: short close = alloc - pnl
                                    cash += (pos['alloc'] - pos['pnl']) * (1 - COST_RATE_ENH)
                                    del positions[ticker]
                                    n_closed += 1

                            # Step B: open only NEW long positions not already held
                            budget   = NAV * long_target_pct
                            n_opened = 0

                            for ticker, size_pct in long_sizes.items():
                                if ticker in positions:
                                    continue           # already held — keep it, skip cost
                                alloc = NAV * size_pct
                                cost  = alloc * COST_RATE_ENH
                                if alloc < NAV * 0.003:      continue  # below floor
                                if budget < alloc + cost:    continue  # over budget
                                cash   -= (alloc + cost)
                                budget -= (alloc + cost)
                                positions[ticker] = {
                                    'alloc': alloc, 'pnl': 0.0, 'high_water': alloc,
                                    'entry_date': dt, 'direction': 'long', 'stopped': False
                                }
                                n_opened += 1

                            # Step C: open only NEW short positions not already held
                            for ticker, size_pct in short_sizes.items():
                                if ticker in positions:
                                    continue           # already held — keep it
                                alloc = NAV * size_pct
                                if alloc < NAV * 0.003: continue
                                total_short = sum(p['alloc'] for p in positions.values()
                                                  if p['direction'] == 'short' and not p['stopped'])
                                if total_short + alloc > NAV * MAX_SHORT_BOOK_ENH: break
                                cash -= alloc * COST_RATE_ENH
                                positions[ticker] = {
                                    'alloc': alloc, 'pnl': 0.0, 'high_water': alloc,
                                    'entry_date': dt, 'direction': 'short', 'stopped': False
                                }

                            rebalance_log.append({
                                'date': dt, 'signal_dt': signal_dt, 'regime': regime_int,
                                'deploy_cap': round(deploy_cap, 3),
                                'n_long_candidates': len(long_candidates),
                                'n_long_opened': n_opened,
                                'n_closed': n_closed,
                                'n_above_thresh': len(above_thresh),
                                'reason': 'ok'
                            })

        # AUDIT FIX 2: single NAV append — ONE record per day, no duplicates
        long_val  = sum(p['alloc'] + p['pnl'] for p in positions.values()
                        if not p['stopped'] and p['direction'] == 'long')
        short_val = sum(p['alloc'] - p['pnl'] for p in positions.values()
                        if not p['stopped'] and p['direction'] == 'short')
        nav_series.append({'Date': dt, 'NAV': max(cash + long_val + short_val, 0)})

    # ── Rebalance diagnostic summary ─────────────────────────────────────────
    rlog_df = pd.DataFrame(rebalance_log)
    print(f"\nRebalance log summary ({len(rlog_df)} rebalances):")
    if not rlog_df.empty:
        print(rlog_df['reason'].value_counts().to_string())
        ok_rebs = rlog_df[rlog_df['reason'] == 'ok']
        if not ok_rebs.empty:
            print(f"  Avg positions opened per rebalance : {ok_rebs['n_long_opened'].mean():.1f}")
            print(f"  Avg positions closed per rebalance : {ok_rebs['n_closed'].mean():.1f}")
            print(f"  Avg candidates above threshold     : {ok_rebs['n_above_thresh'].mean():.1f}")

    # ── Sanity check: verify no duplicate dates in NAV series ────────────────
    nav_df = pd.DataFrame(nav_series).set_index('Date')
    dupes  = nav_df.index.duplicated().sum()
    if dupes > 0:
        print(f"⚠️  WARNING: {dupes} duplicate dates found in NAV series — investigate!")
    else:
        print(f"✅  NAV series clean: {len(nav_df)} unique dates, no duplicates.")

    return nav_df

print("run_backtest() defined.")


# =============================================================================
# BLOCK G — RUN BACKTEST + RESULTS
# =============================================================================
nav_df = run_backtest(STARTING_NAV, daily_ret_pivot)
print(f"Complete : {len(nav_df)} days  |  "
      f"End NAV ₹{nav_df['NAV'].iloc[-1]:,.0f}  |  "
      f"Return {(nav_df['NAV'].iloc[-1]/STARTING_NAV-1)*100:.2f}%")

metrics, daily_rets, drawdown = compute_metrics(nav_df)

print("\n" + "="*55)
print("BACKTEST PERFORMANCE")
print("="*55)
for k, v in metrics.items(): print(f"  {k:<20} : {v}")
print("="*55)

nav_df['Year'] = nav_df.index.year
print("\nYear-by-year:")
for yr, grp in nav_df.groupby('Year'):
    if len(grp) < 2: continue
    print(f"  {yr}: {(grp['NAV'].iloc[-1]/grp['NAV'].iloc[0]-1)*100:+.2f}%")

try:
    cagr = float(metrics.get('CAGR','0%').strip('%')) / 100
    print(f"\n{'✅ TARGET MET' if cagr >= 0.12 else '⚠️  Below target'}: "
          f"CAGR {cagr*100:.1f}%  (target ≥ 12%)")
    if cagr < 0.12:
        print("   → Set ENABLE_TUNING = True below to run parameter grid search.")
except ValueError:
    pass

nav_df.reset_index().to_csv(
    os.path.join(RESULTS_DIR, f'nav_series_{DATE_STAMP}.csv'), index=False)
with open(os.path.join(RESULTS_DIR, f'metrics_{DATE_STAMP}.txt'), 'w', encoding='utf-8') as f:
    f.write("AI Hedge Fund Simulator v2 — Backtest\n")
    f.write("Changes: bi-weekly rebalance, TOP_K=20, BUY=0.85, cash_floor=2%, "
            "bull_deploy=99%, DD_defence=-12%/-20%, momentum_topup=1.25x.\n")
    f.write("="*55 + "\n")
    for k, v in metrics.items(): f.write(f"  {k:<20} : {v}\n")
print(f"Saved results → {RESULTS_DIR}")

# ── Optional tuning grid ──────────────────────────────────────────────────────
ENABLE_TUNING = False   # ← flip True if CAGR still below 12%

if ENABLE_TUNING:
    print("\nRunning parameter tuning grid ...")
    best_cagr, best_params, results_grid = -999.0, {}, []

    for bt, ltk, ivw in itertools.product(
        [0.82, 0.83, 0.85, 0.87],   # buy threshold
        [15, 20, 25],                # top K
        [0.50, 0.60, 0.70],          # inv vol weight
    ):
        BUY_THRESHOLD_ENH = bt
        LONG_TOP_K        = ltk
        INV_VOL_WEIGHT    = ivw
        CONVICTION_WEIGHT = 1.0 - ivw
        try:
            nav_tmp     = run_backtest(STARTING_NAV, daily_ret_pivot)
            m_tmp,_,_   = compute_metrics(nav_tmp)
            cagr_tmp    = float(m_tmp['CAGR'].strip('%')) / 100
            sharpe_tmp  = float(m_tmp['Sharpe Ratio'])
            results_grid.append({'BUY': bt,'TOP_K': ltk,'INV_VOL': ivw,
                                  'CAGR': cagr_tmp,'Sharpe': sharpe_tmp})
            if cagr_tmp > best_cagr:
                best_cagr   = cagr_tmp
                best_params = {'BUY_THRESHOLD_ENH': bt,'LONG_TOP_K': ltk,'INV_VOL_WEIGHT': ivw}
                print(f"  New best CAGR={cagr_tmp*100:.1f}%  Sharpe={sharpe_tmp:.3f}  {best_params}")
        except Exception as e:
            print(f"  Error ({bt},{ltk},{ivw}): {e}")

    print(pd.DataFrame(results_grid).sort_values('CAGR',ascending=False).head(5).to_string(index=False))

    BUY_THRESHOLD_ENH = best_params.get('BUY_THRESHOLD_ENH', 0.85)
    LONG_TOP_K        = best_params.get('LONG_TOP_K', 20)
    INV_VOL_WEIGHT    = best_params.get('INV_VOL_WEIGHT', 0.60)
    CONVICTION_WEIGHT = 1.0 - INV_VOL_WEIGHT
    print(f"\nBest params restored. Re-running backtest ...")
    nav_best     = run_backtest(STARTING_NAV, daily_ret_pivot)
    m_best,_,_   = compute_metrics(nav_best)
    print("\n" + "="*55 + "\nTUNED BACKTEST PERFORMANCE\n" + "="*55)
    for k, v in m_best.items(): print(f"  {k:<20} : {v}")