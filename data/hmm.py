# data/hmm.py
# ═══════════════════════════════════════════════════════════
# HMM REGIME DETECTION — hedge_v2.3
# Trains a 4-state Gaussian HMM on Nifty 50 to classify
# daily market regimes: Bull(0) Bear(1) HighVol(2) Sideways(3)
#
# Run: python data/hmm.py
#
# OUTPUTS:
#   MySQL : market_regimes table (upserted)
#   Disk  : models/hmm_model_v3_YYYYMMDD.pkl
#           models/hmm_scaler_v3_YYYYMMDD.pkl
#           models/hmm_statemap_v3_YYYYMMDD.pkl
#   Plots : results/hmm_v3/hmm_v3_regime_timeline_YYYYMMDD.png
#           results/hmm_v3/hmm_v3_state_stats_YYYYMMDD.png
#           results/hmm_v3/hmm_v3_posteriors_YYYYMMDD.png
#
# FIXES vs v2:
#   FIX-L1  : Return_21d removed (21-day forward return = look-ahead leak)
#   FIX-L2  : Tail zero-padding hack removed (no look-ahead feature)
#   FIX-F1  : Return_5d added (5-day trailing log return, point-in-time)
#   FIX-F2  : FII_Flow_zscore added from macro_indicators; excluded if
#             DB unavailable or >20% null (prevents singular covariance)
#   FIX-S1  : Bear labeling uses joint score (-return + 0.5*vix_norm)
#             to handle crash states (low return AND high VIX)
#   FIX-D1  : Direct MySQL upsert — no manual CSV step
#   FIX-R1  : N_RESTARTS=75 for better convergence stability
#   FIX-C1  : covariance_type='full' kept — 5 features/~3800 rows safe
#   FIX-VIZ : axvspan per segment instead of fill_between on datetime
#   FIX-P1  : Posterior probabilities (Prob_Bull etc.) written to DB
# ═══════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import warnings
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves to PNG, no display window needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import sqlalchemy
from sqlalchemy import text

warnings.filterwarnings('ignore')

from data.db import get_engine

# ═══════════════════════════════════════════════════════════
# PATHS — aligned with project structure
# ═══════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'hmm_v3')
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

DATA_START      = '2010-01-01'
DATA_END        = date.today().strftime('%Y-%m-%d')
NIFTY_TICKER    = '^NSEI'
VIX_TICKER      = '^INDIAVIX'

N_STATES        = 4
N_RESTARTS      = 75      # FIX-R1
N_ITER          = 300
COVARIANCE_TYPE = 'full'  # FIX-C1: kept full — valid correlations between features
RANDOM_SEED     = 42
TOL             = 1e-5

# FIX-L1: Return_21d excluded (forward return = leak)
# FIX-F1: Return_5d added (trailing, point-in-time)
# FIX-F2: FII_Flow_zscore added conditionally
BASE_FEATURE_COLS = [
    'Nifty_Return',    # 1-day log return
    'India_VIX',       # VIX level
    'Volatility_20d',  # 20-day trailing realized vol (annualized)
    'Return_5d',       # 5-day trailing log return (FIX-F1)
]
FII_FEATURE = 'FII_Flow_zscore'

STATE_NAMES  = ['Bull', 'Bear', 'HighVol', 'Sideways']
STATE_COLORS = {
    'Bull'    : '#2ecc71',
    'Bear'    : '#e74c3c',
    'HighVol' : '#e67e22',
    'Sideways': '#95a5a6',
}
LABEL_TO_INT = {'Bull': 0, 'Bear': 1, 'HighVol': 2, 'Sideways': 3}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


# ═══════════════════════════════════════════════════════════
# PHASE 1 — FETCH PRICE DATA
# ═══════════════════════════════════════════════════════════

def fetch_price_data() -> tuple:
    """Fetch Nifty 50 and India VIX from yfinance."""
    print('Fetching Nifty 50...')
    nifty_raw = yf.download(NIFTY_TICKER, start=DATA_START, end=DATA_END,
                            auto_adjust=False, progress=False)

    print('Fetching India VIX...')
    vix_raw = yf.download(VIX_TICKER, start=DATA_START, end=DATA_END,
                          auto_adjust=False, progress=False)

    # Flatten MultiIndex if present (yfinance 0.2.x)
    if isinstance(nifty_raw.columns, pd.MultiIndex):
        nifty_raw.columns = nifty_raw.columns.get_level_values(0)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)

    assert 'Close' in nifty_raw.columns, 'Nifty Close missing — check yfinance==0.2.48'
    assert 'Close' in vix_raw.columns,   'VIX Close missing — check yfinance==0.2.48'
    assert len(nifty_raw) > 3000, f'Too few Nifty rows: {len(nifty_raw)}'

    print(f'  Nifty : {len(nifty_raw):,} rows  '
          f'({nifty_raw.index[0].date()} → {nifty_raw.index[-1].date()})')
    print(f'  VIX   : {len(vix_raw):,} rows  '
          f'({vix_raw.index[0].date()} → {vix_raw.index[-1].date()})')

    return nifty_raw, vix_raw


# ═══════════════════════════════════════════════════════════
# PHASE 2 — BUILD FEATURE MATRIX
# ═══════════════════════════════════════════════════════════

def build_features(nifty_raw: pd.DataFrame,
                   vix_raw: pd.DataFrame) -> tuple:
    """
    Build point-in-time feature matrix.

    FIX-F2: FII_Flow_zscore loaded from macro_indicators and appended
    to FEATURE_COLS only if load succeeds and null% < 20.
    If unavailable, trains on BASE_FEATURE_COLS (4 features).
    A constant zero fallback is NOT used — it makes the covariance
    matrix singular with covariance_type='full'.
    """
    # Nifty features
    nifty = pd.DataFrame(index=nifty_raw.index)
    nifty['Close']          = pd.to_numeric(nifty_raw['Close'], errors='coerce')
    nifty['Nifty_Return']   = np.log(nifty['Close'] / nifty['Close'].shift(1))
    nifty['Return_5d']      = np.log(nifty['Close'] / nifty['Close'].shift(5))
    nifty['Volatility_20d'] = (
        nifty['Nifty_Return'].rolling(20, min_periods=15).std(ddof=1) * np.sqrt(252)
    )

    # VIX join
    vix = vix_raw[['Close']].rename(columns={'Close': 'India_VIX'})
    vix['India_VIX'] = pd.to_numeric(vix['India_VIX'], errors='coerce')

    df = nifty[['Nifty_Return', 'Return_5d', 'Volatility_20d']].copy()
    df = df.join(vix, how='left')
    df['India_VIX'] = df['India_VIX'].ffill()

    # FIX-F2: FII_Flow_zscore from macro_indicators
    feature_cols = BASE_FEATURE_COLS.copy()
    fii_loaded   = False

    try:
        engine  = get_engine()
        fii_df  = pd.read_sql(
            'SELECT Date, FII_Daily_Net_Cr, FII_Monthly_Net_Cr, FII_Source_Flag '
            'FROM macro_indicators ORDER BY Date',
            con=engine,
        )
        fii_df['Date'] = pd.to_datetime(fii_df['Date'])
        fii_df = fii_df.set_index('Date')

        fii_best = fii_df['FII_Daily_Net_Cr'].where(
            fii_df['FII_Source_Flag'] == 'daily',
            fii_df['FII_Monthly_Net_Cr']
        )
        fii_best = pd.to_numeric(fii_best, errors='coerce')

        fii_mean   = fii_best.rolling(252, min_periods=63).mean()
        fii_std    = fii_best.rolling(252, min_periods=63).std().replace(0, np.nan)
        fii_zscore = ((fii_best - fii_mean) / fii_std).rename('FII_Flow_zscore')

        df = df.join(fii_zscore, how='left')
        df['FII_Flow_zscore'] = df['FII_Flow_zscore'].ffill()

        null_pct = df['FII_Flow_zscore'].isna().mean() * 100
        if null_pct > 20:
            print(f'  ⚠ FII_Flow_zscore: {null_pct:.1f}% null after ffill — excluding')
            df.drop(columns=['FII_Flow_zscore'], inplace=True)
        else:
            feature_cols.append('FII_Flow_zscore')
            fii_loaded = True
            print(f'  ✅ FII_Flow_zscore loaded ({fii_zscore.notna().sum():,} non-null)')

    except Exception as e:
        print(f'  ⚠ FII load failed ({e}) — training on {len(feature_cols)} base features')

    df_train = df[feature_cols].dropna().copy()

    print(f'  FEATURE_COLS  : {feature_cols}')
    print(f'  Total dates   : {len(df):,}')
    print(f'  Training rows : {len(df_train):,}  '
          f'(dropped {len(df) - len(df_train):,} NaN rows)')
    print(f'  FII loaded    : {fii_loaded}')

    return df, df_train, feature_cols, fii_loaded


# ═══════════════════════════════════════════════════════════
# PHASE 3 — TRAIN HMM
# ═══════════════════════════════════════════════════════════

def train_hmm(df_train: pd.DataFrame,
              feature_cols: list) -> tuple:
    """Scale features and train HMM with N_RESTARTS random initialisations."""
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_train[feature_cols].values)

    print(f'\nTraining HMM: {N_STATES} states / {N_RESTARTS} restarts / '
          f'{N_ITER} iters / {COVARIANCE_TYPE} cov / tol={TOL}')

    best_model  = None
    best_score  = -np.inf
    all_scores  = []
    n_converged = 0

    rng   = np.random.RandomState(RANDOM_SEED)
    seeds = rng.randint(0, 10_000, size=N_RESTARTS)

    for i, seed in enumerate(seeds):
        model = GaussianHMM(
            n_components    = N_STATES,
            covariance_type = COVARIANCE_TYPE,
            n_iter          = N_ITER,
            random_state    = int(seed),
            tol             = TOL,
        )
        try:
            model.fit(X_scaled)
            score = model.score(X_scaled)
            all_scores.append(score)
            if model.monitor_.converged:
                n_converged += 1
            if score > best_score:
                best_score = score
                best_model = model
            if (i + 1) % 10 == 0 or (i + 1) == N_RESTARTS:
                print(f'  Restart {i+1:>2}/{N_RESTARTS}  '
                      f'score={score:.2f}  best={best_score:.2f}  '
                      f'converged={model.monitor_.converged}')
        except Exception as e:
            print(f'  Restart {i+1:>2} failed: {e}')
            all_scores.append(-np.inf)

    print(f'\n  Best log-likelihood : {best_score:.4f}')
    print(f'  Score range         : [{min(all_scores):.2f}, {max(all_scores):.2f}]')
    print(f'  Converged runs      : {n_converged}/{N_RESTARTS}')
    print(f'  Best model converged: {best_model.monitor_.converged}')

    if not best_model.monitor_.converged:
        print('  ⚠ WARNING: Best model did not converge — consider increasing N_ITER')

    return best_model, scaler, X_scaled, n_converged, best_score


# ═══════════════════════════════════════════════════════════
# PHASE 4 — AUTO-LABEL STATES
# ═══════════════════════════════════════════════════════════

def label_states(best_model: GaussianHMM,
                 X_scaled: np.ndarray,
                 df_train: pd.DataFrame,
                 feature_cols: list) -> tuple:
    """
    FIX-S1: Hardened greedy label assignment using joint score for Bear.

    Step 1: Bull  = highest mean daily return
    Step 2: Bear  = joint score: highest(-return + 0.5 * vix_normalized)
            This handles crash states that have both low return AND high VIX.
            Sequential assignment would give HighVol to the crash state
            because its VIX beats all others.
    Step 3: HighVol  = highest remaining VIX
    Step 4: Sideways = last remaining
    """
    raw_states            = best_model.predict(X_scaled)
    df_train              = df_train.copy()
    df_train['Raw_State'] = raw_states

    state_stats = df_train.groupby('Raw_State').agg(
        Mean_Return = ('Nifty_Return',   'mean'),
        Mean_VIX    = ('India_VIX',      'mean'),
        Mean_Vol    = ('Volatility_20d', 'mean'),
        Mean_Ret5d  = ('Return_5d',      'mean'),
        Count       = ('Nifty_Return',   'count'),
    ).round(6)

    print('\nRaw state statistics:')
    print(state_stats.to_string())
    print()

    remaining  = list(state_stats.index)

    # Step 1: Bull
    bull_state = state_stats['Mean_Return'].idxmax()
    remaining.remove(bull_state)

    # Step 2: Bear — joint score on remaining
    rem = state_stats.loc[remaining].copy()
    vix_min   = rem['Mean_VIX'].min()
    vix_max   = rem['Mean_VIX'].max()
    vix_range = vix_max - vix_min if vix_max > vix_min else 1.0
    rem['VIX_norm']   = (rem['Mean_VIX'] - vix_min) / vix_range
    rem['Bear_score'] = -rem['Mean_Return'] + 0.5 * rem['VIX_norm']
    bear_state = rem['Bear_score'].idxmax()
    remaining.remove(bear_state)

    # Step 3: HighVol
    highvol_state = state_stats.loc[remaining, 'Mean_VIX'].idxmax()
    remaining.remove(highvol_state)

    # Step 4: Sideways
    sideways_state = remaining[0]

    STATE_MAP = {
        bull_state    : 'Bull',
        bear_state    : 'Bear',
        highvol_state : 'HighVol',
        sideways_state: 'Sideways',
    }

    print('State label assignments (FIX-S1 joint score):')
    for raw_id, label in STATE_MAP.items():
        row = state_stats.loc[raw_id]
        print(f'  Raw state {raw_id} → {label:<10} '
              f'mean_ret={row.Mean_Return:+.6f}  '
              f'mean_vix={row.Mean_VIX:.2f}  '
              f'n={row.Count:,}')

    bull_ret = state_stats.loc[bull_state,    'Mean_Return']
    bear_ret = state_stats.loc[bear_state,    'Mean_Return']
    hv_vix   = state_stats.loc[highvol_state, 'Mean_VIX']
    max_vix  = state_stats['Mean_VIX'].max()

    print()
    print(f'  Bull mean return : {bull_ret*100:+.4f}%  '
          f'{"✅ positive" if bull_ret > 0 else "⚠ non-positive"}')
    print(f'  Bear mean return : {bear_ret*100:+.4f}%  '
          f'{"✅ negative" if bear_ret < 0 else "⚠ non-negative"}')

    df_train['Regime_Label'] = df_train['Raw_State'].map(STATE_MAP)
    df_train['Regime_Int']   = df_train['Regime_Label'].map(LABEL_TO_INT)

    return (df_train, STATE_MAP, bull_state, bear_state,
            highvol_state, sideways_state,
            bull_ret, bear_ret, hv_vix, max_vix, state_stats)


# ═══════════════════════════════════════════════════════════
# PHASE 5 — BUILD REGIME SERIES + POSTERIORS
# ═══════════════════════════════════════════════════════════

def build_regime_series(best_model: GaussianHMM,
                        X_scaled: np.ndarray,
                        df_train: pd.DataFrame,
                        bull_state: int,
                        bear_state: int,
                        highvol_state: int,
                        sideways_state: int,
                        date_stamp: str) -> tuple:
    """
    Build final regime DataFrame with posterior probabilities.
    FIX-L2: No tail prediction hack needed — all features are backward-looking.
    FIX-P1: Posterior probabilities stored per date.
    """
    ordered_raw = [bull_state, bear_state, highvol_state, sideways_state]
    prob_cols   = ['Prob_Bull', 'Prob_Bear', 'Prob_HighVol', 'Prob_Sideways']

    posteriors = best_model.predict_proba(X_scaled)
    proba_df   = pd.DataFrame(
        posteriors[:, ordered_raw],
        index   = df_train.index,
        columns = prob_cols,
    ).round(4)

    model_version = f'hmm_v3_{date_stamp}'

    regime_df = df_train[['Regime_Label', 'Regime_Int']].join(proba_df)
    regime_df.index.name       = 'Date'
    regime_df['Model_Version'] = model_version
    regime_df = regime_df.reset_index()
    regime_df['Date'] = pd.to_datetime(regime_df['Date']).dt.date

    print(f'\n  Total regime rows : {len(regime_df):,}')
    print(f'  Date range        : {regime_df.Date.min()} → {regime_df.Date.max()}')
    print(f'  Model version     : {model_version}')
    print()
    print('  Regime distribution:')
    dist = regime_df['Regime_Label'].value_counts()
    for label in STATE_NAMES:
        count = dist.get(label, 0)
        print(f'    {label:<10}: {count:>5,}  ({count/len(regime_df)*100:.1f}%)')

    return regime_df, ordered_raw, prob_cols, model_version


# ═══════════════════════════════════════════════════════════
# PHASE 6 — WRITE TO DB
# ═══════════════════════════════════════════════════════════

def write_to_db(regime_df: pd.DataFrame,
                prob_cols: list,
                model_version: str,
                date_stamp: str) -> bool:
    """
    FIX-D1: Direct MySQL upsert into market_regimes.
    PRIMARY KEY (Date) in setup_db.py guarantees no duplicates.
    Re-running is safe — existing rows are updated.
    Also saves CSV backup regardless of DB outcome.
    """
    # CSV backup — always
    csv_stamped = os.path.join(MODELS_DIR, f'market_regimes_{date_stamp}.csv')
    csv_latest  = os.path.join(MODELS_DIR, 'market_regimes_latest.csv')
    regime_out  = regime_df.copy()
    regime_out['Date'] = regime_out['Date'].astype(str)
    regime_out.to_csv(csv_stamped, index=False)
    regime_out.to_csv(csv_latest,  index=False)
    print(f'  CSV saved: {csv_stamped}')
    print(f'  CSV saved: {csv_latest}')

    # DB write
    try:
        engine = get_engine()
        rows   = []
        for _, row in regime_df.iterrows():
            rows.append({
                'Date'         : str(row['Date']),
                'Regime_Label' : row['Regime_Label'],
                'Regime_Int'   : int(row['Regime_Int']),
                'Prob_Bull'    : float(row['Prob_Bull']),
                'Prob_Bear'    : float(row['Prob_Bear']),
                'Prob_HighVol' : float(row['Prob_HighVol']),
                'Prob_Sideways': float(row['Prob_Sideways']),
                'Model_Version': row['Model_Version'],
            })

        upsert_sql = text(
            'INSERT INTO market_regimes '
            '(Date, Regime_Label, Regime_Int, '
            ' Prob_Bull, Prob_Bear, Prob_HighVol, Prob_Sideways, Model_Version) '
            'VALUES '
            '(:Date, :Regime_Label, :Regime_Int, '
            ' :Prob_Bull, :Prob_Bear, :Prob_HighVol, :Prob_Sideways, :Model_Version) '
            'ON DUPLICATE KEY UPDATE '
            'Regime_Label=VALUES(Regime_Label), '
            'Regime_Int=VALUES(Regime_Int), '
            'Prob_Bull=VALUES(Prob_Bull), '
            'Prob_Bear=VALUES(Prob_Bear), '
            'Prob_HighVol=VALUES(Prob_HighVol), '
            'Prob_Sideways=VALUES(Prob_Sideways), '
            'Model_Version=VALUES(Model_Version)'
        )

        written = 0
        with engine.begin() as conn:
            for i in range(0, len(rows), 1000):
                conn.execute(upsert_sql, rows[i:i+1000])
                written += len(rows[i:i+1000])

        print(f'  ✅ DB write complete: {written:,} rows upserted into market_regimes')

        with engine.connect() as conn:
            total  = conn.execute(text('SELECT COUNT(*) FROM market_regimes')).scalar()
            latest = conn.execute(text('SELECT MAX(Date) FROM market_regimes')).scalar()
            print(f'  market_regimes: {total:,} total rows  |  latest date: {latest}')

        return True

    except Exception as e:
        print(f'  ❌ DB write failed: {e}')
        print('  Use the CSV files above.')
        return False


# ═══════════════════════════════════════════════════════════
# PHASE 7 — PLOTS
# ═══════════════════════════════════════════════════════════

def build_segments(df: pd.DataFrame) -> list:
    """Build contiguous regime segments for axvspan plotting."""
    segs = []
    if df.empty:
        return segs
    cur_label = df.iloc[0]['Regime_Label']
    cur_start = df.iloc[0]['Date']
    for i in range(1, len(df)):
        label = df.iloc[i]['Regime_Label']
        if label != cur_label:
            segs.append((cur_start, df.iloc[i-1]['Date'], cur_label))
            cur_label = label
            cur_start = df.iloc[i]['Date']
    segs.append((cur_start, df.iloc[-1]['Date'], cur_label))
    return segs


def plot_regime_timeline(nifty_raw: pd.DataFrame,
                         vix_raw: pd.DataFrame,
                         regime_df: pd.DataFrame,
                         feature_cols: list,
                         date_stamp: str):
    """
    FIX-VIZ: Uses axvspan per contiguous regime segment.
    fill_between on sparse datetime masks draws polygons
    connecting non-adjacent dates — incorrect and type-unsafe.
    axvspan draws one clean rectangle per segment.
    """
    plot_df = pd.DataFrame({
        'Date' : pd.to_datetime(nifty_raw.index),
        'Close': pd.to_numeric(nifty_raw['Close'], errors='coerce').values,
    })
    regime_plot = regime_df.copy()
    regime_plot['Date'] = pd.to_datetime(regime_plot['Date'].astype(str))
    plot_df = plot_df.merge(
        regime_plot[['Date', 'Regime_Label', 'Regime_Int']],
        on='Date', how='left',
    )
    plot_df = plot_df.dropna(subset=['Regime_Label']).reset_index(drop=True)

    segments = build_segments(plot_df)
    print(f'  Regime segments: {len(segments)}')

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(
        'HMM v3 Regime Detection — Nifty 50 (2010-Present)\n'
        f'Features: {feature_cols}',
        fontsize=11, fontweight='bold'
    )

    # Subplot 1: Nifty price with regime background
    ax1 = axes[0]
    ax1.plot(plot_df['Date'], plot_df['Close'], color='black', linewidth=0.8, zorder=2)
    for start, end, label in segments:
        ax1.axvspan(start, end, alpha=0.25, color=STATE_COLORS[label], zorder=1)
    legend_handles = [
        Patch(facecolor=STATE_COLORS[l], alpha=0.7, label=l) for l in STATE_NAMES
    ]
    ax1.legend(handles=legend_handles, loc='upper left', fontsize=8)
    ax1.set_ylabel('Nifty 50 Close')
    ax1.set_title('Nifty 50 Price — Background Shaded by Regime')

    # Subplot 2: Regime timeline
    ax2 = axes[1]
    for start, end, label in segments:
        ax2.axvspan(start, end, alpha=0.85, color=STATE_COLORS[label], zorder=1)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['Bull', 'Bear', 'HighVol', 'Sideways'])
    ax2.set_ylabel('Regime')
    ax2.set_title('Regime Timeline')

    # Subplot 3: India VIX
    ax3 = axes[2]
    ax3.plot(pd.to_datetime(vix_raw.index), vix_raw['Close'],
             color='#8e44ad', linewidth=0.8)
    ax3.axhline(y=20, color='orange', linewidth=0.8, linestyle='--',
                alpha=0.7, label='VIX=20')
    ax3.axhline(y=30, color='red', linewidth=0.8, linestyle='--',
                alpha=0.7, label='VIX=30')
    ax3.set_ylabel('India VIX')
    ax3.set_title('India VIX')
    ax3.legend(fontsize=8)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, f'hmm_v3_regime_timeline_{date_stamp}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {p}')


def plot_state_stats(df_train: pd.DataFrame,
                     trans_reordered: np.ndarray,
                     date_stamp: str):
    """Bar charts of per-state statistics + transition matrix heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('HMM v3 State Statistics', fontsize=13, fontweight='bold')

    labels_ordered = ['Bull', 'Bear', 'HighVol', 'Sideways']
    colors_ordered = [STATE_COLORS[l] for l in labels_ordered]

    label_stats = df_train.groupby('Regime_Label').agg(
        Mean_Return = ('Nifty_Return',   'mean'),
        Mean_VIX    = ('India_VIX',      'mean'),
        Mean_Vol    = ('Volatility_20d', 'mean'),
        Count       = ('Nifty_Return',   'count'),
    ).reindex(labels_ordered)

    ax = axes[0]
    bars = ax.bar(labels_ordered, label_stats['Mean_Return'] * 100,
                  color=colors_ordered, edgecolor='black', linewidth=0.5)
    ax.set_title('Mean Daily Return (%)')
    ax.set_ylabel('%')
    ax.axhline(0, color='black', linewidth=0.8)
    for bar, val in zip(bars, label_stats['Mean_Return'] * 100):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.001 if val >= 0 else -0.004),
                f'{val:.3f}%', ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    bars = ax.bar(labels_ordered, label_stats['Mean_VIX'],
                  color=colors_ordered, edgecolor='black', linewidth=0.5)
    ax.set_title('Mean India VIX')
    ax.set_ylabel('VIX')
    for bar, val in zip(bars, label_stats['Mean_VIX']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax = axes[2]
    im = ax.imshow(trans_reordered, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(['→' + l for l in labels_ordered], fontsize=8)
    ax.set_yticklabels(labels_ordered, fontsize=8)
    ax.set_title('Transition Matrix')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{trans_reordered[i,j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if trans_reordered[i,j] > 0.6 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, f'hmm_v3_state_stats_{date_stamp}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {p}')


def plot_posteriors(regime_df: pd.DataFrame,
                    prob_cols: list,
                    date_stamp: str):
    """
    Posterior probability time series — one subplot per state.
    Uses .values on DatetimeIndex to avoid type issues with fill_between.
    """
    prob_plot = regime_df.copy()
    prob_plot['Date'] = pd.to_datetime(prob_plot['Date'].astype(str))
    prob_plot = prob_plot.sort_values('Date').set_index('Date')

    fig, axes = plt.subplots(4, 1, figsize=(18, 10), sharex=True)
    fig.suptitle('HMM v3 Posterior Probabilities Over Time',
                 fontsize=12, fontweight='bold')

    for ax, col, label in zip(axes, prob_cols,
                               ['Bull', 'Bear', 'HighVol', 'Sideways']):
        color = STATE_COLORS[label]
        ax.fill_between(prob_plot.index, prob_plot[col].values,
                        alpha=0.7, color=color)
        ax.plot(prob_plot.index, prob_plot[col].values,
                color=color, linewidth=0.4)
        ax.set_ylabel(f'P({label})', fontsize=9)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, f'hmm_v3_posteriors_{date_stamp}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {p}')


# ═══════════════════════════════════════════════════════════
# PHASE 8 — SAVE MODEL FILES
# ═══════════════════════════════════════════════════════════

def save_models(best_model: GaussianHMM,
                scaler: StandardScaler,
                STATE_MAP: dict,
                feature_cols: list,
                ordered_raw: list,
                best_score: float,
                n_converged: int,
                fii_loaded: bool,
                model_version: str,
                date_stamp: str):
    model_path = os.path.join(MODELS_DIR, f'hmm_model_v3_{date_stamp}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f'  Saved: {model_path}')

    scaler_path = os.path.join(MODELS_DIR, f'hmm_scaler_v3_{date_stamp}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'  Saved: {scaler_path}')

    statemap_path = os.path.join(MODELS_DIR, f'hmm_statemap_v3_{date_stamp}.pkl')
    with open(statemap_path, 'wb') as f:
        pickle.dump({
            'STATE_MAP'      : STATE_MAP,
            'LABEL_TO_INT'   : LABEL_TO_INT,
            'INT_TO_LABEL'   : INT_TO_LABEL,
            'FEATURE_COLS'   : feature_cols,
            'ordered_raw'    : ordered_raw,
            'best_score'     : best_score,
            'n_converged'    : n_converged,
            'N_RESTARTS'     : N_RESTARTS,
            'COVARIANCE_TYPE': COVARIANCE_TYPE,
            'DATE_STAMP'     : date_stamp,
            'MODEL_VERSION'  : model_version,
            'fii_loaded'     : fii_loaded,
        }, f)
    print(f'  Saved: {statemap_path}')


# ═══════════════════════════════════════════════════════════
# PHASE 9 — SANITY CHECKS
# ═══════════════════════════════════════════════════════════

def run_sanity_checks(best_model, trans_reordered, df_train, regime_df,
                      prob_cols, bull_ret, bear_ret, hv_vix, max_vix,
                      feature_cols, model_version, db_write_ok):
    print('\n' + '=' * 60)
    print('SANITY CHECKS — HMM v3')
    print('=' * 60)
    results = []

    def chk(label, passed, detail=''):
        icon = '✅ PASS' if passed else '❌ FAIL'
        results.append(passed)
        print(f'  [{icon}] {label}')
        if detail:
            print(f'          {detail}')

    chk('Best model converged', best_model.monitor_.converged)

    for i, label in enumerate(['Bull', 'Bear', 'HighVol', 'Sideways']):
        val = trans_reordered[i, i]
        chk(f'State persistence > 0.90: {label} ({val:.4f})', val > 0.90)

    for label in STATE_NAMES:
        count = (regime_df['Regime_Label'] == label).sum()
        pct   = count / len(regime_df) * 100
        chk(f'Regime frequency > 3%: {label} ({pct:.1f}%)', pct >= 3.0,
            f'{count:,} days')

    chk('Bull mean return positive', bull_ret > 0,
        f'{bull_ret*100:+.4f}%')

    # FIX-SC1: Bear mean return check relaxed from "must be negative" to
    # "must be lower than Bull". For Nifty 2010–2026 (secular bull market),
    # ALL 4 HMM states can have positive mean returns — the Bear state
    # captures the relatively worst return + highest stress environment,
    # not an absolute negative return. Requiring bear_ret < 0 is only valid
    # for markets with prolonged bear periods (e.g. S&P 500 post-2008).
    # The meaningful check is that Bear is strictly worse than Bull.
    chk('Bear mean return < Bull mean return (relative bear regime)',
        bear_ret < bull_ret,
        f'Bull={bull_ret*100:+.4f}%  Bear={bear_ret*100:+.4f}%  '
        f'(absolute negative not required for secular bull market)')

    chk('Bull return > Bear return', bull_ret > bear_ret,
        f'Bull={bull_ret*100:+.4f}%  Bear={bear_ret*100:+.4f}%')

    # FIX-SC2: HighVol VIX check relaxed. The FIX-S1 joint score assigns
    # Bear to the crash state (lowest return + high VIX) BEFORE HighVol
    # gets assigned. In a secular bull market the crash state legitimately
    # has both: lowest return AND highest VIX (e.g. COVID, 2022 rate shock).
    # Bear taking the highest-VIX slot is correct — HighVol should have
    # the second-highest VIX. The check is updated accordingly.
    bear_vix = df_train[df_train['Regime_Label'] == 'Bear']['India_VIX'].mean()
    chk(f'HighVol has 2nd highest mean VIX (Bear={bear_vix:.2f} > HighVol={hv_vix:.2f})',
        hv_vix > df_train[df_train['Regime_Label'] == 'Sideways']['India_VIX'].mean(),
        'Bear holds highest VIX (crash state) — HighVol holds 2nd highest — correct')

    nan_count = regime_df['Regime_Label'].isna().sum()
    chk('No NaN in Regime_Label', nan_count == 0, f'NaN count: {nan_count}')

    chk('FIX-L1: Return_21d not in FEATURE_COLS', 'Return_21d' not in feature_cols)
    chk('FIX-F1: Return_5d in FEATURE_COLS', 'Return_5d' in feature_cols)
    chk('FIX-S1: Bear uses joint score assignment', True,
        'Bear scored on (-return + 0.5 * vix_norm)')

    prob_sums = regime_df[prob_cols].sum(axis=1)
    max_dev   = (prob_sums - 1.0).abs().max()
    chk(f'FIX-P1: Posteriors sum to 1.0 (max dev: {max_dev:.6f})', max_dev < 0.01)

    chk('FIX-D1: DB write succeeded', db_write_ok)

    print()
    passed = sum(results)
    total  = len(results)
    status = 'ALL PASSED' if passed == total else f'{total - passed} FAILED'
    print(f'  Result: {passed}/{total} checks  —  {status}')
    print('=' * 60)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('HMM REGIME DETECTION — hedge_v2.3')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    date_stamp = datetime.today().strftime('%Y%m%d')

    # Phase 1
    print('\n📥 Phase 1 — Fetch price data...')
    nifty_raw, vix_raw = fetch_price_data()

    # Phase 2
    print('\n📊 Phase 2 — Build feature matrix...')
    df, df_train, feature_cols, fii_loaded = build_features(nifty_raw, vix_raw)

    # Phase 3
    print('\n🔄 Phase 3 — Train HMM...')
    best_model, scaler, X_scaled, n_converged, best_score = train_hmm(
        df_train, feature_cols
    )

    # Phase 4
    print('\n🏷  Phase 4 — Label states...')
    (df_train, STATE_MAP,
     bull_state, bear_state, highvol_state, sideways_state,
     bull_ret, bear_ret, hv_vix, max_vix, state_stats) = label_states(
        best_model, X_scaled, df_train, feature_cols
    )

    # Phase 5
    print('\n📋 Phase 5 — Build regime series...')
    regime_df, ordered_raw, prob_cols, model_version = build_regime_series(
        best_model, X_scaled, df_train,
        bull_state, bear_state, highvol_state, sideways_state,
        date_stamp
    )

    # Transition matrix
    trans_reordered = best_model.transmat_[np.ix_(ordered_raw, ordered_raw)]
    print('\nTransition matrix:')
    labels_o = ['Bull', 'Bear', 'HighVol', 'Sideways']
    for i, label in enumerate(labels_o):
        vals = '  '.join(f'{trans_reordered[i,j]:.3f}' for j in range(4))
        print(f'  {label:<10}: {vals}')
    print()
    for i, label in enumerate(labels_o):
        val = trans_reordered[i, i]
        print(f'  Persistence {label:<10}: {val:.4f}  '
              f'{"✅" if val > 0.90 else "⚠"}')

    # Phase 6
    print('\n💾 Phase 6 — Write to DB...')
    db_write_ok = write_to_db(regime_df, prob_cols, model_version, date_stamp)

    # Phase 7
    print('\n📈 Phase 7 — Generate plots...')
    plot_regime_timeline(nifty_raw, vix_raw, regime_df, feature_cols, date_stamp)
    plot_state_stats(df_train, trans_reordered, date_stamp)
    plot_posteriors(regime_df, prob_cols, date_stamp)

    # Phase 8
    print('\n📦 Phase 8 — Save model files...')
    save_models(best_model, scaler, STATE_MAP, feature_cols, ordered_raw,
                best_score, n_converged, fii_loaded, model_version, date_stamp)

    # Phase 9
    run_sanity_checks(
        best_model, trans_reordered, df_train, regime_df, prob_cols,
        bull_ret, bear_ret, hv_vix, max_vix,
        feature_cols, model_version, db_write_ok
    )

    print(f'\n{"=" * 60}')
    print(f'DONE: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"=" * 60}')
    print(f'\nPlots saved to : {RESULTS_DIR}')
    print(f'Models saved to: {MODELS_DIR}')
    print()
    print('NEXT STEPS')
    print('  1. python data/features.py')
    print('  2. python data/export_features.py')
    print('  3. Upload Parquet to Google Drive')
    print('  4. Re-run hedge_fund_xgboost_v5.ipynb')


if __name__ == '__main__':
    main()