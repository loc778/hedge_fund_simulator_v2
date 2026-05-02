
# USAGE:
#   python data/export_features.py             # full export
#   python data/export_features.py --tier A    # Tier A only
#   python data/export_features.py --tier AB   # Tier A+B only

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import argparse
from datetime import datetime

from config import TABLES, FEATURES, PROJECT_ROOT
from data.db import get_engine

engine = get_engine()

EXPORT_DIR = os.path.join(PROJECT_ROOT, FEATURES["export_dir"])

# Tier filter mapping for CLI argument
TIER_FILTER = {
    "A"  : [1],
    "B"  : [2],
    "C"  : [3],
    "AB" : [1, 2],
    "ABC": [1, 2, 3],
}


def get_row_count(tier_filter: list | None) -> int:
    where = ""
    if tier_filter:
        tiers = ", ".join(str(t) for t in tier_filter)
        where = f"WHERE Data_Tier IN ({tiers})"
    with engine.connect() as conn:
        from sqlalchemy import text
        count = conn.execute(
            text(f"SELECT COUNT(*) FROM {TABLES['features']} {where}")
        ).scalar()
    return count


def load_features(tier_filter: list | None) -> pd.DataFrame:
    where = ""
    params = {}
    if tier_filter:
        tiers = ", ".join(str(t) for t in tier_filter)
        where = f"WHERE Data_Tier IN ({tiers})"

    query = f"SELECT * FROM {TABLES['features']} {where} ORDER BY Ticker, Date"

    print(f"  Loading from features_master{' (filtered)' if where else ''}...")
    chunks = []
    chunk_size = 100_000

    for chunk in pd.read_sql(query, con=engine, chunksize=chunk_size):
        # Drop DB internal column
        chunk = chunk.drop(columns=["id"], errors="ignore")

        # Ensure correct dtypes for Parquet compatibility
        chunk["Date"] = pd.to_datetime(chunk["Date"])

        # Nullable integer columns: Target_Direction_Median/Tertile, flags
        int8_nullable = [
            "Price_Gap_Flag", "SMA200_Available", "SMA50_Available",
            "ADX14_Available", "Volatility20d_Available", "Sentiment_Available",
            "Fundamentals_Available", "Data_Tier",
            "Target_Direction_Median", "Target_Direction_Tertile",
        ]
        for col in int8_nullable:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("Int8")  # nullable int (pd.NA for NULLs)

        # FII_Source_Flag: keep as string (already VARCHAR in DB)
        if "FII_Source_Flag" in chunk.columns:
            chunk["FII_Source_Flag"] = chunk["FII_Source_Flag"].astype("string")

        chunks.append(chunk)
        rows_so_far = sum(len(c) for c in chunks)
        print(f"    loaded {rows_so_far:,} rows...", end="\r")

    df = pd.concat(chunks, ignore_index=True)
    print(f"  ✅ Loaded {len(df):,} rows × {len(df.columns)} columns          ")
    return df


def export_parquet(df: pd.DataFrame, tier_label: str) -> tuple[str, str]:
    """
    Write DataFrame to parquet. Returns (timestamped_path, latest_path).
    Uses snappy compression (fast read/write, reasonable size).
    """
    os.makedirs(EXPORT_DIR, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_tier{tier_label}" if tier_label != "ABC" else ""

    timestamped_name = f"features_master{suffix}_{ts}.parquet"
    latest_name      = f"features_master{suffix}_latest.parquet"

    timestamped_path = os.path.join(EXPORT_DIR, timestamped_name)
    latest_path      = os.path.join(EXPORT_DIR, latest_name)

    print(f"  Writing {timestamped_name}...")
    df.to_parquet(timestamped_path, engine="pyarrow", compression="snappy", index=False)

    # Overwrite _latest (always points to most recent export)
    if os.path.exists(latest_path):
        os.remove(latest_path)
    df.to_parquet(latest_path, engine="pyarrow", compression="snappy", index=False)

    return timestamped_path, latest_path


def print_summary(df: pd.DataFrame, timestamped_path: str, latest_path: str):
    file_size_mb = os.path.getsize(latest_path) / (1024 ** 2)

    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"  Rows          : {len(df):,}")
    print(f"  Columns       : {len(df.columns)}")
    print(f"  Date range    : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  Tickers       : {df['Ticker'].nunique()}")
    print(f"\n  Tier breakdown:")

    tier_names = {1: "A", 2: "B", 3: "C"}
    for tier_num, tier_name in tier_names.items():
        n = (df["Data_Tier"] == tier_num).sum()
        if n > 0:
            tickers = df.loc[df["Data_Tier"] == tier_num, "Ticker"].nunique()
            print(f"    Tier {tier_name}: {n:>10,} rows | {tickers:>4} tickers")

    print(f"\n  File size     : {file_size_mb:.1f} MB")
    print(f"  Timestamped   : {timestamped_path}")
    print(f"  Latest        : {latest_path}")

def main():
    parser = argparse.ArgumentParser(description="Export features_master to Parquet")
    parser.add_argument(
        "--tier",
        choices=list(TIER_FILTER.keys()),
        default="ABC",
        help="Tier filter: A, B, C, AB, or ABC (default: ABC = all tiers)",
    )
    args = parser.parse_args()

    tier_filter = TIER_FILTER[args.tier]
    tier_label  = args.tier

    print("=" * 60)
    print("FEATURE EXPORT — hedge_v2")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tier    : {tier_label} → Data_Tier IN {tier_filter}")
    print(f"Output  : {EXPORT_DIR}/")
    print("=" * 60)

    # Row count check before loading
    expected_rows = get_row_count(tier_filter)
    if expected_rows == 0:
        print("\n❌ features_master is empty or no rows match tier filter.")
        print("   Run python data/features.py first.")
        return

    print(f"\n  Expected rows: {expected_rows:,}")

    # Load
    print("\n📥 Loading features_master...")
    df = load_features(tier_filter)

    # Validate row count matches
    if len(df) != expected_rows:
        print(f"  ⚠️  Row count mismatch: expected {expected_rows:,}, got {len(df):,}")

    # Export
    print(f"\n💾 Writing Parquet...")
    timestamped_path, latest_path = export_parquet(df, tier_label)

    # Summary
    print_summary(df, timestamped_path, latest_path)

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()