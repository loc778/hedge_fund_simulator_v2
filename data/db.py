# data/db.py
# Shared database utilities — hedge_v2
# Identical pattern to prototype db.py, points to hedge_v2_db

from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import insert
from dotenv import load_dotenv
from urllib.parse import quote_plus
import os

load_dotenv()

def get_engine():
    user     = os.getenv('DB_USER')
    password = quote_plus(os.getenv('DB_PASSWORD'))
    host     = os.getenv('DB_HOST')
    db       = os.getenv('DB_NAME', 'hedge_v2_db')   # defaults to hedge_v2_db

    DB_URL = f"mysql+mysqlconnector://{user}:{password}@{host}/{db}"
    return create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)


def upsert_ignore(table, conn, keys, data_iter):
    """Insert rows, silently skip if duplicate key exists (INSERT IGNORE)."""
    stmt = insert(table.table).prefix_with("IGNORE")
    data = [dict(zip(keys, row)) for row in data_iter]
    conn.execute(stmt, data)


def save_to_db(df, table_name, engine, chunk_size=5000):
    """
    Standard save function used by all scripts.
    Always uses INSERT IGNORE via upsert_ignore to prevent duplicate crashes.
    Chunked to prevent memory issues on large datasets (500 stocks x 15 years).
    """
    total = len(df)
    print(f"💾 Saving {total:,} rows to {table_name}...")

    df.to_sql(
        name      = table_name,
        con       = engine,
        if_exists = "append",
        index     = False,
        method    = upsert_ignore,
        chunksize = chunk_size,
    )

    print(f"✅ Saved {total:,} rows to {table_name}")