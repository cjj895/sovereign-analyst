import sqlite3
from pathlib import Path

DB_PATH = Path("data/sovereign.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    type        TEXT    NOT NULL, -- buy, sell, dividend, income, expense, SPLIT
    asset       TEXT,
    ticker      TEXT,
    price       REAL,
    quantity    REAL,
    amount      REAL,
    description TEXT,
    ratio       REAL,      -- Only used for SPLIT type (e.g. 10.0 for 10-for-1)
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tx_date   ON transactions (date);
CREATE INDEX IF NOT EXISTS idx_tx_ticker ON transactions (ticker);
"""

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.close()
    print(f"Database initialized at {DB_PATH}")

if __name__ == "__main__":
    init_db()
