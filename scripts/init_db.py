"""
Initialise all SQLite tables for the Sovereign Analyst project.

Each store class auto-creates its own table on first instantiation,
so this script simply instantiates every store to guarantee the full
schema exists.

Run:  .venv/bin/python scripts/init_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.database import (
    TransactionStore,
    FilingMetadataStore,
    SignalStore,
    ProcessedFilingStore,
)

DB_PATH = Path("data/sovereign.db")


def init_db(db_path: str | Path = DB_PATH) -> None:
    TransactionStore(db_path)
    FilingMetadataStore(db_path)
    SignalStore(db_path)
    ProcessedFilingStore(db_path)
    print(f"Database initialised at {db_path}")


if __name__ == "__main__":
    init_db()
