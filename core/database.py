from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


class TransactionStore:
    """
    SQLite-backed persistent store for portfolio transactions.

    Provides full CRUD operations and a pandas-compatible load interface so
    PortfolioManager can replace the CSV source with no downstream changes.

    Database: data/sovereign.db  (auto-created on first use)

    Schema
    ------
    transactions
        id          INTEGER  PK AUTOINCREMENT
        date        TEXT     ISO-8601 date string  e.g. "2025-08-05"
        type        TEXT     buy | sell | dividend | income | expense
        asset       TEXT     Human-readable name   e.g. "Apple Inc"
        ticker      TEXT     Exchange symbol        e.g. "AAPL"
        price       REAL     Per-share price (NULL for non-investment rows)
        quantity    REAL     Always positive; direction implied by type
        amount      REAL     Signed cash impact (negative = outflow)
        description TEXT     Free-text note
        created_at  TEXT     Auto-set to UTC timestamp of insertion
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS transactions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        date        TEXT    NOT NULL,
        type        TEXT    NOT NULL,
        asset       TEXT,
        ticker      TEXT,
        price       REAL,
        quantity    REAL,
        amount      REAL,
        description TEXT,
        created_at  TEXT    DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_tx_date   ON transactions (date);
    CREATE INDEX IF NOT EXISTS idx_tx_ticker ON transactions (ticker);
    CREATE INDEX IF NOT EXISTS idx_tx_type   ON transactions (type);
    """

    def __init__(self, db_path: str | Path = "data/sovereign.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")  # concurrent-read safe
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)

    @staticmethod
    def _null(val: Any) -> Any:
        """Convert NaN / empty string to None for SQLite NULL storage."""
        if val is None:
            return None
        try:
            import math
            if isinstance(val, float) and math.isnan(val):
                return None
        except Exception:
            pass
        if isinstance(val, str) and val.strip() in ("", "nan", "NaN"):
            return None
        return val

    # ------------------------------------------------------------------ #
    #  Queries                                                            #
    # ------------------------------------------------------------------ #

    def is_empty(self) -> bool:
        with self._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM transactions"
            ).fetchone()[0]
        return count == 0

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM transactions"
            ).fetchone()[0]

    def load_transactions(self) -> pd.DataFrame:
        """
        Load all transactions as a DataFrame with the same schema as the
        legacy CSV so all downstream PortfolioManager methods work unchanged.

        Columns: date (datetime), type (str), asset, ticker,
                 price (float), quantity (float), amount (float), description
        """
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT date, type, asset, ticker, price, quantity, amount, description "
                "FROM transactions ORDER BY date ASC, id ASC",
                conn,
            )

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])
        df["type"] = df["type"].str.strip().str.lower()
        numeric_cols = ["price", "quantity", "amount"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        return df.reset_index(drop=True)

    def get_by_id(self, transaction_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM transactions WHERE id = ?", (transaction_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_by_ticker(self, ticker: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM transactions WHERE ticker = ? ORDER BY date ASC",
                (ticker.upper(),),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    #  Writes                                                             #
    # ------------------------------------------------------------------ #

    def add_transaction(
        self,
        date: str,
        tx_type: str,
        asset: str | None = None,
        ticker: str | None = None,
        price: float | None = None,
        quantity: float | None = None,
        amount: float | None = None,
        description: str | None = None,
    ) -> int:
        """
        Insert a single transaction row.

        Parameters
        ----------
        date     : ISO-8601 string  e.g. "2025-08-05"
        tx_type  : "buy" | "sell" | "dividend" | "income" | "expense"

        Returns the new row's auto-incremented id.
        """
        sql = """
        INSERT INTO transactions
            (date, type, asset, ticker, price, quantity, amount, description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, (
                date,
                tx_type.strip().lower(),
                self._null(asset),
                self._null(ticker),
                self._null(price),
                self._null(quantity),
                self._null(amount),
                self._null(description),
            ))
            return cursor.lastrowid

    def update_transaction(self, transaction_id: int, **fields: Any) -> bool:
        """
        Update one or more columns of an existing transaction by id.
        Only columns in the schema are accepted; others are silently ignored.

        Example
        -------
        store.update_transaction(5, price=195.0, quantity=12.0)
        """
        allowed = {
            "date", "type", "asset", "ticker",
            "price", "quantity", "amount", "description",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False

        set_clause = ", ".join(f"{col} = ?" for col in updates)
        sql = f"UPDATE transactions SET {set_clause} WHERE id = ?"

        with self._connect() as conn:
            cursor = conn.execute(sql, (*updates.values(), transaction_id))
        return cursor.rowcount > 0

    def delete_transaction(self, transaction_id: int) -> bool:
        """Delete a transaction by id. Returns True if a row was removed."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM transactions WHERE id = ?", (transaction_id,)
            )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------ #
    #  Seeding                                                            #
    # ------------------------------------------------------------------ #

    def seed_from_csv(self, csv_path: str | Path) -> int:
        """
        Bulk-import all rows from a CSV file into the database.

        Safe to call repeatedly — skips the import entirely if the
        transactions table already has rows (idempotent).

        Returns the number of rows inserted (0 if already seeded).
        """
        if not self.is_empty():
            return 0

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Seed CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df["type"] = df["type"].str.strip().str.lower()

        sql = """
        INSERT INTO transactions
            (date, type, asset, ticker, price, quantity, amount, description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (
                str(row["date"]),
                str(row["type"]),
                self._null(row.get("asset")),
                self._null(row.get("ticker")),
                self._null(row.get("price")),
                self._null(row.get("quantity")),
                self._null(row.get("amount")),
                self._null(row.get("description")),
            )
            for _, row in df.iterrows()
        ]

        with self._connect() as conn:
            conn.executemany(sql, rows)

        return len(rows)
