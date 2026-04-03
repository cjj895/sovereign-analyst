from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

# Shared database path used by both stores
DEFAULT_DB_PATH = Path("data/sovereign.db")


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


class FilingMetadataStore:
    """
    SQLite-backed store for SEC filing download records.

    Operates on the same data/sovereign.db as TransactionStore.

    Schema
    ------
    filings_metadata
        accession_number  TEXT  PK   Unique SEC identifier e.g. "0000320193-24-000123"
        ticker            TEXT       Exchange symbol e.g. "AAPL"
        cik               TEXT       10-digit Central Index Key
        form_type         TEXT       "10-K" or "10-Q"
        filing_date       TEXT       Date filed with the SEC (ISO-8601)
        period_of_report  TEXT       Financial period end date (ISO-8601)
        local_path        TEXT       Absolute path to the saved .txt file
        file_size_bytes   INTEGER    Byte count of the downloaded file
        created_at        TEXT       UTC timestamp of when the record was inserted
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS filings_metadata (
        accession_number  TEXT    PRIMARY KEY,
        ticker            TEXT    NOT NULL,
        cik               TEXT    NOT NULL,
        form_type         TEXT    NOT NULL,
        filing_date       TEXT    NOT NULL,
        period_of_report  TEXT    NOT NULL,
        local_path        TEXT    NOT NULL,
        file_size_bytes   INTEGER NOT NULL,
        created_at        TEXT    DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_fm_ticker    ON filings_metadata (ticker);
    CREATE INDEX IF NOT EXISTS idx_fm_form_type ON filings_metadata (form_type);
    CREATE INDEX IF NOT EXISTS idx_fm_period    ON filings_metadata (period_of_report);
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)

    # ------------------------------------------------------------------ #
    #  Queries                                                            #
    # ------------------------------------------------------------------ #

    def filing_exists(self, accession_number: str) -> bool:
        """Return True if a filing record already exists for this accession number."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM filings_metadata WHERE accession_number = ?",
                (accession_number,),
            ).fetchone()
        return row is not None

    def get_filings_for_ticker(
        self,
        ticker: str,
        form_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return all downloaded filings for a ticker, newest first.
        Optionally filter by form_type ("10-K" or "10-Q").
        """
        sql = (
            "SELECT * FROM filings_metadata WHERE ticker = ?"
            + (" AND form_type = ?" if form_type else "")
            + " ORDER BY period_of_report DESC"
        )
        params = (ticker.upper(), form_type) if form_type else (ticker.upper(),)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_all_filings(self) -> pd.DataFrame:
        """Return all filing records as a DataFrame sorted by period descending."""
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM filings_metadata ORDER BY period_of_report DESC",
                conn,
            )

    # ------------------------------------------------------------------ #
    #  Writes                                                             #
    # ------------------------------------------------------------------ #

    def log_filing(
        self,
        accession_number: str,
        ticker: str,
        cik: str,
        form_type: str,
        filing_date: str,
        period_of_report: str,
        local_path: str | Path,
        file_size_bytes: int,
    ) -> bool:
        """
        Insert a filing record. Skips silently if the accession_number already exists.
        Returns True if a new row was inserted, False if it was a duplicate.
        """
        if self.filing_exists(accession_number):
            return False

        sql = """
        INSERT INTO filings_metadata
            (accession_number, ticker, cik, form_type, filing_date, period_of_report,
             local_path, file_size_bytes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.execute(sql, (
                accession_number,
                ticker.upper(),
                cik,
                form_type,
                filing_date,
                period_of_report,
                str(local_path),
                file_size_bytes,
            ))
        return True


class SignalStore:
    """
    SQLite-backed store for market signals: news headlines and transcript events.

    Operates on the same data/sovereign.db as TransactionStore and FilingMetadataStore.

    Schema
    ------
    signals
        id           INTEGER  PK AUTOINCREMENT
        ticker       TEXT     Exchange symbol  e.g. "AAPL"
        type         TEXT     "news" or "transcript"
        content      TEXT     JSON string: {title, link, publisher} for news;
                              {earnings_date, link} for transcript events
        timestamp    TEXT     ISO-8601 publish/event datetime
        external_id  TEXT     Unique yfinance UUID — prevents duplicate inserts
        created_at   TEXT     UTC timestamp of when the record was inserted
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS signals (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker      TEXT    NOT NULL,
        type        TEXT    NOT NULL,
        content     TEXT    NOT NULL,
        timestamp   TEXT    NOT NULL,
        external_id TEXT    UNIQUE,
        created_at  TEXT    DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_sig_ticker    ON signals (ticker);
    CREATE INDEX IF NOT EXISTS idx_sig_type      ON signals (type);
    CREATE INDEX IF NOT EXISTS idx_sig_timestamp ON signals (timestamp);
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)

    # ------------------------------------------------------------------ #
    #  Queries                                                            #
    # ------------------------------------------------------------------ #

    def signal_exists(self, external_id: str) -> bool:
        """Return True if a signal with this external_id already exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM signals WHERE external_id = ?", (external_id,)
            ).fetchone()
        return row is not None

    def get_signals_for_ticker(
        self,
        ticker: str,
        signal_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Return the most recent signals for a ticker, newest first.
        Optionally filter by signal_type ("news" or "transcript").
        """
        sql = (
            "SELECT * FROM signals WHERE ticker = ?"
            + (" AND type = ?" if signal_type else "")
            + " ORDER BY timestamp DESC LIMIT ?"
        )
        params = (
            (ticker.upper(), signal_type, limit)
            if signal_type
            else (ticker.upper(), limit)
        )
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_all_signals(self) -> pd.DataFrame:
        """Return all signals as a DataFrame sorted by timestamp descending."""
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM signals ORDER BY timestamp DESC", conn
            )

    # ------------------------------------------------------------------ #
    #  Writes                                                             #
    # ------------------------------------------------------------------ #

    def log_signal(
        self,
        ticker: str,
        signal_type: str,
        content: str,
        timestamp: str,
        external_id: str | None = None,
    ) -> bool:
        """
        Insert a signal record. Skips silently if external_id already exists.
        Returns True if a new row was inserted, False if it was a duplicate.

        Parameters
        ----------
        ticker      : Exchange symbol e.g. "AAPL"
        signal_type : "news" or "transcript"
        content     : JSON string with the signal payload
        timestamp   : ISO-8601 datetime of the event
        external_id : Unique identifier from the source (yfinance UUID) to
                      prevent duplicate inserts across runs
        """
        if external_id and self.signal_exists(external_id):
            return False

        sql = """
        INSERT INTO signals (ticker, type, content, timestamp, external_id)
        VALUES (?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.execute(sql, (
                ticker.upper(),
                signal_type,
                content,
                timestamp,
                external_id,
            ))
        return True
