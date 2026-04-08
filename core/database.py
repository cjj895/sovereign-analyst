from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DB_PATH = Path("data/sovereign.db")


# ------------------------------------------------------------------ #
#  Base Store                                                         #
# ------------------------------------------------------------------ #

class _BaseStore:
    """
    Shared foundation for all SQLite-backed stores.

    Subclasses define a class-level ``_SCHEMA`` string containing
    ``CREATE TABLE IF NOT EXISTS`` statements.  The base class handles
    connection management, WAL mode, and schema initialisation.
    """

    _SCHEMA: str = ""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)


class TransactionStore(_BaseStore):
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
        type        TEXT     buy | sell | dividend | income | expense | split
        asset       TEXT     Human-readable name   e.g. "Apple Inc"
        ticker      TEXT     Exchange symbol        e.g. "AAPL"
        price       REAL     Per-share price (NULL for non-investment rows)
        quantity    REAL     Always positive; direction implied by type
        amount      REAL     Signed cash impact (negative = outflow)
        description TEXT     Free-text note
        ratio       REAL     Split ratio (e.g. 10.0 for 10-for-1), NULL for non-split rows
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
        ratio       REAL,
        created_at  TEXT    DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_tx_date   ON transactions (date);
    CREATE INDEX IF NOT EXISTS idx_tx_ticker ON transactions (ticker);
    CREATE INDEX IF NOT EXISTS idx_tx_type   ON transactions (type);
    """

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
        Load all transactions as a DataFrame in chronological order.

        Columns: date (datetime), type (str), asset, ticker,
                 price (float), quantity (float), amount (float),
                 description, ratio (float, NULL for non-split rows)
        """
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT date, type, asset, ticker, price, quantity, "
                "amount, description, ratio "
                "FROM transactions ORDER BY date ASC, id ASC",
                conn,
            )

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])
        df["type"] = df["type"].str.strip().str.lower()
        numeric_cols = ["price", "quantity", "amount", "ratio"]
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
                "SELECT * FROM transactions WHERE ticker = ? ORDER BY date ASC, id ASC",
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
        ratio: float | None = None,
    ) -> int:
        """
        Insert a single transaction row.

        Parameters
        ----------
        date     : ISO-8601 string  e.g. "2025-08-05"
        tx_type  : "buy" | "sell" | "dividend" | "income" | "expense" | "split"
        ratio    : Split ratio (e.g. 10.0 for 10-for-1), only used for "split" type

        Returns the new row's auto-incremented id.
        """
        sql = """
        INSERT INTO transactions
            (date, type, asset, ticker, price, quantity, amount, description, ratio)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                self._null(ratio),
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
            "price", "quantity", "amount", "description", "ratio",
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
            (date, type, asset, ticker, price, quantity, amount, description, ratio)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                self._null(row.get("ratio")),
            )
            for _, row in df.iterrows()
        ]

        with self._connect() as conn:
            conn.executemany(sql, rows)

        return len(rows)


class FilingMetadataStore(_BaseStore):
    """
    SQLite-backed store for SEC filing download records.

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


class SignalStore(_BaseStore):
    """
    SQLite-backed store for market signals: news headlines and transcript events.

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


class ProcessedFilingStore(_BaseStore):
    """
    SQLite-backed store that tracks which SEC filings have been preprocessed.

    For each filing it records:
      - Paths to the clean full-document Markdown file
      - Paths to the extracted Item 1A (Risk Factors) and Item 7 (MD&A) section files
      - Paths to the JSON sidecar files holding the text chunks for each section
      - Chunk counts for quick reporting

    Schema
    ------
    processed_files
        id                 INTEGER  PK AUTOINCREMENT
        accession_number   TEXT     UNIQUE
        ticker             TEXT     Exchange symbol  e.g. "AAPL"
        form_type          TEXT     "10-K" or "10-Q"
        clean_path         TEXT     Absolute path to full-doc clean Markdown
        risk_factors_path  TEXT     Absolute path to Item 1A section Markdown
        risk_chunks_path   TEXT     Absolute path to Item 1A chunks JSON sidecar
        mda_path           TEXT     Absolute path to Item 7 section Markdown
        mda_chunks_path    TEXT     Absolute path to Item 7 chunks JSON sidecar
        risk_chunk_count   INTEGER  Number of chunks produced for Item 1A
        mda_chunk_count    INTEGER  Number of chunks produced for Item 7
        processed_at       TEXT     UTC timestamp of processing
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS processed_files (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        accession_number   TEXT    NOT NULL UNIQUE,
        ticker             TEXT    NOT NULL,
        form_type          TEXT    NOT NULL,
        clean_path         TEXT,
        risk_factors_path  TEXT,
        risk_chunks_path   TEXT,
        mda_path           TEXT,
        mda_chunks_path    TEXT,
        risk_chunk_count   INTEGER DEFAULT 0,
        mda_chunk_count    INTEGER DEFAULT 0,
        processed_at       TEXT    DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_pf_ticker ON processed_files (ticker);
    CREATE INDEX IF NOT EXISTS idx_pf_form   ON processed_files (form_type);
    """

    # ------------------------------------------------------------------ #
    #  Queries                                                            #
    # ------------------------------------------------------------------ #

    def is_processed(self, accession_number: str) -> bool:
        """Return True if a processed record already exists for this filing."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM processed_files WHERE accession_number = ?",
                (accession_number,),
            ).fetchone()
        return row is not None

    def get_processed_for_ticker(
        self,
        ticker: str,
        form_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return all processed filing records for a ticker, newest first.
        Optionally filter by form_type ("10-K" or "10-Q").
        """
        sql = (
            "SELECT * FROM processed_files WHERE ticker = ?"
            + (" AND form_type = ?" if form_type else "")
            + " ORDER BY processed_at DESC"
        )
        params = (ticker.upper(), form_type) if form_type else (ticker.upper(),)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_all_processed(self) -> pd.DataFrame:
        """Return all processed records as a DataFrame."""
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM processed_files ORDER BY processed_at DESC",
                conn,
            )

    # ------------------------------------------------------------------ #
    #  Writes                                                             #
    # ------------------------------------------------------------------ #

    def log_processed(
        self,
        accession_number: str,
        ticker: str,
        form_type: str,
        clean_path: str | Path | None = None,
        risk_factors_path: str | Path | None = None,
        risk_chunks_path: str | Path | None = None,
        mda_path: str | Path | None = None,
        mda_chunks_path: str | Path | None = None,
        risk_chunk_count: int = 0,
        mda_chunk_count: int = 0,
    ) -> bool:
        """
        Insert a processed filing record.
        Skips silently if the accession_number already exists.
        Returns True if a new row was inserted, False if it was a duplicate.
        """
        if self.is_processed(accession_number):
            return False

        sql = """
        INSERT INTO processed_files
            (accession_number, ticker, form_type, clean_path,
             risk_factors_path, risk_chunks_path, mda_path, mda_chunks_path,
             risk_chunk_count, mda_chunk_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.execute(sql, (
                accession_number,
                ticker.upper(),
                form_type,
                str(clean_path) if clean_path else None,
                str(risk_factors_path) if risk_factors_path else None,
                str(risk_chunks_path) if risk_chunks_path else None,
                str(mda_path) if mda_path else None,
                str(mda_chunks_path) if mda_chunks_path else None,
                risk_chunk_count,
                mda_chunk_count,
            ))
        return True


class AnalystNoteStore(_BaseStore):
    """
    SQLite-backed store for AI-generated analyst notes.

    Each note is produced by the Gemini API for a specific ticker, drawing
    on SEC filing chunks (Risk Factors, MD&A) and recent news signals.

    Schema
    ------
    analyst_notes
        id               INTEGER  PK AUTOINCREMENT
        ticker           TEXT     Exchange symbol  e.g. "AAPL"
        accession_number TEXT     Filing that sourced the note (NULL if multi-filing)
        model            TEXT     Gemini model used  e.g. "gemini-2.0-flash"
        summary          TEXT     One-paragraph executive summary
        risks            TEXT     JSON array of top risk strings
        sentiment        TEXT     Management sentiment: "positive" | "neutral" | "negative"
        raw_response     TEXT     Full LLM response text (for auditing)
        created_at       TEXT     UTC timestamp of generation
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS analyst_notes (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker           TEXT    NOT NULL,
        accession_number TEXT,
        model            TEXT    NOT NULL,
        summary          TEXT    NOT NULL,
        risks            TEXT    NOT NULL,
        sentiment        TEXT    NOT NULL,
        raw_response     TEXT    NOT NULL,
        created_at       TEXT    DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_an_ticker     ON analyst_notes (ticker);
    CREATE INDEX IF NOT EXISTS idx_an_created_at ON analyst_notes (created_at);
    """

    def _init_db(self) -> None:
        """Run schema creation then idempotent column migrations."""
        super()._init_db()
        self._run_migrations()

    def _run_migrations(self) -> None:
        """
        Apply incremental schema migrations that cannot use CREATE IF NOT EXISTS.

        Each ALTER TABLE is wrapped in a try/except so the method is safe
        to call on both new and existing databases.
        """
        migrations = [
            "ALTER TABLE analyst_notes ADD COLUMN delta_summary TEXT",
            "ALTER TABLE analyst_notes ADD COLUMN confidence_score REAL",
            "ALTER TABLE analyst_notes ADD COLUMN intensity_delta INTEGER",
        ]
        with self._connect() as conn:
            for sql in migrations:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError:
                    pass  # column already exists

    # ------------------------------------------------------------------ #
    #  Queries                                                            #
    # ------------------------------------------------------------------ #

    def get_latest_note(self, ticker: str) -> dict[str, Any] | None:
        """Return the most recently generated analyst note (not a delta) for a ticker, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM analyst_notes "
                "WHERE ticker = ? AND delta_summary IS NULL "
                "ORDER BY created_at DESC LIMIT 1",
                (ticker.upper(),),
            ).fetchone()
        return dict(row) if row else None

    def get_notes_for_ticker(self, ticker: str, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recent notes for a ticker, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM analyst_notes WHERE ticker = ? ORDER BY created_at DESC LIMIT ?",
                (ticker.upper(), limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_notes(self) -> pd.DataFrame:
        """Return all analyst notes as a DataFrame, newest first."""
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM analyst_notes ORDER BY created_at DESC", conn
            )

    # ------------------------------------------------------------------ #
    #  Writes                                                             #
    # ------------------------------------------------------------------ #

    def log_note(
        self,
        ticker: str,
        model: str,
        summary: str,
        risks: list[str],
        sentiment: str,
        raw_response: str,
        accession_number: str | None = None,
    ) -> int:
        """
        Persist a generated analyst note.

        Parameters
        ----------
        ticker           : Exchange symbol e.g. "AAPL"
        model            : Gemini model name e.g. "gemini-2.0-flash"
        summary          : Executive summary paragraph
        risks            : List of top risk strings (stored as JSON)
        sentiment        : "positive" | "neutral" | "negative"
        raw_response     : Full LLM response text
        accession_number : Source filing accession number (optional)

        Returns the new row's auto-incremented id.
        """
        import json as _json

        sql = """
        INSERT INTO analyst_notes
            (ticker, accession_number, model, summary, risks, sentiment, raw_response)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, (
                ticker.upper(),
                accession_number,
                model,
                summary,
                _json.dumps(risks, ensure_ascii=False),
                sentiment,
                raw_response,
            ))
            return cursor.lastrowid

    def log_delta(
        self,
        ticker: str,
        model: str,
        verdict: str,
        delta_json: str,
        raw_response: str,
        accession_number_new: str | None = None,
        accession_number_old: str | None = None,
        intensity_delta: int | None = None,
    ) -> int:
        """
        Persist a Surgical Delta analysis as an analyst_notes row.

        Parameters
        ----------
        ticker               : Exchange symbol e.g. "AAPL"
        model                : Gemini model name e.g. "gemini-2.0-flash"
        verdict              : One-sentence overall verdict from the delta
        delta_json           : Full JSON string returned by Gemini (added/removed/softened)
        raw_response         : Unmodified LLM response text (for auditing)
        accession_number_new : Accession number of the newer filing
        accession_number_old : Accession number of the older filing
        intensity_delta      : Proprietary intensity score -10 to +10.
                               Positive = language became MORE severe/alarming.
                               Negative = language became more muted/softened.
                               None = not yet scored.

        Sentiment is inferred from verdict text:
            "worsened" or "obscured" → "negative"
            "improved"               → "positive"
            otherwise                → "neutral"

        Returns the new row's auto-incremented id.
        """
        import json as _json

        verdict_lower = verdict.lower()
        if any(w in verdict_lower for w in ("worsen", "obscur", "hidden", "buried", "weaken")):
            sentiment = "negative"
        elif "improv" in verdict_lower:
            sentiment = "positive"
        else:
            sentiment = "neutral"

        ref = f"{accession_number_new or 'NEW'} vs {accession_number_old or 'OLD'}"

        sql = """
        INSERT INTO analyst_notes
            (ticker, accession_number, model, summary, risks, sentiment,
             raw_response, delta_summary, intensity_delta)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, (
                ticker.upper(),
                ref,
                model,
                verdict,
                _json.dumps([], ensure_ascii=False),
                sentiment,
                raw_response,
                delta_json,
                intensity_delta,
            ))
            return cursor.lastrowid

    def get_latest_delta(self, ticker: str) -> dict[str, Any] | None:
        """Return the most recently generated delta note for a ticker, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM analyst_notes "
                "WHERE ticker = ? AND delta_summary IS NOT NULL "
                "ORDER BY created_at DESC LIMIT 1",
                (ticker.upper(),),
            ).fetchone()
        return dict(row) if row else None

    def update_confidence(self, note_id: int, score: float) -> bool:
        """
        Persist the computed Source-Trace confidence score onto a note row.

        Parameters
        ----------
        note_id : Primary key of the analyst_notes row to update.
        score   : Confidence score in [0, 100].

        Returns True if a row was updated, False if the id was not found.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE analyst_notes SET confidence_score = ? WHERE id = ?",
                (round(score, 2), note_id),
            )
        return cursor.rowcount > 0
