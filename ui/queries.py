"""
ui/queries.py
-------------
All database and external data access functions for the Sovereign Cockpit.

Every function is wrapped with @st.cache_data so that repeated Streamlit
reruns never make redundant SQLite reads or network calls.  The cache is
intentionally layered *on top of* PortfolioManager's own 15-min price
cache — the two caches serve different scopes (Streamlit session vs. disk).

TTL guide
---------
60 s   — live prices and OHLCV  (acceptable staleness for an equity UI)
300 s  — analyst notes / tickers (rarely change during a session)
600 s  — Chroma traces           (embeddings are deterministic; safe to hold longer)
900 s  — full portfolio pipeline (yfinance is already cached at the engine layer)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yfinance as yf

# ── project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.database import AnalystNoteStore, TransactionStore
from core.portfolio_engine import PortfolioManager

_DB_PATH = _ROOT / "data" / "sovereign.db"

# ── helper ────────────────────────────────────────────────────────────────────

def _note_store() -> AnalystNoteStore:
    return AnalystNoteStore(_DB_PATH)


def _pm() -> PortfolioManager:
    return PortfolioManager(db_path=_DB_PATH)


# ── ticker list ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_tickers() -> list[str]:
    """Return all unique tickers that have investment-type transactions."""
    store = TransactionStore(_DB_PATH)
    df = store.load_transactions()
    if df.empty:
        return []
    investment_types = {"buy", "sell", "dividend", "split"}
    mask = df["type"].isin(investment_types) & df["ticker"].notna()
    return sorted(df.loc[mask, "ticker"].str.upper().unique().tolist())


# ── portfolio pipeline ────────────────────────────────────────────────────────

@st.cache_data(ttl=900)
def get_portfolio() -> dict[str, Any]:
    """
    Run the full portfolio pipeline and return the result dict.

    Keys: holdings (DataFrame), realised (DataFrame), cash (dict), summary (DataFrame)
    Returns an empty-holdings dict if the DB has no transactions.
    """
    pm = _pm()
    try:
        return pm.run_pipeline()
    except Exception as exc:
        return {
            "holdings": pd.DataFrame(),
            "realised": pd.DataFrame(),
            "cash": {},
            "summary": pd.DataFrame(),
            "_error": str(exc),
        }


# ── OHLCV for chart ───────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted OHLCV bars for *ticker* between *start* and *end* (ISO dates).

    Returns an empty DataFrame if yfinance returns nothing or raises.
    The returned DataFrame index is DatetimeIndex; columns are Open/High/Low/Close/Volume.
    """
    try:
        df: pd.DataFrame = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # Normalise MultiIndex columns that some yfinance versions return
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()


# ── analyst notes ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_notes(ticker: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent analyst notes for *ticker*, newest first."""
    store = _note_store()
    try:
        return store.get_notes_for_ticker(ticker.upper(), limit=limit)
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_latest_delta(ticker: str) -> dict[str, Any] | None:
    """Return the latest delta note for *ticker*, or None."""
    store = _note_store()
    try:
        return store.get_latest_delta(ticker.upper())
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_all_notes() -> pd.DataFrame:
    """Return all analyst notes as a DataFrame, newest first."""
    store = _note_store()
    try:
        return store.get_all_notes()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_latest_note(ticker: str) -> dict[str, Any] | None:
    """Return the single most recent analyst note for *ticker*, or None."""
    store = _note_store()
    try:
        return store.get_latest_note(ticker.upper())
    except Exception:
        return None


# ── source-trace via ChromaDB ─────────────────────────────────────────────────

@st.cache_data(ttl=600)
def get_chroma_trace(risks_json: str, ticker: str) -> list[dict[str, Any]]:
    """
    For each risk string in *risks_json* (a JSON-encoded list of strings),
    find the closest matching 1 k-char chunk in ChromaDB and score it.

    Args:
        risks_json: JSON string so that @st.cache_data can hash the argument.
        ticker:     Ticker to filter ChromaDB results.

    Returns a list of trace dicts with keys:
        risk, best_chunk, distance, claim_score (0-100), grade (HIGH/MEDIUM/LOW)

    Returns [{"error": "..."}] when the GEMINI_API_KEY is absent or ChromaDB
    cannot be reached — callers should check for the "error" key.
    """
    from core.analysis import QueryEngine

    PREVIEW_CHARS = 350

    try:
        risks: list[str] = json.loads(risks_json)
    except (json.JSONDecodeError, ValueError):
        return [{"error": "Invalid risks JSON — cannot perform source-trace."}]

    if not risks:
        return []

    try:
        qe = QueryEngine(db_path=_DB_PATH)
    except EnvironmentError as exc:
        return [{"error": f"Source-trace unavailable: {exc}"}]
    except Exception as exc:
        return [{"error": f"ChromaDB init failed: {exc}"}]

    results: list[dict[str, Any]] = []
    for risk in risks:
        try:
            hits = qe.query(
                question=risk,
                ticker=ticker.upper(),
                section="risk_factors",
                n_results=1,
            )
        except Exception:
            hits = []

        if hits:
            best = hits[0]
            distance = float(best.get("distance", 1.0))
            claim_score = max(0, min(100, round((1.0 - distance) * 100)))
            chunk_text = str(best.get("chunk", ""))[:PREVIEW_CHARS]
            year = best.get("year", "")
        else:
            distance = 1.0
            claim_score = 0
            chunk_text = "(no matching chunk found in ChromaDB)"
            year = ""

        grade = "HIGH" if claim_score >= 80 else "MEDIUM" if claim_score >= 60 else "LOW"

        results.append({
            "risk": risk,
            "best_chunk": chunk_text,
            "distance": distance,
            "claim_score": claim_score,
            "grade": grade,
            "year": year,
        })

    return results
