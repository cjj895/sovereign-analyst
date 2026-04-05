"""
core/analysis.py
----------------
QueryEngine: semantic search over embedded SEC filing chunks.

Usage
-----
    from core.analysis import QueryEngine

    qe = QueryEngine()

    # Unrestricted — searches across all tickers and sections
    results = qe.query("What are the main liquidity risks?")

    # Scoped to a single ticker
    results = qe.query("Discuss revenue growth drivers", ticker="AAPL")

    # Scoped to a ticker + section
    results = qe.query("Forward-looking guidance on margins", ticker="AAPL", section="mda")

    for r in results:
        print(r["ticker"], r["section"], r["year"], r["distance"])
        print(r["chunk"][:300])
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

import chromadb
from chromadb.config import Settings

from core.database import DEFAULT_DB_PATH

# ------------------------------------------------------------------ #
#  Defaults                                                           #
# ------------------------------------------------------------------ #

DEFAULT_CHROMA_PATH = Path("data/chroma_db")
COLLECTION_NAME = "sovereign_filings"
EMBED_MODEL = "models/text-embedding-004"


# ------------------------------------------------------------------ #
#  QueryEngine                                                        #
# ------------------------------------------------------------------ #

class QueryEngine:
    """
    Semantic search engine over embedded SEC filing chunks.

    Embeds natural language questions with Gemini text-embedding-004
    (task_type='retrieval_query') and retrieves the most relevant chunks
    from the persistent ChromaDB collection.

    Parameters
    ----------
    db_path     : Path to sovereign.db (unused at query time, reserved for
                  future hybrid SQL + vector queries).
    chroma_path : Path to the ChromaDB PersistentClient directory.
    api_key     : Gemini API key.  Falls back to GEMINI_API_KEY env var.
    """

    COLLECTION = COLLECTION_NAME
    EMBED_MODEL = EMBED_MODEL

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        chroma_path: str | Path = DEFAULT_CHROMA_PATH,
        api_key: str | None = None,
    ) -> None:
        load_dotenv()

        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to .env or pass api_key= to QueryEngine()."
            )

        self._genai = genai.Client(api_key=resolved_key)

        chroma_path = Path(chroma_path)
        chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def indexed_count(self) -> int:
        """Total number of chunks currently indexed in ChromaDB."""
        return self._collection.count()

    def query(
        self,
        question: str,
        ticker: str | None = None,
        section: str | None = None,
        form_type: str | None = None,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Embed `question` and return the top-n most semantically relevant
        filing chunks.

        Parameters
        ----------
        question  : Natural language question (e.g. "What are liquidity risks?")
        ticker    : Restrict to a single ticker, e.g. "AAPL".  None = all tickers.
        section   : "risk_factors" | "mda" | None (both).
        form_type : "10-K" | "10-Q" | None (both).
        n_results : Number of results to return (default 5).

        Returns
        -------
        list[dict] where each dict contains:
            chunk      : str   — the raw text of the chunk
            ticker     : str
            section    : str   — "risk_factors" or "mda"
            form_type  : str   — "10-K" or "10-Q"
            year       : int
            accession_number : str
            chunk_idx  : int
            distance   : float — cosine distance (lower = more similar)
        """
        if not question.strip():
            return []

        embedding = self._embed_question(question)
        where = self._build_where(ticker, section, form_type)

        # ChromaDB requires n_results <= collection size
        cap = min(n_results, max(self._collection.count(), 1))

        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results":        cap,
            "include":          ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        response = self._collection.query(**kwargs)

        return self._parse_response(response)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _embed_question(self, question: str) -> list[float]:
        """
        Embed a query string with task_type='retrieval_query'.

        This asymmetric task type is intentionally different from the
        'retrieval_document' used during indexing — Gemini optimises the
        two embeddings for maximum retrieval accuracy.
        """
        response = self._genai.models.embed_content(
            model=self.EMBED_MODEL,
            contents=question,
            config=genai_types.EmbedContentConfig(task_type="retrieval_query"),
        )
        return response.embeddings[0].values

    @staticmethod
    def _build_where(
        ticker: str | None,
        section: str | None,
        form_type: str | None,
    ) -> dict[str, Any] | None:
        """
        Build a ChromaDB metadata filter dict, or None if no filters are needed.

        Combines multiple filters with $and when more than one is active.
        """
        clauses: list[dict[str, Any]] = []

        if ticker:
            clauses.append({"ticker": {"$eq": ticker.upper()}})
        if section:
            clauses.append({"section": {"$eq": section}})
        if form_type:
            clauses.append({"form_type": {"$eq": form_type.upper()}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Flatten the ChromaDB query response into a list of result dicts.
        """
        results: list[dict[str, Any]] = []

        ids_list       = response.get("ids",        [[]])[0]
        documents_list = response.get("documents",  [[]])[0]
        metadatas_list = response.get("metadatas",  [[]])[0]
        distances_list = response.get("distances",  [[]])[0]

        for doc_id, chunk, meta, dist in zip(
            ids_list, documents_list, metadatas_list, distances_list
        ):
            results.append({
                "doc_id":           doc_id,
                "chunk":            chunk,
                "ticker":           meta.get("ticker", ""),
                "section":          meta.get("section", ""),
                "form_type":        meta.get("form_type", ""),
                "year":             meta.get("year", 0),
                "accession_number": meta.get("accession_number", ""),
                "chunk_idx":        meta.get("chunk_idx", -1),
                "distance":         round(float(dist), 6),
            })

        return results
