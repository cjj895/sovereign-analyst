"""
scripts/embed_filings.py
------------------------
Embedding pipeline: reads every preprocessed filing row from SQLite,
loads the Risk Factors and MD&A chunk JSON sidecars from disk, generates
an embedding for each chunk via the Gemini API, and upserts into a
persistent ChromaDB collection.

Idempotent — uses ChromaDB upsert with deterministic document IDs, so
re-running the script for the same filing is always safe.

Document ID format
------------------
    {accession_number}__{section}__{chunk_idx:04d}

    e.g.  0000320193-25-000079__risk_factors__0042

Run
---
    .venv/bin/python scripts/embed_filings.py
    .venv/bin/python scripts/embed_filings.py --ticker AAPL
    .venv/bin/python scripts/embed_filings.py --ticker AAPL --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

import chromadb
from chromadb.config import Settings

from core.database import ProcessedFilingStore

# ------------------------------------------------------------------ #
#  Logging                                                            #
# ------------------------------------------------------------------ #

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Constants                                                          #
# ------------------------------------------------------------------ #

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "sovereign_filings"
EMBED_MODEL = "models/text-embedding-004"

# Seconds to sleep between Gemini embedding calls (per-thread).
# Free tier: 1,500 RPM total → 0.1s per request leaves headroom.
EMBED_SLEEP = 0.1

# Maximum concurrent Gemini API calls across all threads.
# At 3 workers × 0.1s sleep ≈ 30 RPM safety margin against free-tier limits.
MAX_EMBED_WORKERS = 3

# Sections we process.  Maps section key → column name in processed_files.
SECTIONS: dict[str, str] = {
    "risk_factors": "risk_chunks_path",
    "mda":          "mda_chunks_path",
}


# ------------------------------------------------------------------ #
#  ChromaDB helpers                                                   #
# ------------------------------------------------------------------ #

def get_collection(chroma_path: Path = CHROMA_PATH) -> chromadb.Collection:
    """
    Return (or create) the sovereign_filings ChromaDB collection.

    Uses cosine distance — the right metric for normalised embeddings from
    Gemini's text-embedding-004 model.
    """
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ------------------------------------------------------------------ #
#  Gemini embedding helper (with tenacity retry for 429 errors)      #
# ------------------------------------------------------------------ #

def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if the exception looks like a Gemini 429 rate-limit."""
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg


def embed_text(client: genai.Client, text: str) -> list[float]:
    """
    Generate a retrieval-document embedding for `text` using the Gemini API.

    Decorated with tenacity to automatically retry on 429 RESOURCE_EXHAUSTED
    errors with exponential back-off (2s → 4s → 8s → 16s → 60s cap, 5 tries).

    Returns a list[float] of length 768 (text-embedding-004 output dimension).
    """
    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def _call() -> list[float]:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=genai_types.EmbedContentConfig(task_type="retrieval_document"),
        )
        return response.embeddings[0].values

    return _call()


# ------------------------------------------------------------------ #
#  Per-filing embedding                                               #
# ------------------------------------------------------------------ #

def embed_filing(
    row: dict[str, Any],
    genai_client: genai.Client,
    collection: chromadb.Collection,
    dry_run: bool = False,
    api_semaphore: threading.Semaphore | None = None,
    upsert_lock: threading.Lock | None = None,
) -> dict[str, int]:
    """
    Process one row from the processed_files table.

    For each section (risk_factors, mda):
      1. Load JSON chunks from the sidecar path stored in the DB row.
      2. Generate a Gemini embedding per chunk (with tenacity retry on 429).
      3. Batch all chunks for the section and upsert into ChromaDB.

    Parameters
    ----------
    api_semaphore : Shared semaphore capping concurrent Gemini API calls
                    across all threads.  Pass None for single-threaded use.
    upsert_lock   : Shared lock serializing ChromaDB upsert calls across
                    threads.  Pass None for single-threaded use.

    Returns a dict {section: chunks_upserted} for reporting.
    """
    accession = row["accession_number"]
    ticker     = row["ticker"].upper()
    form_type  = row["form_type"]

    # Derive filing year from processed_at timestamp ("2025-10-31 …")
    year = int(str(row.get("processed_at", "0000"))[:4])

    counts: dict[str, int] = {}

    for section, col in SECTIONS.items():
        chunks_path_str: str | None = row.get(col)
        if not chunks_path_str:
            log.debug("  %s | %s | no chunks path — skipping.", accession, section)
            counts[section] = 0
            continue

        chunks_path = Path(chunks_path_str)
        if not chunks_path.exists():
            log.warning("  %s | %s | sidecar not on disk: %s", accession, section, chunks_path)
            counts[section] = 0
            continue

        chunks: list[str] = json.loads(chunks_path.read_text(encoding="utf-8"))
        if not chunks:
            counts[section] = 0
            continue

        log.info(
            "  %s | %s | %s | %d chunks",
            ticker, form_type, section, len(chunks),
        )

        doc_ids:    list[str]         = []
        embeddings: list[list[float]] = []
        documents:  list[str]         = []
        metadatas:  list[dict]        = []

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            doc_id = f"{accession}__{section}__{i:04d}"

            if dry_run:
                log.info("    [DRY RUN] would embed doc_id=%s", doc_id)
                continue

            # Acquire semaphore to cap concurrent Gemini API calls
            if api_semaphore is not None:
                api_semaphore.acquire()
            try:
                embedding = embed_text(genai_client, chunk)
                time.sleep(EMBED_SLEEP)
            finally:
                if api_semaphore is not None:
                    api_semaphore.release()

            doc_ids.append(doc_id)
            embeddings.append(embedding)
            documents.append(chunk)
            metadatas.append({
                "ticker":           ticker,
                "form_type":        form_type,
                "section":          section,
                "year":             year,
                "accession_number": accession,
                "chunk_idx":        i,
            })

        if doc_ids:
            with upsert_lock or threading.Lock():
                collection.upsert(
                    ids=doc_ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )

        counts[section] = len(doc_ids)

    return counts


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main(
    ticker_filter: str | None = None,
    dry_run: bool = False,
    max_workers: int = MAX_EMBED_WORKERS,
) -> None:
    """
    Embed all preprocessed filings in parallel, optionally restricted to
    one ticker.

    Reads from the processed_files table and JSON sidecar files on disk.
    Upserts embeddings into ChromaDB at data/chroma_db/.

    Concurrency controls
    --------------------
    api_semaphore : Caps concurrent Gemini API calls to `max_workers` so we
                    stay within rate limits even when multiple filings are
                    processed simultaneously.
    upsert_lock   : Serialises ChromaDB upsert() calls — the ChromaDB
                    PersistentClient is not guaranteed to be thread-safe for
                    concurrent writes.

    Error isolation
    ---------------
    If one filing's embedding fails (e.g. malformed sidecar, persistent 429),
    it is logged and the pool continues with the remaining filings.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY not set in .env — aborting.")
        sys.exit(1)

    genai_client  = genai.Client(api_key=api_key)
    collection    = get_collection()
    api_semaphore = threading.Semaphore(max_workers)
    upsert_lock   = threading.Lock()

    log.info(
        "ChromaDB collection '%s' ready — %d docs already indexed.",
        COLLECTION_NAME,
        collection.count(),
    )

    store = ProcessedFilingStore()
    df    = store.get_all_processed()

    if df.empty:
        log.error("No preprocessed filings found. Run scripts/preprocess_filings.py first.")
        sys.exit(1)

    if ticker_filter:
        df = df[df["ticker"].str.upper() == ticker_filter.upper()]
        if df.empty:
            log.error("No preprocessed filings for ticker '%s'.", ticker_filter)
            sys.exit(1)

    log.info("Embedding %d filing(s) with %d worker(s).", len(df), max_workers)

    rows = [row.to_dict() for _, row in df.iterrows()]

    total_risk = total_mda = 0
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                embed_filing,
                row,
                genai_client,
                collection,
                dry_run,
                api_semaphore,
                upsert_lock,
            ): row["accession_number"]
            for row in rows
        }
        for future in as_completed(futures):
            accession = futures[future]
            try:
                counts = future.result()
                total_risk += counts.get("risk_factors", 0)
                total_mda  += counts.get("mda", 0)
                log.info(
                    "Done: %s  (risk=%d, mda=%d)",
                    accession,
                    counts.get("risk_factors", 0),
                    counts.get("mda", 0),
                )
            except Exception as exc:
                log.error(
                    "Failed to embed %s: %s",
                    accession, exc, exc_info=True,
                )
                errors.append(accession)

    print("\n" + "=" * 65)
    print("EMBED SUMMARY")
    print("=" * 65)
    print(f"  Collection     : {COLLECTION_NAME}")
    print(f"  Total indexed  : {collection.count()} docs")
    print(f"  This run       : {total_risk} risk chunks + {total_mda} MD&A chunks upserted")
    if errors:
        print(f"  Errors         : {len(errors)} filing(s) failed → {errors}")
    if dry_run:
        print("  (DRY RUN — no embeddings were generated or stored)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed preprocessed SEC filings into ChromaDB (parallel, with retry)."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Only embed filings for this ticker (e.g. AAPL).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk through filings without calling the Gemini API or writing to ChromaDB.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_EMBED_WORKERS,
        metavar="N",
        help=f"Max concurrent Gemini API calls (default: {MAX_EMBED_WORKERS}).",
    )
    args = parser.parse_args()
    main(ticker_filter=args.ticker, dry_run=args.dry_run, max_workers=args.workers)
