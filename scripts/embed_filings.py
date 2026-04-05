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
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

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

# Seconds to sleep between Gemini embedding calls.
# Free tier: 1,500 RPM → ~0.04s minimum; 0.1s is a comfortable buffer.
EMBED_SLEEP = 0.1

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
#  Gemini embedding helper                                            #
# ------------------------------------------------------------------ #

def embed_text(client: genai.Client, text: str) -> list[float]:
    """
    Generate a retrieval-document embedding for `text` using the Gemini API.

    Returns a list[float] of length 768 (text-embedding-004 output dimension).
    """
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type="retrieval_document"),
    )
    return response.embeddings[0].values


# ------------------------------------------------------------------ #
#  Per-filing embedding                                               #
# ------------------------------------------------------------------ #

def embed_filing(
    row: dict[str, Any],
    genai_client: genai.Client,
    collection: chromadb.Collection,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Process one row from the processed_files table.

    For each section (risk_factors, mda):
      1. Load JSON chunks from the sidecar path stored in the DB row.
      2. Generate a Gemini embedding per chunk.
      3. Upsert into ChromaDB with structured metadata.

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

        doc_ids:    list[str]       = []
        embeddings: list[list[float]] = []
        documents:  list[str]       = []
        metadatas:  list[dict]      = []

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            doc_id = f"{accession}__{section}__{i:04d}"

            if dry_run:
                log.info("    [DRY RUN] would embed doc_id=%s", doc_id)
                continue

            embedding = embed_text(genai_client, chunk)
            time.sleep(EMBED_SLEEP)

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

def main(ticker_filter: str | None = None, dry_run: bool = False) -> None:
    """
    Embed all preprocessed filings, optionally restricted to one ticker.

    Reads from the processed_files table and JSON sidecar files on disk.
    Upserts embeddings into ChromaDB at data/chroma_db/.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY not set in .env — aborting.")
        sys.exit(1)

    genai_client = genai.Client(api_key=api_key)

    collection = get_collection()
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

    log.info("Processing %d filing(s).", len(df))

    total_risk = total_mda = 0

    for _, row in df.iterrows():
        accession = row["accession_number"]
        ticker    = row["ticker"].upper()
        form_type = row["form_type"]

        log.info("Filing: %s  [%s  %s]", accession, ticker, form_type)

        counts = embed_filing(
            row=row.to_dict(),
            genai_client=genai_client,
            collection=collection,
            dry_run=dry_run,
        )

        total_risk += counts.get("risk_factors", 0)
        total_mda  += counts.get("mda", 0)

    print("\n" + "=" * 65)
    print("EMBED SUMMARY")
    print("=" * 65)
    print(f"  Collection     : {COLLECTION_NAME}")
    print(f"  Total indexed  : {collection.count()} docs")
    print(f"  This run       : {total_risk} risk chunks + {total_mda} MD&A chunks upserted")
    if dry_run:
        print("  (DRY RUN — no embeddings were generated or stored)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed preprocessed SEC filings into ChromaDB.")
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
    args = parser.parse_args()
    main(ticker_filter=args.ticker, dry_run=args.dry_run)
