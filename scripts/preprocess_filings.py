"""
scripts/preprocess_filings.py
------------------------------
Preprocessing pipeline for raw SEC EDGAR filings.

Pipeline per filing
-------------------
1. Extract the primary HTML document from the SEC SGML wrapper.
2. Strip HTML with BeautifulSoup and convert structural tags to Markdown.
3. Use compiled regex patterns to extract Item 1A (Risk Factors) and
   Item 7 (MD&A) sections.
4. Chunk each section into 1,000-character segments with 150-character
   overlap and save as JSON sidecar files.
5. Log all output paths to the processed_files table in sovereign.db.

Run
---
    .venv/bin/python scripts/preprocess_filings.py
    .venv/bin/python scripts/preprocess_filings.py --ticker AAPL
"""
from __future__ import annotations

import json
import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.database import ProcessedFilingStore

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Paths                                                              #
# ------------------------------------------------------------------ #

RAW_DIR = PROJECT_ROOT / "data" / "raw_filings"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ------------------------------------------------------------------ #
#  Compiled Regex — module-level so they are only compiled once       #
# ------------------------------------------------------------------ #

# Matches "Item 1A" heading variants: "ITEM 1A.", "Item 1A:", "Item 1A –"
_ITEM_1A = re.compile(
    r"item\s+1a\.?\s*[:\-\u2013\u2014]?\s*risk\s+factors",
    re.IGNORECASE,
)

# Matches "Item 7" heading variants (10-K: MD&A)
_ITEM_7 = re.compile(
    r"item\s+7\.?\s*[:\-\u2013\u2014]?\s*management.{0,15}s?\s+discussion",
    re.IGNORECASE,
)

# Matches "Item 2" heading variants (10-Q: MD&A is Part I, Item 2)
_ITEM_2_MDA = re.compile(
    r"item\s+2\.?\s*[:\-\u2013\u2014]?\s*management.{0,15}s?\s+discussion",
    re.IGNORECASE,
)

# Matches the start of ANY Item heading — used to find section boundaries
_NEXT_ITEM = re.compile(
    r"^item\s+\d+[a-z]?[\.\s\-\u2013\u2014]",
    re.IGNORECASE | re.MULTILINE,
)


# ------------------------------------------------------------------ #
#  Step 1 — SEC SGML → primary HTML extraction                       #
# ------------------------------------------------------------------ #

def _extract_primary_document(raw_text: str, form_type: str) -> str | None:
    """
    The raw filing is an SEC SGML wrapper containing multiple <DOCUMENT>
    blocks. Find the block whose <TYPE> tag exactly matches `form_type`
    (e.g. "10-K" or "10-Q") and return the content between <TEXT>...</TEXT>.

    Returns None if no matching block is found.
    """
    # Split on <DOCUMENT> boundaries (case-insensitive, handles \r\n)
    doc_blocks = re.split(r"<DOCUMENT>", raw_text, flags=re.IGNORECASE)

    for block in doc_blocks:
        type_match = re.match(r"\s*<TYPE>([^\n\r]+)", block, re.IGNORECASE)
        if not type_match:
            continue
        block_type = type_match.group(1).strip()
        if block_type.upper() != form_type.upper():
            continue

        # Extract content between <TEXT>...</TEXT>
        text_match = re.search(
            r"<TEXT>(.*?)</TEXT>",
            block,
            re.IGNORECASE | re.DOTALL,
        )
        if text_match:
            return text_match.group(1).strip()

    return None


# ------------------------------------------------------------------ #
#  Step 2 — HTML → clean Markdown                                     #
# ------------------------------------------------------------------ #

def html_to_markdown(html: str) -> str:
    """
    Parse `html` with BeautifulSoup and convert structural tags to
    Markdown. Returns a clean, human-readable Markdown string.

    Conversions applied:
        h1 → # heading
        h2 → ## heading
        h3 → ## heading  (flattened — 10-K h3 are sub-section titles)
        h4-h6 → ### heading
        b / strong → **bold**
        li → - bullet
        br → newline
        table → tab-separated rows (preserves numbers)
        All other tags → stripped, text extracted
    """
    soup = BeautifulSoup(html, "html.parser")

    lines: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, str):
            text = node.strip()
            if text:
                lines.append(text)
            return

        if not isinstance(node, Tag):
            return

        tag = node.name.lower() if node.name else ""

        if tag == "br":
            lines.append("")
            return

        if tag in ("script", "style"):
            return

        if tag == "h1":
            lines.append(f"\n# {node.get_text(strip=True)}\n")
            return

        if tag == "h2":
            lines.append(f"\n## {node.get_text(strip=True)}\n")
            return

        if tag in ("h3", "h4", "h5", "h6"):
            lines.append(f"\n### {node.get_text(strip=True)}\n")
            return

        if tag in ("b", "strong"):
            text = node.get_text(strip=True)
            if text:
                lines.append(f"**{text}**")
            return

        if tag == "li":
            lines.append(f"- {node.get_text(' ', strip=True)}")
            return

        if tag == "table":
            for row in node.find_all("tr"):
                cells = [td.get_text(" ", strip=True) for td in row.find_all(["td", "th"])]
                if any(cells):
                    lines.append("\t".join(cells))
            lines.append("")
            return

        if tag == "p":
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(f"\n{text}\n")
            return

        # Default: recurse into children
        for child in node.children:
            _walk(child)

    body = soup.find("body") or soup
    _walk(body)

    # Collapse runs of blank lines to at most two
    result_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                result_lines.append("")
        else:
            blank_count = 0
            result_lines.append(line)

    return "\n".join(result_lines).strip()


# ------------------------------------------------------------------ #
#  Step 3 — Section extraction                                        #
# ------------------------------------------------------------------ #

def _best_match_for_patterns(
    clean_text: str,
    patterns: list[re.Pattern[str]],
    min_chars: int = 200,
) -> str | None:
    """
    Evaluate all occurrences of all patterns against clean_text and
    return the candidate with the most content.

    For each match, the section end is the next Item heading after it
    (or end-of-document). Candidates shorter than `min_chars` are
    rejected to filter out Table-of-Contents references.
    """
    best: str | None = None

    for pattern in patterns:
        for start_match in pattern.finditer(clean_text):
            end_pos = len(clean_text)
            for m in _NEXT_ITEM.finditer(clean_text, pos=start_match.end()):
                end_pos = m.start()
                break

            candidate = clean_text[start_match.start():end_pos].strip()

            if len(candidate) >= min_chars:
                if best is None or len(candidate) > len(best):
                    best = candidate

    return best


def extract_section(clean_text: str, section: str) -> str | None:
    """
    Extract a named section from the clean text using compiled regex.

    Parameters
    ----------
    clean_text : The full Markdown/plain text of the filing.
    section    : "risk_factors" (Item 1A) or "mda" (Item 7 / Item 2).

    Strategy
    --------
    SEC filings include a Table of Contents near the top that contains
    short one-line references to "Item 1A. Risk Factors" etc. A naive
    first-match approach returns only the TOC entry (a few dozen chars).

    Instead we evaluate ALL occurrences of the target header and select
    the one whose content (up to the next Item heading) is longest.
    This reliably skips TOC references and lands on the full section.

    For MD&A specifically, 10-K filings use Item 7 and 10-Q filings use
    Item 2 — both patterns are tried and the longest result wins.

    Returns None if no occurrence with >=200 chars of content is found.
    """
    if section == "risk_factors":
        patterns = [_ITEM_1A]
    else:
        # Try Item 7 (10-K) and Item 2 (10-Q) — return whichever is longest
        patterns = [_ITEM_7, _ITEM_2_MDA]

    best = _best_match_for_patterns(clean_text, patterns)

    if best is None:
        log.warning("Section '%s' not found (or too short) in text.", section)

    return best


# ------------------------------------------------------------------ #
#  Step 4 — Chunking                                                  #
# ------------------------------------------------------------------ #

def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """
    Split `text` into chunks of at most `size` characters with `overlap`
    characters of context carried over between consecutive chunks.

    Parameters
    ----------
    text    : The text to chunk.
    size    : Maximum characters per chunk (default 1,000).
    overlap : Characters of overlap between consecutive chunks (default 150).

    Returns a list of non-empty chunk strings.
    """
    if not text:
        return []

    step = size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be less than size ({size})")

    chunks: list[str] = []
    i = 0
    while i < len(text):
        chunk = text[i : i + size]
        if chunk.strip():
            chunks.append(chunk)
        i += step

    return chunks


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _parse_filename(raw_path: Path) -> dict[str, str]:
    """
    Parse metadata from a raw filing filename.
    Expected format: TICKER_FORMTYPE_DATE.txt  e.g. AAPL_10-K_2025-10-31.txt

    Returns dict with keys: ticker, form_type, date, stem.
    """
    stem = raw_path.stem  # e.g. "AAPL_10-K_2025-10-31"
    parts = stem.split("_", 2)
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {raw_path.name}")
    return {
        "ticker": parts[0].upper(),
        "form_type": parts[1],   # e.g. "10-K"
        "date": parts[2],        # e.g. "2025-10-31"
        "stem": stem,
    }


def _extract_accession_number(raw_text: str) -> str | None:
    """
    Pull the accession number out of the SEC SGML header.
    Format in header: "ACCESSION NUMBER:\t\t0000320193-25-000079"
    """
    m = re.search(r"ACCESSION NUMBER:\s+([\d\-]+)", raw_text, re.IGNORECASE)
    return m.group(1).strip() if m else None


# ------------------------------------------------------------------ #
#  Step 5 — Orchestration                                             #
# ------------------------------------------------------------------ #

def process_filing(
    raw_path: Path,
    store: ProcessedFilingStore,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    db_lock: threading.Lock | None = None,
) -> dict[str, Any] | None:
    """
    Run the full preprocessing pipeline for a single raw filing file.

    Thread-safe: pass a shared threading.Lock via `db_lock` when calling
    from a ThreadPoolExecutor so SQLite writes are serialized across workers.

    Returns a result dict on success, or None if the filing was already
    processed (idempotent — safe to call repeatedly).

    Result dict keys
    ----------------
        ticker, form_type, accession_number,
        clean_path, risk_factors_path, mda_path,
        risk_chunk_count, mda_chunk_count, skipped
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        meta = _parse_filename(raw_path)
    except ValueError as exc:
        log.error("Skipping %s — %s", raw_path.name, exc)
        return None

    ticker = meta["ticker"]
    form_type = meta["form_type"]
    stem = meta["stem"]

    log.info("Processing  %s  [%s  %s]", raw_path.name, ticker, form_type)

    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")

    accession_number = _extract_accession_number(raw_text)
    if not accession_number:
        log.warning("Could not extract accession number from %s. Skipping.", raw_path.name)
        return None

    if store.is_processed(accession_number):
        log.info("  Already processed — skipping.")
        return {"skipped": True, "accession_number": accession_number}

    # Step 1 — extract primary HTML document from SGML wrapper
    html = _extract_primary_document(raw_text, form_type)
    if not html:
        log.error("  Could not extract primary %s document block — skipping.", form_type)
        return None

    # Step 2 — HTML → clean Markdown
    log.info("  Stripping HTML with BeautifulSoup...")
    clean_text = html_to_markdown(html)

    clean_path = PROCESSED_DIR / f"{stem}_clean.md"
    clean_path.write_text(clean_text, encoding="utf-8")
    log.info("  Clean file → %s  (%d chars)", clean_path.name, len(clean_text))

    # Step 3 — extract sections
    risk_chunk_count = 0
    mda_chunk_count = 0
    risk_factors_path: Path | None = None
    risk_chunks_path: Path | None = None
    mda_path: Path | None = None
    mda_chunks_path: Path | None = None

    risk_text = extract_section(clean_text, "risk_factors")
    if risk_text:
        risk_factors_path = PROCESSED_DIR / f"{stem}_risk_factors.md"
        risk_factors_path.write_text(risk_text, encoding="utf-8")

        # Step 4a — chunk risk factors
        risk_chunks = chunk_text(risk_text, size=chunk_size, overlap=chunk_overlap)
        risk_chunk_count = len(risk_chunks)
        risk_chunks_path = PROCESSED_DIR / f"{stem}_risk_factors_chunks.json"
        risk_chunks_path.write_text(
            json.dumps(risk_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("  Risk Factors → %d chars, %d chunks", len(risk_text), risk_chunk_count)
    else:
        log.warning("  Item 1A (Risk Factors) not found in %s.", raw_path.name)

    mda_text = extract_section(clean_text, "mda")
    if mda_text:
        mda_path = PROCESSED_DIR / f"{stem}_mda.md"
        mda_path.write_text(mda_text, encoding="utf-8")

        # Step 4b — chunk MD&A
        mda_chunks = chunk_text(mda_text, size=chunk_size, overlap=chunk_overlap)
        mda_chunk_count = len(mda_chunks)
        mda_chunks_path = PROCESSED_DIR / f"{stem}_mda_chunks.json"
        mda_chunks_path.write_text(
            json.dumps(mda_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("  MD&A        → %d chars, %d chunks", len(mda_text), mda_chunk_count)
    else:
        log.warning("  Item 7 (MD&A) not found in %s.", raw_path.name)

    # Step 5 — log to DB (serialized via lock when running in parallel)
    with db_lock or threading.Lock():
        store.log_processed(
            accession_number=accession_number,
            ticker=ticker,
            form_type=form_type,
            clean_path=clean_path,
            risk_factors_path=risk_factors_path,
            risk_chunks_path=risk_chunks_path,
            mda_path=mda_path,
            mda_chunks_path=mda_chunks_path,
            risk_chunk_count=risk_chunk_count,
            mda_chunk_count=mda_chunk_count,
        )
    log.info("  Logged to DB  [%s]", accession_number)

    return {
        "skipped": False,
        "ticker": ticker,
        "form_type": form_type,
        "accession_number": accession_number,
        "clean_path": clean_path,
        "risk_factors_path": risk_factors_path,
        "mda_path": mda_path,
        "risk_chunk_count": risk_chunk_count,
        "mda_chunk_count": mda_chunk_count,
    }


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def main(
    ticker_filter: str | None = None,
    max_workers: int = 4,
) -> None:
    """
    Iterate data/raw_filings/*.txt and preprocess each one in parallel.

    Parameters
    ----------
    ticker_filter : Only process filings for this ticker (e.g. "AAPL").
    max_workers   : Thread pool size (default 4).
                    BeautifulSoup is I/O-heavy for large files so threads
                    provide a meaningful speedup even with the GIL.

    Thread safety
    -------------
    - A single threading.Lock serializes all SQLite writes so no two
      threads corrupt the processed_files table simultaneously.
    - If one filing raises an unhandled exception, the error is logged
      and the pool continues with the remaining files.
    """
    store    = ProcessedFilingStore()
    db_lock  = threading.Lock()

    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        log.error("No raw filings found in %s", RAW_DIR)
        return

    if ticker_filter:
        raw_files = [
            f for f in raw_files
            if f.name.upper().startswith(ticker_filter.upper() + "_")
        ]

    log.info(
        "Found %d filing(s) to process (max_workers=%d).",
        len(raw_files), max_workers,
    )

    results: list[dict[str, Any]] = []
    errors:  list[str]            = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(process_filing, raw_path, store, 1000, 150, db_lock): raw_path
            for raw_path in raw_files
        }
        for future in as_completed(futures):
            raw_path = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                log.error(
                    "Unhandled error processing %s: %s",
                    raw_path.name, exc, exc_info=True,
                )
                errors.append(raw_path.name)

    # Summary
    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"  {'File':<40} {'Status':<10} {'Risk':<8} {'MD&A':<8}")
    print(f"  {'-'*40} {'-'*10} {'-'*8} {'-'*8}")
    for r in results:
        if r.get("skipped"):
            print(f"  {'(skipped)':<40} {'SKIP':<10}")
            continue
        name = f"{r['ticker']}_{r['form_type']}"
        risk = str(r["risk_chunk_count"]) + " chunks"
        mda  = str(r["mda_chunk_count"]) + " chunks"
        print(f"  {name:<40} {'OK':<10} {risk:<8} {mda:<8}")
    if errors:
        print(f"\n  ERRORS ({len(errors)} filing(s) failed):")
        for e in errors:
            print(f"    ✗ {e}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess SEC EDGAR filings (parallel)."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Only process filings for this ticker (e.g. AAPL).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel worker threads (default: 4).",
    )
    args = parser.parse_args()
    main(ticker_filter=args.ticker, max_workers=args.workers)
