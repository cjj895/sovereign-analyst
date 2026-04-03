from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import TypedDict

import requests
from dotenv import load_dotenv

# Allow imports from the project root when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.database import FilingMetadataStore

RAW_FILINGS_DIR = PROJECT_ROOT / "data" / "raw_filings"
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

# Validation constants
MIN_FILING_SIZE_BYTES: int = 10_240          # 10 KB
TOC_MARKERS: tuple[str, ...] = (
    "table of contents",
    "item 1.",                               # fallback for older plain-text filings
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


class FilingMetadata(TypedDict):
    cik: str
    accession_number: str
    filing_date: str
    period_of_report: str


# ------------------------------------------------------------------ #
#  Environment                                                        #
# ------------------------------------------------------------------ #

def load_environment() -> None:
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def get_user_agent() -> str:
    agent = os.getenv("USER_AGENT", "").strip()
    if not agent:
        raise EnvironmentError(
            "USER_AGENT missing from .env. Format: Your Name (email@example.com)"
        )
    return agent


# ------------------------------------------------------------------ #
#  SEC API Helpers                                                    #
# ------------------------------------------------------------------ #

def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def sec_get_json(url: str, user_agent: str) -> dict:
    resp = requests.get(
        url,
        headers={"User-Agent": user_agent, "Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def sec_get_text(url: str, user_agent: str) -> str:
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    return resp.text


def get_cik_for_ticker(ticker: str, user_agent: str) -> str:
    ticker_upper = ticker.upper()
    ticker_map = sec_get_json(SEC_TICKER_MAP_URL, user_agent)
    for item in ticker_map.values():
        if str(item.get("ticker", "")).upper() == ticker_upper:
            return f"{int(item['cik_str']):010d}"
    raise ValueError(f"Ticker '{ticker_upper}' not found in SEC ticker map.")


# ------------------------------------------------------------------ #
#  Filing Metadata Retrieval                                         #
# ------------------------------------------------------------------ #

def get_latest_filings_metadata(
    ticker: str,
    form_type: str,
    user_agent: str,
    n: int = 1,
) -> list[FilingMetadata]:
    """
    Return metadata for the n most recent filings of form_type for ticker.

    Searches the SEC Submissions API recent-filings list chronologically
    from newest to oldest and collects up to n matches.

    Raises ValueError if fewer than 1 match is found.
    """
    cik = get_cik_for_ticker(ticker, user_agent)
    submissions = sec_get_json(SEC_SUBMISSIONS_URL.format(cik=cik), user_agent)

    recent = submissions.get("filings", {}).get("recent", {})
    forms             = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates      = recent.get("filingDate", [])
    periods           = recent.get("periodOfReport", [])

    results: list[FilingMetadata] = []
    for i, form in enumerate(forms):
        if form == form_type:
            if i >= len(accession_numbers) or i >= len(filing_dates):
                continue
            results.append({
                "cik": cik,
                "accession_number": accession_numbers[i],
                "filing_date": filing_dates[i],
                "period_of_report": periods[i] if i < len(periods) else filing_dates[i],
            })
        if len(results) == n:
            break

    if not results:
        raise ValueError(
            f"No {form_type} filings found for '{ticker.upper()}' in SEC submissions."
        )

    return results


# ------------------------------------------------------------------ #
#  Validation                                                         #
# ------------------------------------------------------------------ #

def validate_filing_content(text: str) -> dict:
    """
    Validate a downloaded filing against two criteria:

    1. File size >= MIN_FILING_SIZE_BYTES (10 KB).
       A real 10-K/10-Q is several MB; smaller files are error pages.

    2. Text contains at least one TOC marker (case-insensitive).
       "Table of Contents" is present in every structured EDGAR filing.
       Its absence indicates an incomplete or malformed download.

    Returns a dict:
        is_valid       bool   True only when both checks pass.
        size_bytes     int
        size_readable  str    Human-readable e.g. "33.27 MB"
        failure_reason str    Empty string when valid, description when not.
    """
    size = len(text.encode("utf-8"))
    size_readable = _format_size(size)
    text_lower = text.lower()

    if size < MIN_FILING_SIZE_BYTES:
        return {
            "is_valid": False,
            "size_bytes": size,
            "size_readable": size_readable,
            "failure_reason": (
                f"File too small ({size_readable}). "
                "Likely an SEC error page, not a real filing."
            ),
        }

    has_toc = any(marker in text_lower for marker in TOC_MARKERS)
    if not has_toc:
        return {
            "is_valid": False,
            "size_bytes": size,
            "size_readable": size_readable,
            "failure_reason": (
                "No 'Table of Contents' or 'Item 1.' found. "
                "Filing may be incomplete or in an unexpected format."
            ),
        }

    return {
        "is_valid": True,
        "size_bytes": size,
        "size_readable": size_readable,
        "failure_reason": "",
    }


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.2f} MB"
    if size_bytes >= 1_024:
        return f"{size_bytes / 1_024:.1f} KB"
    return f"{size_bytes} B"


# ------------------------------------------------------------------ #
#  Save & Log                                                        #
# ------------------------------------------------------------------ #

def save_filing(
    ticker: str,
    form_type: str,
    filing_date: str,
    text: str,
) -> Path:
    """Write the filing text to data/raw_filings/ and return the path."""
    RAW_FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{sanitize_filename(ticker.upper())}"
        f"_{sanitize_filename(form_type)}"
        f"_{sanitize_filename(filing_date)}.txt"
    )
    path = RAW_FILINGS_DIR / filename
    path.write_text(text, encoding="utf-8")
    return path


def log_to_db(
    store: FilingMetadataStore,
    filing: FilingMetadata,
    ticker: str,
    form_type: str,
    local_path: Path,
    file_size_bytes: int,
) -> None:
    inserted = store.log_filing(
        accession_number=filing["accession_number"],
        ticker=ticker,
        cik=filing["cik"],
        form_type=form_type,
        filing_date=filing["filing_date"],
        period_of_report=filing["period_of_report"],
        local_path=local_path,
        file_size_bytes=file_size_bytes,
    )
    if inserted:
        log.info(
            "Logged to DB: %s %s period=%s path=%s",
            ticker, form_type, filing["period_of_report"], local_path.name,
        )
    else:
        log.info("Already in DB: %s %s %s", ticker, form_type, filing["accession_number"])


# ------------------------------------------------------------------ #
#  Core Fetch Function                                               #
# ------------------------------------------------------------------ #

def fetch_filing(
    ticker: str,
    form_type: str,
    filing: FilingMetadata,
    user_agent: str,
    store: FilingMetadataStore,
) -> Path | None:
    """
    Download a single filing, validate it, save it, and log it to the DB.

    Returns the local Path on success, None on validation failure.
    Does NOT raise — logs errors instead so bulk fetches continue.
    """
    # Skip if already downloaded and logged
    if store.filing_exists(filing["accession_number"]):
        log.info(
            "Skipping (already downloaded): %s %s %s",
            ticker, form_type, filing["period_of_report"],
        )
        return Path(store.get_filings_for_ticker(ticker, form_type)[0]["local_path"])

    cik_no_zeros = str(int(filing["cik"]))
    accession_no_dashes = filing["accession_number"].replace("-", "")
    url = (
        f"{SEC_ARCHIVES_BASE_URL}/{cik_no_zeros}"
        f"/{accession_no_dashes}/{filing['accession_number']}.txt"
    )

    log.info("Downloading %s %s %s ...", ticker, form_type, filing["period_of_report"])
    try:
        text = sec_get_text(url, user_agent)
    except Exception as exc:
        log.error("Download FAILED: %s %s — %s", ticker, form_type, exc)
        return None

    validation = validate_filing_content(text)

    if not validation["is_valid"]:
        log.error(
            "Validation FAILED: %s %s %s — %s",
            ticker, form_type, filing["period_of_report"],
            validation["failure_reason"],
        )
        return None

    path = save_filing(
        ticker=ticker,
        form_type=form_type,
        filing_date=filing["filing_date"],
        text=text,
    )
    log_to_db(store, filing, ticker, form_type, path, validation["size_bytes"])
    log.info(
        "Saved: %s  (%s)", path.name, validation["size_readable"]
    )
    return path


# ------------------------------------------------------------------ #
#  High-Level Helpers                                                #
# ------------------------------------------------------------------ #

def fetch_filings_for_ticker(
    ticker: str,
    user_agent: str,
    store: FilingMetadataStore,
    n_annual: int = 1,
    n_quarterly: int = 4,
) -> dict[str, list[Path | None]]:
    """
    Fetch the latest n_annual 10-K(s) and n_quarterly 10-Q(s) for a ticker.

    Returns a dict mapping form_type → list of local Paths (None = failed).
    """
    results: dict[str, list[Path | None]] = {"10-K": [], "10-Q": []}

    for form_type, n in (("10-K", n_annual), ("10-Q", n_quarterly)):
        try:
            filings = get_latest_filings_metadata(ticker, form_type, user_agent, n=n)
        except ValueError as exc:
            log.warning("Could not retrieve %s metadata for %s: %s", form_type, ticker, exc)
            continue

        for filing in filings:
            path = fetch_filing(ticker, form_type, filing, user_agent, store)
            results[form_type].append(path)

    return results


# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SEC filings (10-K and/or 10-Q) for a ticker."
    )
    parser.add_argument("ticker", type=str, help="Ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--type",
        dest="form_type",
        choices=["10-K", "10-Q", "both"],
        default="both",
        help="Filing type to download (default: both)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help="Number of 10-Q filings to fetch when type is '10-Q' or 'both' (default: 4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_environment()
    user_agent = get_user_agent()
    store = FilingMetadataStore()

    ticker = args.ticker.upper()

    if args.form_type == "both":
        results = fetch_filings_for_ticker(
            ticker, user_agent, store,
            n_annual=1, n_quarterly=args.count,
        )
        for form, paths in results.items():
            for p in paths:
                if p:
                    print(f"  [{form}] {p}")
    else:
        n = 1 if args.form_type == "10-K" else args.count
        try:
            filings = get_latest_filings_metadata(ticker, args.form_type, user_agent, n=n)
        except ValueError as exc:
            log.error(exc)
            return

        for filing in filings:
            path = fetch_filing(ticker, args.form_type, filing, user_agent, store)
            if path:
                print(f"  [{args.form_type}] {path}")


if __name__ == "__main__":
    main()
