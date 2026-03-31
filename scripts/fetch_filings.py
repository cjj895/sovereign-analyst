from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import TypedDict

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILINGS_DIR = PROJECT_ROOT / "data" / "raw_filings"
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"


class FilingMetadata(TypedDict):
    cik: str
    accession_number: str
    filing_date: str


def load_environment() -> None:
    """Load environment variables from project-level .env."""
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=env_path)


def get_user_agent() -> str:
    """Get SEC-compliant user agent from .env."""
    user_agent = os.getenv("USER_AGENT", "").strip()
    if not user_agent:
        raise EnvironmentError(
            "USER_AGENT not found in .env. Use format: Name (email@example.com)."
        )
    return user_agent


def sanitize_filename(value: str) -> str:
    """Convert arbitrary text into a safe file name component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def sec_get_json(url: str, user_agent: str) -> dict:
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def sec_get_text(url: str, user_agent: str) -> str:
    headers = {"User-Agent": user_agent}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def get_cik_for_ticker(ticker: str, user_agent: str) -> str:
    """Resolve ticker to 10-digit SEC CIK string."""
    ticker_upper = ticker.upper()
    ticker_map = sec_get_json(SEC_TICKER_MAP_URL, user_agent)
    for item in ticker_map.values():
        if str(item.get("ticker", "")).upper() == ticker_upper:
            cik_int = int(item["cik_str"])
            return f"{cik_int:010d}"
    raise ValueError(f"Ticker '{ticker_upper}' not found in SEC ticker mapping.")


def get_latest_10k_metadata(ticker: str, user_agent: str) -> FilingMetadata:
    """Fetch latest 10-K metadata for a ticker via SEC submissions endpoint."""
    cik = get_cik_for_ticker(ticker, user_agent)
    submissions_url = SEC_SUBMISSIONS_URL.format(cik=cik)
    submissions = sec_get_json(submissions_url, user_agent)

    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])

    for index, form in enumerate(forms):
        if form == "10-K":
            return {
                "cik": cik,
                "accession_number": accession_numbers[index],
                "filing_date": filing_dates[index],
            }

    raise ValueError(f"No 10-K filings found for ticker '{ticker.upper()}'.")


def save_filing_text(ticker: str, filing_date: str, filing_text: str) -> Path:
    """Persist filing text to data/raw_filings/ and return output path."""
    RAW_FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    safe_ticker = sanitize_filename(ticker.upper())
    safe_date = sanitize_filename(filing_date)
    output_path = RAW_FILINGS_DIR / f"{safe_ticker}_10-K_{safe_date}.txt"
    output_path.write_text(filing_text, encoding="utf-8")
    return output_path


def fetch_latest_10k(ticker: str) -> Path:
    """Fetch latest 10-K filing text from SEC EDGAR and save it locally."""
    load_environment()
    user_agent = get_user_agent()

    filing = get_latest_10k_metadata(ticker, user_agent)
    cik_no_leading_zeros = str(int(filing["cik"]))
    accession_no_dashes = filing["accession_number"].replace("-", "")
    filing_url = (
        f"{SEC_ARCHIVES_BASE_URL}/"
        f"{cik_no_leading_zeros}/"
        f"{accession_no_dashes}/"
        f"{filing['accession_number']}.txt"
    )

    filing_text = sec_get_text(filing_url, user_agent)
    filing_date = filing["filing_date"]
    return save_filing_text(ticker=ticker, filing_date=filing_date, filing_text=filing_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the latest 10-K filing for a ticker."
    )
    parser.add_argument("ticker", type=str, help="Ticker symbol, e.g. AAPL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = fetch_latest_10k(args.ticker)
    print(f"Saved latest 10-K to: {output}")


if __name__ == "__main__":
    main()
