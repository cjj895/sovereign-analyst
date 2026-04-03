from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.database import SignalStore

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# Keywords that indicate a news item links to an earnings call transcript
TRANSCRIPT_KEYWORDS: tuple[str, ...] = (
    "transcript",
    "earnings call",
    "earnings conference",
    "q1 call",
    "q2 call",
    "q3 call",
    "q4 call",
)


# ------------------------------------------------------------------ #
#  News Fetching                                                      #
# ------------------------------------------------------------------ #

def _unix_to_iso(unix_ts: int | float) -> str:
    """Convert a Unix timestamp to an ISO-8601 string in UTC."""
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()


def fetch_news_for_ticker(
    ticker: str,
    store: SignalStore,
    count: int = 10,
) -> int:
    """
    Fetch the latest `count` news items for a ticker via yfinance and
    persist each one to the signals table.

    Returns the number of new records inserted (skips duplicates).
    """
    t = yf.Ticker(ticker)
    try:
        news_items: list[dict[str, Any]] = t.get_news(count=count)
    except Exception as exc:
        log.error("Failed to fetch news for %s: %s", ticker, exc)
        return 0

    inserted = 0
    for item in news_items:
        uuid = item.get("id") or item.get("uuid") or ""
        if not uuid:
            continue

        publish_time = item.get("providerPublishTime") or item.get("published") or 0
        timestamp = _unix_to_iso(publish_time) if publish_time else datetime.now(timezone.utc).isoformat()

        content = json.dumps({
            "title":     item.get("title", ""),
            "link":      item.get("link", ""),
            "publisher": item.get("publisher") or item.get("source", ""),
        })

        if store.log_signal(
            ticker=ticker,
            signal_type="news",
            content=content,
            timestamp=timestamp,
            external_id=uuid,
        ):
            inserted += 1

    log.info("News  | %s | %d new / %d total fetched", ticker, inserted, len(news_items))
    return inserted


# ------------------------------------------------------------------ #
#  Transcript Event Fetching                                         #
# ------------------------------------------------------------------ #

def fetch_transcript_signals_for_ticker(
    ticker: str,
    store: SignalStore,
    news_count: int = 30,
) -> int:
    """
    Detect transcript-related signals for a ticker using two methods:

    Method A — Earnings calendar:
        Uses yfinance `ticker.calendar` to record the most recent/upcoming
        earnings date as a "transcript" signal.  This tells you *when*
        the call happened so you can find the transcript manually or via
        a future AI analysis step.

    Method B — News scan:
        Scans the latest news for titles containing transcript keywords
        (e.g., "Earnings Call Transcript", "Q3 Call").  Any matches are
        stored with type="transcript" and include the article link.

    Returns the number of new records inserted.
    """
    t = yf.Ticker(ticker)
    inserted = 0

    # -- Method A: earnings calendar --
    try:
        cal = t.calendar
        earnings_dates: list[Any] = []
        if isinstance(cal, dict):
            earnings_dates = cal.get("Earnings Date", [])
        elif hasattr(cal, "iloc"):
            earnings_dates = cal.get("Earnings Date", [])

        for ed in earnings_dates[:2]:  # at most 2 (prev + next)
            try:
                ts = pd.Timestamp(ed) if not isinstance(ed, str) else pd.Timestamp(ed)
                ts_iso = ts.isoformat()
                external_id = f"earnings_date_{ticker.upper()}_{ts_iso}"
                content = json.dumps({
                    "earnings_date": ts_iso,
                    "note": "Earnings event date from yfinance calendar.",
                })
                if store.log_signal(
                    ticker=ticker,
                    signal_type="transcript",
                    content=content,
                    timestamp=ts_iso,
                    external_id=external_id,
                ):
                    inserted += 1
                    log.info("Transcript | %s | earnings date logged: %s", ticker, ts_iso)
            except Exception:
                continue
    except Exception as exc:
        log.warning("Could not fetch calendar for %s: %s", ticker, exc)

    # -- Method B: news scan for transcript links --
    try:
        news_items = t.get_news(count=news_count)
    except Exception as exc:
        log.warning("Could not fetch news for transcript scan (%s): %s", ticker, exc)
        news_items = []

    for item in news_items:
        title = item.get("title", "").lower()
        if not any(kw in title for kw in TRANSCRIPT_KEYWORDS):
            continue

        uuid = item.get("id") or item.get("uuid") or ""
        if not uuid:
            continue

        transcript_external_id = f"transcript_news_{uuid}"
        publish_time = item.get("providerPublishTime") or item.get("published") or 0
        timestamp = _unix_to_iso(publish_time) if publish_time else datetime.now(timezone.utc).isoformat()

        content = json.dumps({
            "title":     item.get("title", ""),
            "link":      item.get("link", ""),
            "publisher": item.get("publisher") or item.get("source", ""),
            "note":      "Transcript-related news item detected by keyword filter.",
        })

        if store.log_signal(
            ticker=ticker,
            signal_type="transcript",
            content=content,
            timestamp=timestamp,
            external_id=transcript_external_id,
        ):
            inserted += 1
            log.info(
                "Transcript | %s | news link logged: %s",
                ticker, item.get("title", "")[:60],
            )

    return inserted


# ------------------------------------------------------------------ #
#  Top-Holdings Entry Point                                          #
# ------------------------------------------------------------------ #

def fetch_signals_for_tickers(
    tickers: list[str],
    db_path: str | Path = "data/sovereign.db",
    news_count: int = 10,
) -> dict[str, dict[str, int]]:
    """
    Fetch news and transcript signals for a list of tickers and persist
    them to the signals table.

    Returns a dict mapping ticker → {news: N, transcript: M} inserted counts.
    """
    store = SignalStore(db_path)
    results: dict[str, dict[str, int]] = {}

    for ticker in tickers:
        log.info("--- Fetching signals for %s ---", ticker)
        news_inserted = fetch_news_for_ticker(ticker, store, count=news_count)
        transcript_inserted = fetch_transcript_signals_for_ticker(ticker, store)
        results[ticker] = {"news": news_inserted, "transcript": transcript_inserted}

    return results


# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #

def main() -> None:
    """
    Fetch signals for the top 5 holdings from the live portfolio.
    Run:  .venv/bin/python scripts/fetch_signals.py
    """
    from core.portfolio_engine import PortfolioManager

    pm = PortfolioManager()
    result = pm.run_pipeline()
    holdings = result.get("holdings")

    if holdings is None or holdings.empty:
        log.error("No holdings found. Add transactions first.")
        return

    top_tickers: list[str] = (
        holdings.sort_values("market_value", ascending=False)
        .head(5)
        .index.tolist()
    )

    log.info("Fetching signals for top 5 holdings: %s", top_tickers)
    counts = fetch_signals_for_tickers(top_tickers)

    print("\n" + "=" * 50)
    print("SIGNAL FETCH SUMMARY")
    print("=" * 50)
    for ticker, c in counts.items():
        print(f"  {ticker:<6} | News: {c['news']:>2} new  | Transcripts: {c['transcript']:>2} new")


if __name__ == "__main__":
    import pandas as pd
    main()
