"""
scripts/generate_analyst_notes.py
----------------------------------
AI-powered analyst note generator.

For a given ticker, this script:
  1. Pulls the current portfolio position (quantity, avg cost, market value).
  2. Fetches the latest 10 news headlines from SignalStore.
  3. Loads up to MAX_CHUNKS chunks each from Risk Factors and MD&A section
     JSON sidecar files (from the most recent preprocessed filing).
  4. Sends all context to the Gemini API and requests a structured JSON note
     containing an executive summary, top 3 risk factors, and a management
     sentiment label.
  5. Persists the note to the analyst_notes table via AnalystNoteStore.

Run
---
    .venv/bin/python scripts/generate_analyst_notes.py AAPL
    .venv/bin/python scripts/generate_analyst_notes.py AAPL --show
    .venv/bin/python scripts/generate_analyst_notes.py AAPL --model gemini-2.0-flash
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from core.database import AnalystNoteStore, ProcessedFilingStore, SignalStore
from core.portfolio_engine import PortfolioManager

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

DEFAULT_MODEL = "gemini-2.0-flash"

# Max chunks to include per section — keeps the prompt within token limits
MAX_RISK_CHUNKS = 6
MAX_MDA_CHUNKS  = 6

# ------------------------------------------------------------------ #
#  Data gathering                                                     #
# ------------------------------------------------------------------ #

def _get_position(pm: PortfolioManager, ticker: str) -> dict[str, Any] | None:
    """Return the current portfolio position for ticker, or None if not held."""
    holdings_df, _, _ = pm.recalculate_portfolio()
    if holdings_df.empty or ticker not in holdings_df.index:
        return None
    row = holdings_df.loc[ticker]
    prices = pm.get_live_prices([ticker])
    current_price = prices[0] if prices else None
    capital = float(row.get("capital_invested", 0) or 0)
    qty = float(row.get("quantity", 0) or 0)
    market_val = qty * current_price if current_price else None
    unrealised = (market_val - capital) if market_val is not None else None
    return {
        "quantity":         qty,
        "avg_cost":         float(row.get("avg_cost", 0) or 0),
        "capital_invested": capital,
        "current_price":    current_price,
        "market_value":     market_val,
        "unrealised_pnl":   unrealised,
        "unrealised_pnl_pct": (
            round(unrealised / capital * 100, 2)
            if unrealised is not None and capital else None
        ),
    }


def _get_news_headlines(ticker: str, limit: int = 10) -> list[str]:
    """Return the most recent news headline strings for ticker."""
    signal_store = SignalStore()
    signals = signal_store.get_signals_for_ticker(ticker, signal_type="news", limit=limit)
    headlines: list[str] = []
    for s in signals:
        try:
            payload = json.loads(s["content"])
            title = payload.get("title", "").strip()
            publisher = payload.get("publisher", "").strip()
            ts = s.get("timestamp", "")[:10]
            if title:
                headlines.append(f"[{ts}] {publisher}: {title}")
        except (json.JSONDecodeError, KeyError):
            pass
    return headlines


def _load_chunks(chunks_path_str: str | None, max_chunks: int) -> list[str]:
    """Load up to max_chunks from a JSON sidecar path, or return []."""
    if not chunks_path_str:
        return []
    path = Path(chunks_path_str)
    if not path.exists():
        log.warning("Chunks sidecar not found on disk: %s", path)
        return []
    chunks: list[str] = json.loads(path.read_text(encoding="utf-8"))
    return chunks[:max_chunks]


def _get_filing_chunks(
    ticker: str,
    max_risk: int = MAX_RISK_CHUNKS,
    max_mda: int = MAX_MDA_CHUNKS,
) -> tuple[list[str], list[str], str | None]:
    """
    Return (risk_chunks, mda_chunks, accession_number) for the most recent
    preprocessed filing for ticker. All lists are empty if no filing exists.
    """
    store = ProcessedFilingStore()
    records = store.get_processed_for_ticker(ticker)
    if not records:
        return [], [], None

    latest = records[0]  # newest first from DB query
    risk_chunks = _load_chunks(latest.get("risk_chunks_path"), max_risk)
    mda_chunks  = _load_chunks(latest.get("mda_chunks_path"),  max_mda)
    return risk_chunks, mda_chunks, latest.get("accession_number")


# ------------------------------------------------------------------ #
#  Prompt construction                                                #
# ------------------------------------------------------------------ #

def _build_prompt(
    ticker: str,
    position: dict[str, Any] | None,
    headlines: list[str],
    risk_chunks: list[str],
    mda_chunks: list[str],
) -> str:
    """
    Assemble the full analyst prompt for Gemini.

    The model is asked to return a JSON object with exactly three keys:
        summary   : str   — one paragraph executive summary
        risks     : list  — exactly 3 concise risk strings
        sentiment : str   — "positive" | "neutral" | "negative"
    """
    lines: list[str] = []

    lines.append("You are a senior equity research analyst.")
    lines.append(
        f"Produce a structured analyst note for {ticker} based solely on the "
        "context provided below. Do not fabricate data."
    )
    lines.append("")

    # Position context
    if position:
        lines.append("## Portfolio Position")
        lines.append(f"- Quantity held     : {position['quantity']:.4f} shares")
        lines.append(f"- Average cost      : ${position['avg_cost']:.4f}")
        lines.append(f"- Capital invested  : ${position['capital_invested']:,.2f}")
        if position.get("current_price"):
            lines.append(f"- Current price     : ${position['current_price']:.2f}")
        if position.get("market_value"):
            lines.append(f"- Market value      : ${position['market_value']:,.2f}")
        if position.get("unrealised_pnl_pct") is not None:
            lines.append(f"- Unrealised P/L    : {position['unrealised_pnl_pct']:.2f}%")
    else:
        lines.append(f"## Portfolio Position\n{ticker} is not currently held in the portfolio.")

    lines.append("")

    # News headlines
    if headlines:
        lines.append("## Recent News Headlines")
        for h in headlines:
            lines.append(f"- {h}")
    else:
        lines.append("## Recent News Headlines\nNone available.")

    lines.append("")

    # Risk Factors chunks
    if risk_chunks:
        lines.append("## Risk Factors (from SEC Filing — Item 1A)")
        for i, chunk in enumerate(risk_chunks, 1):
            lines.append(f"\n### Chunk {i}\n{chunk.strip()}")
    else:
        lines.append("## Risk Factors\nNo filing chunks available.")

    lines.append("")

    # MD&A chunks
    if mda_chunks:
        lines.append("## Management Discussion & Analysis (from SEC Filing — Item 7)")
        for i, chunk in enumerate(mda_chunks, 1):
            lines.append(f"\n### Chunk {i}\n{chunk.strip()}")
    else:
        lines.append("## MD&A\nNo filing chunks available.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "Respond with ONLY a valid JSON object — no markdown fences, no explanation. "
        "Use exactly this schema:\n"
        '{\n'
        '  "summary": "<one paragraph executive summary>",\n'
        '  "risks": ["<risk 1>", "<risk 2>", "<risk 3>"],\n'
        '  "sentiment": "<positive|neutral|negative>"\n'
        '}'
    )

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  LLM call                                                           #
# ------------------------------------------------------------------ #

def _call_gemini(
    client: genai.Client,
    prompt: str,
    model: str,
) -> tuple[str, dict[str, Any]]:
    """
    Send `prompt` to Gemini and return (raw_response_text, parsed_dict).

    Raises ValueError if the response cannot be parsed as valid JSON with
    the required keys.
    """
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    raw = response.text.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini did not return valid JSON: {exc}\n\nRaw:\n{raw}") from exc

    required = {"summary", "risks", "sentiment"}
    missing = required - parsed.keys()
    if missing:
        raise ValueError(f"Gemini response missing required keys: {missing}\n\nRaw:\n{raw}")

    sentiment = str(parsed["sentiment"]).lower().strip()
    if sentiment not in {"positive", "neutral", "negative"}:
        log.warning("Unexpected sentiment value '%s' — storing as-is.", sentiment)
    parsed["sentiment"] = sentiment

    if not isinstance(parsed.get("risks"), list):
        raise ValueError(f"'risks' must be a JSON array. Got: {type(parsed['risks'])}")

    return raw, parsed


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def generate_note(
    ticker: str,
    model: str = DEFAULT_MODEL,
    show: bool = False,
) -> int:
    """
    Generate and persist an analyst note for `ticker`.

    Returns the new analyst_notes row id.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY not set in .env — aborting.")
        sys.exit(1)

    ticker = ticker.upper()
    log.info("Generating analyst note for %s using %s", ticker, model)

    # -- 1. Gather context --
    pm = PortfolioManager()
    position = _get_position(pm, ticker)
    headlines = _get_news_headlines(ticker)
    risk_chunks, mda_chunks, accession_number = _get_filing_chunks(ticker)

    log.info(
        "Context: position=%s | headlines=%d | risk_chunks=%d | mda_chunks=%d",
        "held" if position else "not held",
        len(headlines),
        len(risk_chunks),
        len(mda_chunks),
    )

    if not risk_chunks and not mda_chunks:
        log.warning(
            "No filing chunks found for %s. "
            "Run scripts/preprocess_filings.py first, then scripts/embed_filings.py.",
            ticker,
        )

    # -- 2. Build prompt and call LLM --
    prompt = _build_prompt(ticker, position, headlines, risk_chunks, mda_chunks)
    log.info("Calling Gemini API...")

    client = genai.Client(api_key=api_key)
    raw_response, parsed = _call_gemini(client, prompt, model)

    log.info("Sentiment: %s", parsed["sentiment"])

    # -- 3. Persist --
    note_store = AnalystNoteStore()
    note_id = note_store.log_note(
        ticker=ticker,
        model=model,
        summary=str(parsed["summary"]),
        risks=list(parsed["risks"]),
        sentiment=parsed["sentiment"],
        raw_response=raw_response,
        accession_number=accession_number,
    )

    log.info("Note saved to analyst_notes table (id=%d)", note_id)

    if show:
        _print_note(ticker, parsed)

    return note_id


def _print_note(ticker: str, parsed: dict[str, Any]) -> None:
    """Pretty-print a parsed analyst note to stdout."""
    print("\n" + "=" * 70)
    print(f"  ANALYST NOTE — {ticker}")
    print("=" * 70)
    print("\nSUMMARY\n")
    print(f"  {parsed['summary']}")
    print("\nTOP RISKS\n")
    for i, risk in enumerate(parsed["risks"], 1):
        print(f"  {i}. {risk}")
    print(f"\nMANAGEMENT SENTIMENT:  {parsed['sentiment'].upper()}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an AI analyst note for a ticker.")
    parser.add_argument("ticker", type=str, help="Ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the generated note to stdout after saving.",
    )
    args = parser.parse_args()
    generate_note(ticker=args.ticker, model=args.model, show=args.show)
