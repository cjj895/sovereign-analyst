"""
scripts/analyze_deltas.py
--------------------------
Surgical Delta analyser for SEC 10-K Risk Factors (Item 1A).

Compares two annual 10-K filings for the same ticker and asks Gemini to
identify exactly what was ADDED, REMOVED, or SOFTENED in the language.
This is the highest-signal analysis in the pipeline — quiet removals and
softened language are the earliest indicators of hidden risk.

Pipeline
--------
1. Fetch the two most recent preprocessed 10-K records from processed_files,
   joined with filings_metadata for period_of_report dates.
2. Load the full Item 1A Markdown from risk_factors_path on disk.
3. Build the Surgical Delta prompt and call Gemini at temperature=0.1.
4. Parse and validate the structured JSON response.
5. Persist to analyst_notes via AnalystNoteStore.log_delta().

Run
---
    .venv/bin/python scripts/analyze_deltas.py AAPL
    .venv/bin/python scripts/analyze_deltas.py AAPL --show
    .venv/bin/python scripts/analyze_deltas.py AAPL --year-new 2025 --year-old 2024
    .venv/bin/python scripts/analyze_deltas.py AAPL --model gemini-2.0-flash
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

from core.database import AnalystNoteStore, FilingMetadataStore, ProcessedFilingStore

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

# Item 1A can be very long. Truncate to keep within Gemini's context limit.
# At ~4 chars/token, 80,000 chars ≈ 20,000 tokens — well within the 1M limit.
MAX_SECTION_CHARS = 80_000


# ------------------------------------------------------------------ #
#  Filing selection                                                    #
# ------------------------------------------------------------------ #

def _pick_filings(
    ticker: str,
    year_new: int | None = None,
    year_old: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Return (record_new, record_old) from processed_files, enriched with
    period_of_report from filings_metadata.

    If year_new / year_old are provided, the filing whose period_of_report
    starts with that year is selected. Otherwise the two most recent 10-Ks
    are used (newest = new, second = old).

    Raises ValueError if fewer than 2 preprocessed 10-Ks exist for ticker,
    or if a requested year cannot be matched.
    """
    pf_store  = ProcessedFilingStore()
    fm_store  = FilingMetadataStore()

    records = pf_store.get_processed_for_ticker(ticker.upper(), form_type="10-K")
    if len(records) < 2:
        raise ValueError(
            f"Need at least 2 preprocessed 10-K filings for {ticker}. "
            f"Found {len(records)}. Run scripts/fetch_filings.py and "
            "scripts/preprocess_filings.py first."
        )

    # Enrich each processed record with period_of_report from filings_metadata
    enriched: list[dict[str, Any]] = []
    for rec in records:
        acc = rec.get("accession_number", "")
        fm_rows = fm_store.get_filings_for_ticker(ticker.upper(), form_type="10-K")
        period = next(
            (r["period_of_report"] for r in fm_rows if r["accession_number"] == acc),
            None,
        )
        enriched.append({**rec, "period_of_report": period or ""})

    # Sort by period_of_report descending (ISO-8601 strings sort correctly)
    enriched.sort(key=lambda r: r["period_of_report"], reverse=True)

    def _by_year(target_year: int) -> dict[str, Any]:
        match = next(
            (r for r in enriched if str(r["period_of_report"]).startswith(str(target_year))),
            None,
        )
        if match is None:
            available = [r["period_of_report"] for r in enriched]
            raise ValueError(
                f"No 10-K found for {ticker} with period year {target_year}. "
                f"Available periods: {available}"
            )
        return match

    if year_new is not None and year_old is not None:
        return _by_year(year_new), _by_year(year_old)

    return enriched[0], enriched[1]


# ------------------------------------------------------------------ #
#  Text loading                                                        #
# ------------------------------------------------------------------ #

def _load_risk_text(record: dict[str, Any]) -> str:
    """
    Read the full Item 1A Markdown from disk using risk_factors_path.

    Truncates to MAX_SECTION_CHARS to stay within Gemini's context window.
    Raises FileNotFoundError if the path is missing or the file does not exist.
    """
    path_str = record.get("risk_factors_path")
    if not path_str:
        raise FileNotFoundError(
            f"No risk_factors_path for accession {record.get('accession_number')}. "
            "Ensure the filing was preprocessed successfully."
        )
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"Risk factors file not found on disk: {path}"
        )

    text = path.read_text(encoding="utf-8")
    if len(text) > MAX_SECTION_CHARS:
        log.warning(
            "Item 1A text truncated from %d to %d chars for %s.",
            len(text), MAX_SECTION_CHARS, path.name,
        )
        text = text[:MAX_SECTION_CHARS]

    return text


# ------------------------------------------------------------------ #
#  Prompt construction                                                 #
# ------------------------------------------------------------------ #

def _build_delta_prompt(
    ticker: str,
    text_new: str,
    text_old: str,
    period_new: str,
    period_old: str,
) -> str:
    """
    Assemble the Surgical Delta prompt as specified in the plan.

    temperature=0.1 is used at the call site — the prompt itself is
    instruction-heavy to minimise creative drift.
    """
    return f"""You are a forensic equity research analyst specialising in regulatory language shifts.

You are given two versions of Item 1A (Risk Factors) from {ticker}'s annual 10-K filing:
  - FILING A (older): period ending {period_old}
  - FILING B (newer): period ending {period_new}

Your task is to perform a SURGICAL DELTA — identify exactly what changed between the two
filings. Do not summarise. Do not editorialize. Identify the changes with precision.

=== FILING A — {period_old} ===
{text_old}

=== FILING B — {period_new} ===
{text_new}

---

Respond with ONLY a valid JSON object. No markdown fences. No explanation. Use this schema:

{{
  "added": [
    "<new risk disclosure that appears in Filing B but not Filing A>"
  ],
  "removed": [
    "<risk disclosure that existed in Filing A but was dropped in Filing B>"
  ],
  "softened": [
    {{
      "original": "<exact or paraphrased language from Filing A>",
      "revised":  "<corresponding language from Filing B>",
      "interpretation": "<brief explanation of why this weakening is material>"
    }}
  ],
  "verdict": "<one sentence: overall direction of risk posture change — improved, worsened, or obscured — and the most important single finding>"
}}

Rules:
- 'added' items are new risks the company has started disclosing. List each as a
  concise one-sentence description.
- 'removed' items are risks that were previously disclosed but have been quietly
  dropped. This is the most important category — missing disclosures can signal
  management is burying problems.
- 'softened' items are risks where the language became less alarming without the
  underlying risk changing. Examples: "will" -> "may", "severe" -> "potential",
  "material adverse effect" -> "could affect results".
- Keep all arrays empty ([]) if no changes of that type were found.
- The 'verdict' must be exactly one sentence."""


# ------------------------------------------------------------------ #
#  Gemini call                                                         #
# ------------------------------------------------------------------ #

def _call_gemini(
    client: genai.Client,
    prompt: str,
    model: str,
) -> tuple[str, dict[str, Any]]:
    """
    Send the delta prompt to Gemini and return (raw_text, parsed_dict).

    Uses temperature=0.1 — lower than the note generator — because delta
    analysis requires precise identification, not creative synthesis.

    Raises ValueError on invalid JSON or missing required keys.
    """
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    raw = response.text.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini did not return valid JSON: {exc}\n\nRaw:\n{raw}"
        ) from exc

    return raw, _parse_response(parsed, raw)


def _parse_response(parsed: dict[str, Any], raw: str) -> dict[str, Any]:
    """
    Validate that the parsed Gemini response contains the required keys
    and that array fields are lists.

    Returns the (possibly normalised) parsed dict.
    Raises ValueError on schema violations.
    """
    required = {"added", "removed", "softened", "verdict"}
    missing = required - parsed.keys()
    if missing:
        raise ValueError(
            f"Gemini response missing required keys: {missing}\n\nRaw:\n{raw}"
        )

    for array_key in ("added", "removed", "softened"):
        if not isinstance(parsed[array_key], list):
            raise ValueError(
                f"'{array_key}' must be a JSON array. Got: {type(parsed[array_key])}"
            )

    parsed["verdict"] = str(parsed["verdict"]).strip()
    return parsed


# ------------------------------------------------------------------ #
#  Display                                                             #
# ------------------------------------------------------------------ #

def _print_delta(
    ticker: str,
    parsed: dict[str, Any],
    period_new: str,
    period_old: str,
) -> None:
    """Pretty-print the Surgical Delta result to stdout."""
    width = 70
    print("\n" + "=" * width)
    print(f"  SURGICAL DELTA — {ticker}")
    print(f"  Comparing: {period_old}  →  {period_new}")
    print("=" * width)

    added   = parsed.get("added",   [])
    removed = parsed.get("removed", [])
    softened = parsed.get("softened", [])
    verdict  = parsed.get("verdict", "")

    print(f"\nADDED ({len(added)} new disclosures)\n")
    if added:
        for i, item in enumerate(added, 1):
            print(f"  {i}. {item}")
    else:
        print("  None found.")

    print(f"\nREMOVED ({len(removed)} dropped disclosures)  <-- highest signal\n")
    if removed:
        for i, item in enumerate(removed, 1):
            print(f"  {i}. {item}")
    else:
        print("  None found.")

    print(f"\nSOFTENED ({len(softened)} language changes)\n")
    if softened:
        for i, item in enumerate(softened, 1):
            print(f"  {i}. Original : {item.get('original', '')}")
            print(f"     Revised  : {item.get('revised', '')}")
            print(f"     Why it matters: {item.get('interpretation', '')}")
            print()
    else:
        print("  None found.")

    print(f"VERDICT\n\n  {verdict}\n")
    print("=" * width)


# ------------------------------------------------------------------ #
#  Orchestrator                                                        #
# ------------------------------------------------------------------ #

def run_delta(
    ticker: str,
    year_new: int | None = None,
    year_old: int | None = None,
    model: str = DEFAULT_MODEL,
    show: bool = False,
) -> int:
    """
    Run the Surgical Delta pipeline for `ticker` and persist the result.

    Returns the new analyst_notes row id.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY not set in .env — aborting.")
        sys.exit(1)

    ticker = ticker.upper()
    log.info("Surgical Delta for %s using %s", ticker, model)

    # -- 1. Select filings --
    try:
        rec_new, rec_old = _pick_filings(ticker, year_new, year_old)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)

    period_new = rec_new.get("period_of_report", "unknown")
    period_old = rec_old.get("period_of_report", "unknown")
    log.info("NEW filing: %s  (%s)", rec_new["accession_number"], period_new)
    log.info("OLD filing: %s  (%s)", rec_old["accession_number"], period_old)

    # -- 2. Load Item 1A text --
    try:
        text_new = _load_risk_text(rec_new)
        text_old = _load_risk_text(rec_old)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    log.info(
        "Loaded risk text — new: %d chars, old: %d chars",
        len(text_new), len(text_old),
    )

    # -- 3. Build prompt and call Gemini --
    prompt = _build_delta_prompt(ticker, text_new, text_old, period_new, period_old)
    log.info("Calling Gemini API (temperature=0.1)...")

    client = genai.Client(api_key=api_key)
    try:
        raw_response, parsed = _call_gemini(client, prompt, model)
    except ValueError as exc:
        log.error("Failed to parse Gemini response: %s", exc)
        sys.exit(1)

    log.info(
        "Delta parsed — added=%d, removed=%d, softened=%d",
        len(parsed["added"]),
        len(parsed["removed"]),
        len(parsed["softened"]),
    )

    # -- 4. Persist --
    note_store = AnalystNoteStore()
    note_id = note_store.log_delta(
        ticker=ticker,
        model=model,
        verdict=parsed["verdict"],
        delta_json=raw_response,
        raw_response=raw_response,
        accession_number_new=rec_new.get("accession_number"),
        accession_number_old=rec_old.get("accession_number"),
    )
    log.info("Delta saved to analyst_notes (id=%d)", note_id)

    if show:
        _print_delta(ticker, parsed, period_new, period_old)

    return note_id


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Surgical Delta: compare two 10-K Risk Factor sections for a ticker."
    )
    parser.add_argument("ticker", type=str, help="Ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--year-new",
        type=int,
        default=None,
        metavar="YEAR",
        help="Year of the newer filing to use (e.g. 2025). Defaults to most recent.",
    )
    parser.add_argument(
        "--year-old",
        type=int,
        default=None,
        metavar="YEAR",
        help="Year of the older filing to use (e.g. 2024). Defaults to second most recent.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the delta report to stdout after saving.",
    )
    args = parser.parse_args()
    run_delta(
        ticker=args.ticker,
        year_new=args.year_new,
        year_old=args.year_old,
        model=args.model,
        show=args.show,
    )
