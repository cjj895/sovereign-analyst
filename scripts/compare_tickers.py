"""
scripts/compare_tickers.py
---------------------------
Cross-Ticker Competitive Intelligence Engine.

Fetches semantically relevant SEC filing chunks for two or more tickers on a
given theme, then asks Gemini to:
  - Identify risks unique to each company.
  - Identify shared risks where one company's language is more "intense".
  - Assign a Relative Risk Score (-50 to +50) for the primary two tickers,
    where positive = Ticker A is more exposed, negative = Ticker B.

Design decisions
----------------
- QueryEngine queries ChromaDB with a multi-ticker $in filter for efficiency,
  then splits results per-ticker before building the prompt.
- Only the two "primary" tickers (first and second in the list) receive the
  Relative Risk Score. Additional tickers appear in the unique-risks section.
- The Gemini call uses temperature=0.2 — moderate reasoning, not creative.
- Nothing is written to the database; this is a pure analysis/display command.

Run
---
    .venv/bin/python scripts/compare_tickers.py NVDA AMD --theme "China Export Controls"
    .venv/bin/python scripts/compare_tickers.py NVDA AMD INTC --theme "AI Strategy" --n 6
    .venv/bin/python scripts/compare_tickers.py NVDA AMD --theme "Supply Chain" \\
        --model gemini-2.0-flash
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

from core.analysis import QueryEngine

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

DEFAULT_MODEL  = "gemini-2.0-flash"
DEFAULT_CHUNKS = 4    # chunks per ticker fed to Gemini
MAX_CHUNK_CHARS = 600  # preview chars per chunk in the prompt


# ------------------------------------------------------------------ #
#  Data retrieval                                                     #
# ------------------------------------------------------------------ #

def _fetch_chunks_by_ticker(
    qe: QueryEngine,
    tickers: list[str],
    theme: str,
    n_per_ticker: int,
) -> dict[str, list[str]]:
    """
    Query ChromaDB for `theme` across all `tickers` in a single call,
    then bucket results by ticker.

    Returns a dict {ticker: [chunk_text, ...]} with up to `n_per_ticker`
    chunks per company.  Tickers with no indexed data are mapped to [].
    """
    results = qe.query(
        question=theme,
        ticker=tickers,
        section="risk_factors",
        n_results=n_per_ticker * len(tickers),
    )

    buckets: dict[str, list[str]] = {t: [] for t in tickers}
    for r in results:
        t = r["ticker"].upper()
        if t in buckets and len(buckets[t]) < n_per_ticker:
            buckets[t].append(r["chunk"][:MAX_CHUNK_CHARS])

    return buckets


# ------------------------------------------------------------------ #
#  Prompt construction                                                #
# ------------------------------------------------------------------ #

def _build_comparison_prompt(
    tickers: list[str],
    theme: str,
    chunks_by_ticker: dict[str, list[str]],
) -> str:
    """
    Assemble the cross-ticker comparison prompt.

    The Relative Risk Score is defined relative to tickers[0] and tickers[1]:
      positive (+1 to +50): tickers[0] is MORE exposed to the theme.
      negative (-1 to -50): tickers[1] is MORE exposed.
      zero (0)            : exposure is approximately equal.
    """
    ticker_a = tickers[0]
    ticker_b = tickers[1]
    extra    = tickers[2:]

    # Build the filing excerpt blocks
    section_parts: list[str] = []
    for t in tickers:
        chunks = chunks_by_ticker.get(t, [])
        if chunks:
            joined = "\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks))
        else:
            joined = "(No indexed filings found for this ticker.)"
        section_parts.append(f"=== {t} — SEC RISK FACTOR EXCERPTS ===\n{joined}")

    all_sections = "\n\n".join(section_parts)

    # Build the schema with actual ticker names
    unique_risks_schema = "\n".join(
        f'  "{t}": ["<risk unique to {t} on this theme>"]' for t in tickers
    )

    extra_note = (
        f" Additional tickers for unique-risk analysis: {', '.join(extra)}."
        if extra else ""
    )

    return f"""You are a senior equity research analyst performing competitive due diligence.

THEME: "{theme}"
PRIMARY COMPARISON: {ticker_a} vs {ticker_b}{extra_note}

Below are the most semantically relevant excerpts from each company's most
recent 10-K Risk Factor section on the theme of "{theme}".

{all_sections}

---

Your task: analyse how each company is exposed to "{theme}" and compare them.

Respond with ONLY a valid JSON object. No markdown fences. No explanation outside the JSON.
Use this exact schema:

{{
  "unique_risks": {{
{unique_risks_schema}
  }},
  "shared_risks": [
    {{
      "theme_label":    "<short label for this shared risk sub-theme>",
      "{ticker_a}_language": "<key phrase or sentence from {ticker_a}'s filing>",
      "{ticker_b}_language": "<key phrase or sentence from {ticker_b}'s filing>",
      "more_exposed":   "{ticker_a} | {ticker_b} | equal",
      "intensity_note": "<one sentence explaining the language difference>"
    }}
  ],
  "relative_risk_score": <integer -50 to +50. Positive = {ticker_a} is more exposed to "{theme}". Negative = {ticker_b} is more exposed. 0 = equal.>,
  "verdict": "<two sentences: who carries more risk on this theme and the key differentiator>"
}}

Rules:
- "unique_risks" must include only risks clearly mentioned by one company but
  NOT the other. Keep each item to one concise sentence.
- "shared_risks" covers themes BOTH companies disclose. Focus on how language
  INTENSITY differs (e.g. "may impact" vs "will materially adversely affect").
- "relative_risk_score" should reflect both the number and severity of risks.
  A score of ±10 means slightly more exposed; ±30 means significantly more;
  ±50 means one company has a vastly more acute exposure on this theme.
- Use only the filing excerpts provided. Do not inject outside knowledge."""


# ------------------------------------------------------------------ #
#  Gemini call                                                        #
# ------------------------------------------------------------------ #

def _call_gemini(
    client: genai.Client,
    prompt: str,
    model: str,
) -> tuple[str, dict[str, Any]]:
    """
    Send the comparison prompt to Gemini and return (raw_text, parsed_dict).

    Raises ValueError on invalid JSON or a missing required key.
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
        raise ValueError(
            f"Gemini returned invalid JSON: {exc}\n\nRaw response:\n{raw}"
        ) from exc

    required = {"unique_risks", "shared_risks", "relative_risk_score", "verdict"}
    missing  = required - parsed.keys()
    if missing:
        raise ValueError(
            f"Gemini response missing required keys: {missing}\n\nRaw:\n{raw}"
        )

    return raw, parsed


# ------------------------------------------------------------------ #
#  Display                                                            #
# ------------------------------------------------------------------ #

def _print_comparison(
    tickers: list[str],
    theme: str,
    parsed: dict[str, Any],
) -> None:
    """Pretty-print the cross-ticker comparison report."""
    width    = 72
    ticker_a = tickers[0]
    ticker_b = tickers[1]
    score    = int(parsed.get("relative_risk_score", 0))

    if score > 0:
        score_label = f"{ticker_a} MORE EXPOSED  (+{score})"
    elif score < 0:
        score_label = f"{ticker_b} MORE EXPOSED  ({score})"
    else:
        score_label = f"EQUAL EXPOSURE  (0)"

    print("\n" + "=" * width)
    print(f"  COMPETITIVE INTELLIGENCE — {' vs '.join(tickers)}")
    print(f"  THEME: {theme}")
    print("=" * width)

    print(f"\n  RELATIVE RISK SCORE : {score_label}  [range: -50 to +50]")
    print(f"\n  VERDICT\n  {parsed.get('verdict', '')}\n")

    # Unique risks
    unique = parsed.get("unique_risks", {})
    print(f"  {'─' * (width - 2)}")
    print(f"  UNIQUE RISKS (per company)\n")
    for t in tickers:
        items = unique.get(t, [])
        if items:
            print(f"  [{t}]")
            for item in items:
                print(f"    • {item}")
        else:
            print(f"  [{t}]  — none identified")
        print()

    # Shared risks
    shared = parsed.get("shared_risks", [])
    print(f"  {'─' * (width - 2)}")
    print(f"  SHARED RISKS — Language Intensity Comparison\n")
    if not shared:
        print(f"  None identified for this theme.\n")
    else:
        for i, item in enumerate(shared, 1):
            print(f"  [{i}] {item.get('theme_label', '')}")
            print(f"      {ticker_a}: \"{item.get(f'{ticker_a}_language', item.get('ticker_a_language', ''))}\"")
            print(f"      {ticker_b}: \"{item.get(f'{ticker_b}_language', item.get('ticker_b_language', ''))}\"")
            print(f"      More exposed : {item.get('more_exposed', '?')}")
            print(f"      Note         : {item.get('intensity_note', '')}")
            print()

    print("=" * width)


# ------------------------------------------------------------------ #
#  Orchestrator                                                       #
# ------------------------------------------------------------------ #

def compare_tickers(
    tickers: list[str],
    theme: str,
    n_chunks: int = DEFAULT_CHUNKS,
    model: str = DEFAULT_MODEL,
    show: bool = True,
) -> dict[str, Any]:
    """
    Run a cross-ticker comparison for `tickers` on `theme`.

    Parameters
    ----------
    tickers  : Two or more ticker symbols (first two are primary comparison).
    theme    : The topic to compare, e.g. "China Export Controls".
    n_chunks : Chunks per ticker fed to the prompt (default 4).
    model    : Gemini model name.
    show     : Print the formatted report to stdout (default True).

    Returns the parsed Gemini response dict.
    """
    if len(tickers) < 2:
        raise ValueError("At least two tickers are required for comparison.")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env")

    tickers = [t.upper() for t in tickers]
    log.info("Comparing %s on theme: '%s'", " vs ".join(tickers), theme)

    # -- 1. Initialise QueryEngine --
    qe = QueryEngine()

    if qe.indexed_count == 0:
        raise ValueError(
            "ChromaDB is empty. Run 'sovereign.py embed' first "
            "to index the SEC filings before comparing tickers."
        )

    # -- 2. Fetch relevant chunks per ticker --
    log.info("Fetching up to %d chunks per ticker from ChromaDB...", n_chunks)
    chunks_by_ticker = _fetch_chunks_by_ticker(qe, tickers, theme, n_chunks)

    for t, chunks in chunks_by_ticker.items():
        if not chunks:
            log.warning(
                "No chunks found for %s — make sure its filings are "
                "preprocessed and embedded.", t,
            )
        else:
            log.info("  %s: %d chunk(s) retrieved.", t, len(chunks))

    # -- 3. Build prompt and call Gemini --
    prompt = _build_comparison_prompt(tickers, theme, chunks_by_ticker)
    client = genai.Client(api_key=api_key)

    log.info("Calling Gemini (%s, temperature=0.2)...", model)
    _, parsed = _call_gemini(client, prompt, model)

    log.info(
        "Comparison complete — score=%s, shared_risks=%d",
        parsed.get("relative_risk_score"),
        len(parsed.get("shared_risks", [])),
    )

    # -- 4. Display --
    if show:
        _print_comparison(tickers, theme, parsed)

    return parsed


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Cross-Ticker Competitive Intelligence: compare two or more companies "
            "on a given theme using their SEC filing risk factors."
        )
    )
    parser.add_argument(
        "tickers",
        type=str,
        nargs="+",
        metavar="TICKER",
        help="Two or more ticker symbols e.g. NVDA AMD INTC",
    )
    parser.add_argument(
        "--theme",
        type=str,
        required=True,
        help='Theme to compare on, e.g. "China Export Controls".',
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_CHUNKS,
        metavar="N",
        help=f"Filing chunks per ticker fed to Gemini (default: {DEFAULT_CHUNKS}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL}).",
    )
    args = parser.parse_args()

    if len(args.tickers) < 2:
        parser.error("Provide at least two tickers for comparison.")

    try:
        compare_tickers(
            tickers=args.tickers,
            theme=args.theme,
            n_chunks=args.n,
            model=args.model,
            show=True,
        )
    except (ValueError, EnvironmentError) as exc:
        log.error("%s", exc)
        sys.exit(1)
