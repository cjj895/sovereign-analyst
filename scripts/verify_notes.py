"""
scripts/verify_notes.py
------------------------
Source-Trace Audit: verify every AI-generated risk claim against the raw
SEC filing text stored in ChromaDB.

For each risk string in an analyst note, the script queries ChromaDB for
the single most semantically similar chunk and converts the cosine distance
into a 0–100 Confidence Score. The average across all risks becomes the note's
overall Confidence Score and is optionally written back to the analyst_notes
table.

Audit Logic
-----------
FOR each risk_string in note["risks"]:
    1. Query ChromaDB (ticker-scoped, section=risk_factors, n_results=1)
    2. claim_score = max(0, min(100, round((1 - distance) * 100)))
       HIGH   80–100  claim directly traceable to filing text
       MEDIUM 60–79   claim partially backed, likely inferred
       LOW     0–59   claim lacks direct textual backing
AGGREGATE: confidence_score = mean(all claim_scores)

Run
---
    .venv/bin/python scripts/verify_notes.py AAPL
    .venv/bin/python scripts/verify_notes.py AAPL --note-id 3
    .venv/bin/python scripts/verify_notes.py AAPL --save
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.analysis import QueryEngine
from core.database import AnalystNoteStore

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
#  Scoring constants                                                   #
# ------------------------------------------------------------------ #

THRESHOLD_HIGH   = 80
THRESHOLD_MEDIUM = 60

PREVIEW_CHARS = 300  # how many chars of the best chunk to show


# ------------------------------------------------------------------ #
#  Core audit logic                                                    #
# ------------------------------------------------------------------ #

def _grade(score: int) -> str:
    """Map a 0–100 score to a human-readable grade."""
    if score >= THRESHOLD_HIGH:
        return "HIGH"
    if score >= THRESHOLD_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _trace_claim(
    qe: QueryEngine,
    risk_string: str,
    ticker: str,
) -> dict[str, Any]:
    """
    Query ChromaDB for the single best-matching chunk for a risk claim and
    return a scored trace record.

    Returns
    -------
    dict with keys:
        risk        : original risk string
        best_chunk  : first PREVIEW_CHARS chars of the top-matching chunk
        distance    : cosine distance (0 = identical, 1 = orthogonal)
        claim_score : 0–100 integer
        grade       : "HIGH" | "MEDIUM" | "LOW"

    If ChromaDB is empty (no embeddings indexed), returns claim_score=0.
    """
    if qe.indexed_count == 0:
        log.warning(
            "ChromaDB is empty — cannot trace claim. "
            "Run scripts/embed_filings.py first."
        )
        return {
            "risk":        risk_string,
            "best_chunk":  "(no embeddings indexed)",
            "distance":    1.0,
            "claim_score": 0,
            "grade":       "LOW",
        }

    results = qe.query(
        question=risk_string,
        ticker=ticker,
        section="risk_factors",
        n_results=1,
    )

    if not results:
        return {
            "risk":        risk_string,
            "best_chunk":  "(no results returned)",
            "distance":    1.0,
            "claim_score": 0,
            "grade":       "LOW",
        }

    top      = results[0]
    distance = top["distance"]
    score    = max(0, min(100, round((1 - distance) * 100)))
    chunk_preview = top["chunk"][:PREVIEW_CHARS].strip()

    return {
        "risk":        risk_string,
        "best_chunk":  chunk_preview,
        "distance":    distance,
        "claim_score": score,
        "grade":       _grade(score),
    }


# ------------------------------------------------------------------ #
#  Reporting                                                           #
# ------------------------------------------------------------------ #

def _verdict_text(score: int) -> str:
    grade = _grade(score)
    if grade == "HIGH":
        return "HIGH — claims are well-grounded in filing text"
    if grade == "MEDIUM":
        return "MEDIUM — most claims supported; some may be inferred"
    return "LOW — claims lack direct textual backing (review for hallucination)"


def _print_report(
    ticker: str,
    note_id: int,
    traces: list[dict[str, Any]],
    confidence_score: float,
    saved: bool,
) -> None:
    """Pretty-print the Source-Trace audit result to stdout."""
    width = 70
    print("\n" + "=" * width)
    print(f"  SOURCE-TRACE AUDIT — {ticker}")
    print("=" * width)
    print(f"  Note ID  : {note_id}")
    print(f"  Risks    : {len(traces)}")
    print()

    for i, t in enumerate(traces, 1):
        print(f"  [{i}] {t['risk'][:75]}")
        print(f"      Grade     : {t['grade']} (score={t['claim_score']})")
        print(f"      Distance  : {t['distance']:.4f}")
        print(f"      Best chunk: \"{t['best_chunk'][:120]}...\"")
        print()

    score_int = round(confidence_score)
    print(f"  OVERALL CONFIDENCE SCORE : {score_int} / 100")
    print(f"  VERDICT                  : {_verdict_text(score_int)}")
    if saved:
        print(f"  Saved to analyst_notes   : YES (id={note_id})")
    print("=" * width)


# ------------------------------------------------------------------ #
#  Orchestrator                                                        #
# ------------------------------------------------------------------ #

def verify_note(
    ticker: str,
    note_id: int | None = None,
    save: bool = False,
) -> float:
    """
    Run the Source-Trace audit for `ticker`.

    Parameters
    ----------
    ticker  : Exchange symbol e.g. "NVDA"
    note_id : Specific analyst_notes row to audit.  If None, uses the
              latest standard note for the ticker.
    save    : If True, write the confidence_score back to the database.

    Returns the computed confidence_score (0.0–100.0).
    """
    ticker = ticker.upper()
    store  = AnalystNoteStore()

    # -- Load note --
    if note_id is not None:
        notes = store.get_notes_for_ticker(ticker, limit=50)
        note = next((n for n in notes if n["id"] == note_id), None)
        if note is None:
            raise ValueError(f"No analyst note found for {ticker} with id={note_id}.")
    else:
        note = store.get_latest_note(ticker)
        if note is None:
            raise ValueError(
                f"No analyst note found for {ticker}. "
                f"Run scripts/generate_analyst_notes.py {ticker} first."
            )

    resolved_id = note["id"]

    # -- Parse risks --
    try:
        risks: list[str] = json.loads(note["risks"])
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(
            f"Failed to parse risks JSON for note id={resolved_id}: {exc}"
        ) from exc

    if not risks:
        log.warning("Note id=%d has no risks to audit.", resolved_id)
        return 0.0

    log.info(
        "Auditing note id=%d for %s — %d risk(s) to trace.",
        resolved_id, ticker, len(risks),
    )

    # -- Initialise QueryEngine --
    qe = QueryEngine()

    # -- Trace each claim --
    traces: list[dict[str, Any]] = []
    for risk_str in risks:
        log.info("Tracing: %s", risk_str[:60])
        trace = _trace_claim(qe, risk_str, ticker)
        traces.append(trace)
        log.info(
            "  → grade=%s  score=%d  distance=%.4f",
            trace["grade"], trace["claim_score"], trace["distance"],
        )

    # -- Aggregate --
    scores = [t["claim_score"] for t in traces]
    confidence_score = sum(scores) / len(scores)

    # -- Optionally persist --
    if save:
        updated = store.update_confidence(resolved_id, confidence_score)
        log.info(
            "Confidence score %.1f saved to note id=%d: %s",
            confidence_score, resolved_id, updated,
        )

    # -- Print report --
    _print_report(ticker, resolved_id, traces, confidence_score, saved=save)

    return confidence_score


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Source-Trace Audit: verify analyst note risks against ChromaDB."
    )
    parser.add_argument("ticker", type=str, help="Ticker symbol e.g. NVDA")
    parser.add_argument(
        "--note-id",
        type=int,
        default=None,
        metavar="ID",
        help="Specific analyst_notes row id to audit (default: latest note).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write the computed confidence_score back to the analyst_notes table.",
    )
    args = parser.parse_args()
    try:
        verify_note(ticker=args.ticker, note_id=args.note_id, save=args.save)
    except (ValueError, EnvironmentError) as exc:
        log.error("%s", exc)
        sys.exit(1)
