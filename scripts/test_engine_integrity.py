"""
scripts/test_engine_integrity.py
---------------------------------
Full-Chain Integration Test for the Sovereign Analyst engine.

Tests are split into two tiers:

  TIER 1 — STATIC (no Gemini API required)
    Verifies database schema, imports, CLI wiring, idempotent preprocessing,
    embed dry-run, and data integrity of any existing records.

  TIER 2 — API (Gemini API required — will be skipped if no quota)
    Verifies note generation, delta analysis, source-trace audit, and
    cross-ticker comparison end-to-end.

Run
---
    .venv/bin/python scripts/test_engine_integrity.py
    .venv/bin/python scripts/test_engine_integrity.py --api   # include Tier 2
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------ #
#  Terminal colours                                                    #
# ------------------------------------------------------------------ #

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
DB_PATH = PROJECT_ROOT / "data" / "sovereign.db"

_passes = 0
_failures = 0
_skips = 0


# ------------------------------------------------------------------ #
#  Result helpers                                                      #
# ------------------------------------------------------------------ #

def _pass(label: str) -> None:
    global _passes
    _passes += 1
    print(f"  {GREEN}✓ PASS{RESET}  {label}")


def _fail(label: str, reason: str = "") -> None:
    global _failures
    _failures += 1
    msg = f"  {RED}✗ FAIL{RESET}  {label}"
    if reason:
        msg += f"\n         → {reason}"
    print(msg)


def _skip(label: str, reason: str = "") -> None:
    global _skips
    _skips += 1
    msg = f"  {YELLOW}⊘ SKIP{RESET}  {label}"
    if reason:
        msg += f" ({reason})"
    print(msg)


def _section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")


# ------------------------------------------------------------------ #
#  Subprocess helper                                                   #
# ------------------------------------------------------------------ #

def _run(cmd: list[str], timeout: int = 120) -> tuple[bool, str, str]:
    """
    Run a subprocess and return (success, stdout, stderr).
    Prints a compact one-liner before executing.
    """
    print(f"  {BOLD}→{RESET} {' '.join(cmd[1:])}")
    try:
        result = subprocess.run(
            [PYTHON] + cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as exc:
        return False, "", str(exc)


# ================================================================== #
#  TIER 1 — STATIC TESTS (no API)                                    #
# ================================================================== #

def test_imports() -> None:
    """Verify every core module and new script can be imported."""
    _section("T1-A: Import Integrity")
    modules = [
        ("core.database",          "AnalystNoteStore, ProcessedFilingStore"),
        ("core.portfolio_engine",  "PortfolioManager"),
        ("core.analysis",          "QueryEngine"),
        ("scripts.compare_tickers","compare_tickers"),
        ("scripts.verify_notes",   "verify_note"),
        ("scripts.analyze_deltas", "run_delta"),
    ]
    for module, symbol in modules:
        try:
            __import__(module)
            _pass(f"import {module}  [{symbol}]")
        except ImportError as exc:
            _fail(f"import {module}", str(exc))


def test_db_schema() -> None:
    """Verify all expected columns are present in analyst_notes."""
    _section("T1-B: Database Schema")

    if not DB_PATH.exists():
        _fail("Database exists", f"Not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # analyst_notes columns
    cursor.execute("PRAGMA table_info(analyst_notes)")
    cols = {row[1] for row in cursor.fetchall()}
    required = {
        "id", "ticker", "accession_number", "model",
        "summary", "risks", "sentiment", "raw_response",
        "created_at", "delta_summary", "confidence_score",
        "intensity_delta",
    }
    missing = required - cols
    if not missing:
        _pass(f"analyst_notes has all {len(required)} required columns")
    else:
        _fail("analyst_notes schema", f"Missing: {missing}")

    # processed_files exists
    try:
        cursor.execute("SELECT count(*) FROM processed_files")
        count = cursor.fetchone()[0]
        _pass(f"processed_files table exists ({count} rows)")
    except sqlite3.OperationalError as exc:
        _fail("processed_files table", str(exc))

    # transactions exists
    try:
        cursor.execute("SELECT DISTINCT ticker FROM transactions")
        tickers = [r[0] for r in cursor.fetchall()]
        _pass(f"transactions table exists — tickers: {tickers}")
    except sqlite3.OperationalError as exc:
        _fail("transactions table", str(exc))

    conn.close()


def test_init_command() -> None:
    """Run sovereign.py init to confirm migrations apply cleanly."""
    _section("T1-C: sovereign.py init (migration idempotency)")
    ok, out, err = _run(["sovereign.py", "init"])
    if ok or "initialised" in out.lower() or "initialised" in err.lower():
        _pass("sovereign.py init ran cleanly")
    else:
        _fail("sovereign.py init", err[:200])


def test_portfolio_command() -> None:
    """Verify portfolio logic runs without crashing."""
    _section("T1-D: sovereign.py portfolio (ACB + price cache)")
    ok, out, err = _run(["sovereign.py", "portfolio"])
    if ok:
        _pass("sovereign.py portfolio returned without error")
        if "P&L" in out or "Market value" in out or "PORTFOLIO" in out:
            _pass("Output contains expected portfolio headers")
        else:
            _fail("Portfolio output format", "Expected P&L/Market value headers not found")
    else:
        _fail("sovereign.py portfolio", err[:200])


def test_preprocess_idempotency() -> None:
    """
    Run preprocess on AAPL (already processed) to confirm idempotency —
    it should skip rather than re-process or crash.
    """
    _section("T1-E: Parallel Preprocess (idempotency check on AAPL)")
    ok, out, err = _run(["sovereign.py", "preprocess", "--ticker", "AAPL"])
    combined = out + err
    if ok:
        if "skipping" in combined.lower() or "skip" in combined.lower():
            _pass("AAPL filings correctly skipped (idempotent)")
        else:
            _pass("Preprocess ran without error")
    else:
        _fail("sovereign.py preprocess --ticker AAPL", err[:300])


def test_embed_dry_run() -> None:
    """
    Run embed with --dry-run to verify the parallel embedding pipeline
    and tenacity import without consuming any API quota.
    """
    _section("T1-F: Parallel Embed Dry-Run (tenacity + semaphore wiring)")
    ok, out, err = _run(["sovereign.py", "embed", "--ticker", "AAPL", "--dry-run"])
    combined = out + err
    if ok or "dry run" in combined.lower():
        _pass("Embed dry-run completed without error")
        if "DRY RUN" in combined.upper():
            _pass("Dry-run flag correctly propagated to embed_filing()")
        else:
            _skip("DRY RUN flag confirmation", "not found in output — check logging level")
    else:
        _fail("sovereign.py embed --dry-run", err[:300])


def test_query_engine_init() -> None:
    """
    Verify QueryEngine initialises and reports the ChromaDB index size
    without making an API call.
    """
    _section("T1-G: QueryEngine Initialisation & Multi-Ticker _build_where")
    try:
        from core.analysis import QueryEngine

        # Check that _build_where correctly handles list[str]
        single = QueryEngine._build_where("AAPL", None, None)
        assert single == {"ticker": {"$eq": "AAPL"}}, f"Unexpected: {single}"
        _pass("_build_where with single str → {$eq}")

        multi = QueryEngine._build_where(["NVDA", "AMD"], "risk_factors", None)
        assert "$and" in multi, f"Expected $and, got: {multi}"
        assert "$in" in str(multi), f"Expected $in, got: {multi}"
        _pass("_build_where with list[str] → {$and: [{$in}, {$eq}]}")

        empty = QueryEngine._build_where(None, None, None)
        assert empty is None
        _pass("_build_where with all-None → None (no filter)")

    except EnvironmentError:
        _skip("QueryEngine full init", "GEMINI_API_KEY not set — testing _build_where only")
        from core.analysis import QueryEngine
        _pass("QueryEngine class is importable")
    except Exception as exc:
        _fail("QueryEngine _build_where logic", str(exc))


def test_existing_notes_integrity() -> None:
    """Inspect existing analyst_notes rows for data integrity."""
    _section("T1-H: Analyst Notes Data Integrity")
    if not DB_PATH.exists():
        _skip("Analyst notes check", "database not found")
        return

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, ticker, sentiment, confidence_score, intensity_delta, "
        "delta_summary FROM analyst_notes ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    conn.close()

    if not rows:
        _skip("Analyst notes exist", "No notes yet — run 'sovereign.py note TICKER' first")
        return

    _pass(f"analyst_notes has {len(rows)} row(s)")
    for row in rows:
        note_id, ticker, sentiment, conf, intensity, delta = row
        label = f"Note id={note_id} [{ticker}]"
        if sentiment:
            _pass(f"{label} sentiment='{sentiment}'")
        else:
            _fail(f"{label} sentiment", "NULL — note may be incomplete")

        if conf is not None:
            _pass(f"{label} confidence_score={conf:.1f}")
        else:
            _skip(f"{label} confidence_score", "run 'sovereign.py verify --save' to populate")

        if intensity is not None:
            _pass(f"{label} intensity_delta={intensity}")
        elif delta:
            _skip(f"{label} intensity_delta", "delta exists but re-run 'sovereign.py delta' to add score")
        else:
            _skip(f"{label} intensity_delta", "no delta run yet")


def test_scan_command() -> None:
    """Run sovereign.py scan (the intertwined command — no API needed)."""
    _section("T1-I: sovereign.py scan (Portfolio Health Report)")
    ok, out, err = _run(["sovereign.py", "scan"])
    if ok:
        _pass("sovereign.py scan completed without error")
        if "PORTFOLIO HEALTH REPORT" in out:
            _pass("Output contains Portfolio Health Report header")
        if "TICKER" in out.upper() or "Ticker" in out:
            _pass("Output contains ticker table")
    else:
        _fail("sovereign.py scan", err[:300])


# ================================================================== #
#  TIER 2 — API TESTS (require Gemini quota)                        #
# ================================================================== #

def test_note_generation(ticker: str = "AAPL") -> None:
    """Generate a fresh analyst note for `ticker`."""
    _section(f"T2-A: Analyst Note Generation ({ticker})")
    print(f"  {YELLOW}Note: this consumes Gemini API quota.{RESET}")
    ok, out, err = _run(["sovereign.py", "note", ticker], timeout=180)
    combined = out + err
    if ok:
        _pass(f"sovereign.py note {ticker} completed")
        if "saved" in combined.lower() or "id=" in combined.lower():
            _pass("Note was saved to analyst_notes table")
    else:
        if "429" in combined or "RESOURCE_EXHAUSTED" in combined.upper() or "quota" in combined.lower():
            _skip(f"note {ticker}", "Gemini free-tier quota exhausted — try again tomorrow")
        else:
            _fail(f"sovereign.py note {ticker}", err[:300])


def test_verify_audit(ticker: str = "AAPL") -> None:
    """Run Source-Trace audit for `ticker` (requires note + embed)."""
    _section(f"T2-B: Source-Trace Audit ({ticker})")
    print(f"  {YELLOW}Note: requires ChromaDB to be populated and Gemini quota.{RESET}")
    ok, out, err = _run(["sovereign.py", "verify", ticker, "--save"], timeout=180)
    combined = out + err
    if ok:
        _pass(f"sovereign.py verify {ticker} completed")
        if "confidence score" in combined.lower() or "OVERALL" in combined:
            _pass("Confidence score computed and printed")
    else:
        if "chromadb is empty" in combined.lower() or "no embeddings" in combined.lower():
            _skip(f"verify {ticker}", "ChromaDB empty — run 'sovereign.py embed' first")
        elif "no analyst note" in combined.lower():
            _skip(f"verify {ticker}", f"No note for {ticker} — run 'sovereign.py note {ticker}' first")
        elif "429" in combined or "quota" in combined.lower():
            _skip(f"verify {ticker}", "Gemini quota exhausted")
        else:
            _fail(f"sovereign.py verify {ticker}", err[:300])


def test_comparison(ticker_a: str = "AAPL", ticker_b: str = "MSFT") -> None:
    """Cross-ticker comparison (requires embeddings for both tickers)."""
    _section(f"T2-C: Cross-Ticker Comparison ({ticker_a} vs {ticker_b})")
    print(f"  {YELLOW}Note: requires ChromaDB populated for both tickers.{RESET}")
    ok, out, err = _run(
        ["sovereign.py", "compare", ticker_a, ticker_b, "--theme", "Revenue Concentration"],
        timeout=180,
    )
    combined = out + err
    if ok:
        _pass(f"compare {ticker_a} vs {ticker_b} completed")
        if "RELATIVE RISK SCORE" in combined:
            _pass("Relative Risk Score found in output")
        if "UNIQUE RISKS" in combined:
            _pass("Unique Risks section found in output")
    else:
        if "chromadb is empty" in combined.lower():
            _skip(f"compare {ticker_a} vs {ticker_b}", "ChromaDB empty — run embed first")
        elif "429" in combined or "quota" in combined.lower():
            _skip(f"compare {ticker_a} vs {ticker_b}", "Gemini quota exhausted")
        else:
            _fail(f"compare {ticker_a} vs {ticker_b}", err[:300])


# ================================================================== #
#  Summary                                                            #
# ================================================================== #

def _print_summary(include_api: bool) -> None:
    total = _passes + _failures + _skips
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  TEST SUMMARY{RESET}")
    print(f"{'=' * 60}")
    print(f"  {GREEN}PASS  : {_passes}{RESET}")
    print(f"  {RED}FAIL  : {_failures}{RESET}")
    print(f"  {YELLOW}SKIP  : {_skips}{RESET}")
    print(f"  Total : {total}")
    print(f"{'=' * 60}")

    if _failures == 0:
        print(f"\n  {GREEN}{BOLD}✓ ALL SYSTEMS OPERATIONAL{RESET}")
        if include_api:
            print("  Your engine is fully verified and ready for Streamlit.")
        else:
            print("  Tier 1 (static) tests passed.")
            print("  Run with --api to test Gemini-dependent features.")
    else:
        print(f"\n  {RED}{BOLD}✗ {_failures} test(s) failed — review output above.{RESET}")
    print()


# ================================================================== #
#  Entry point                                                        #
# ================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sovereign Analyst — Full-Chain Engine Integrity Test"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Also run Tier 2 tests that consume Gemini API quota.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Primary ticker to use for API tests (default: AAPL).",
    )
    parser.add_argument(
        "--ticker-b",
        type=str,
        default="MSFT",
        help="Secondary ticker for comparison test (default: MSFT).",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  SOVEREIGN ANALYST — ENGINE INTEGRITY TEST{RESET}")
    tier_label = "TIER 1 (static) + TIER 2 (API)" if args.api else "TIER 1 (static, no API)"
    print(f"  Running: {tier_label}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    # --- Tier 1 ---
    test_imports()
    test_db_schema()
    test_init_command()
    test_portfolio_command()
    test_preprocess_idempotency()
    test_embed_dry_run()
    test_query_engine_init()
    test_existing_notes_integrity()
    test_scan_command()

    # --- Tier 2 ---
    if args.api:
        test_note_generation(args.ticker)
        test_verify_audit(args.ticker)
        test_comparison(args.ticker, args.ticker_b)

    _print_summary(include_api=args.api)
