"""
Audit script: verifies that get_full_context() returns a complete and
correctly structured 'Sovereign View' for a single ticker.

Usage:
    .venv/bin/python scripts/audit_sovereign_view.py [TICKER]

If no TICKER is provided, it defaults to the largest holding by market value.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.portfolio_engine import PortfolioManager


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def audit(ticker: str | None = None) -> None:
    pm = PortfolioManager()

    # Resolve ticker if not provided
    if not ticker:
        pipeline = pm.run_pipeline()
        holdings = pipeline.get("holdings")
        if holdings is None or holdings.empty:
            print("ERROR: No holdings found. Please add transactions first.")
            return
        ticker = (
            holdings.sort_values("market_value", ascending=False)
            .index[0]
        )
        print(f"No ticker specified — using largest holding: {ticker}")

    print(f"\nBuilding Sovereign View for: {ticker}")
    ctx = pm.get_full_context(ticker)

    # ------------------------------------------------------------------ #
    #  1. Position                                                        #
    # ------------------------------------------------------------------ #
    print_section("POSITION")
    pos = ctx.get("position")
    if pos:
        print(f"  Quantity          : {pos['quantity']}")
        print(f"  Avg Cost          : ${pos['avg_cost']:,.4f}")
        print(f"  Capital Invested  : ${pos['capital_invested']:,.2f}")
        print(f"  Current Price     : ${pos['current_price']:,.2f}")
        print(f"  Market Value      : ${pos['market_value']:,.2f}")
        print(f"  Unrealised P/L    : ${pos['unrealised_pnl']:,.2f}  ({pos['unrealised_pnl_pct']:.2f}%)")
    else:
        print(f"  {ticker} is not in current holdings.")

    # ------------------------------------------------------------------ #
    #  2. Filings                                                         #
    # ------------------------------------------------------------------ #
    print_section("FILINGS")
    filings = ctx.get("filings", [])
    if filings:
        for f in filings:
            print(
                f"  [{f['form_type']}]  period={f['period_of_report']}  "
                f"filed={f['filing_date']}  "
                f"size={f['file_size_bytes'] // 1024:,} KB"
            )
    else:
        print(f"  No filings downloaded yet. Run: .venv/bin/python scripts/fetch_filings.py {ticker}")

    # ------------------------------------------------------------------ #
    #  3. News                                                            #
    # ------------------------------------------------------------------ #
    print_section("LATEST NEWS  (up to 10)")
    news = ctx.get("news", [])
    if news:
        for n in news:
            ts = n.get("timestamp", "")[:10]
            title = n.get("title", "N/A")[:70]
            publisher = n.get("publisher", "")
            print(f"  [{ts}]  {publisher:<20}  {title}")
    else:
        print(f"  No news signals. Run: .venv/bin/python scripts/fetch_signals.py")

    # ------------------------------------------------------------------ #
    #  4. Transcripts                                                     #
    # ------------------------------------------------------------------ #
    print_section("TRANSCRIPT EVENTS  (up to 5)")
    transcripts = ctx.get("transcripts", [])
    if transcripts:
        for tr in transcripts:
            ts = tr.get("timestamp", "")[:10]
            if "earnings_date" in tr:
                print(f"  [{ts}]  Earnings date: {tr['earnings_date'][:10]}")
            else:
                title = tr.get("title", "N/A")[:70]
                print(f"  [{ts}]  {title}")
    else:
        print(f"  No transcript signals found for {ticker}.")

    # ------------------------------------------------------------------ #
    #  5. Schema Validation                                               #
    # ------------------------------------------------------------------ #
    print_section("AUDIT CHECKS")
    checks = {
        "ticker present":      bool(ctx.get("ticker")),
        "generated_at present":bool(ctx.get("generated_at")),
        "position is dict or None": ctx.get("position") is None or isinstance(ctx.get("position"), dict),
        "filings is list":     isinstance(ctx.get("filings"), list),
        "news is list":        isinstance(ctx.get("news"), list),
        "transcripts is list": isinstance(ctx.get("transcripts"), list),
    }
    all_passed = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}]  {check}")

    print()
    if all_passed:
        print("  Sovereign View is complete and correctly structured.")
    else:
        print("  WARNING: One or more checks failed. Review the output above.")

    print(f"\n  Generated at: {ctx['generated_at']}")


if __name__ == "__main__":
    ticker_arg = sys.argv[1].upper() if len(sys.argv) > 1 else None
    audit(ticker_arg)
