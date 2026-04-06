"""
sovereign.py
------------
Central CLI entry point for the Sovereign Analyst project.

Every pipeline script is accessible from here as a subcommand.
Imports are deferred into each subcommand handler so startup is fast
regardless of which command is used.

Usage
-----
    python sovereign.py init
    python sovereign.py portfolio
    python sovereign.py fetch-filings AAPL [--annual 1] [--quarterly 4]
    python sovereign.py preprocess [--ticker AAPL]
    python sovereign.py embed [--ticker AAPL] [--dry-run]
    python sovereign.py signals
    python sovereign.py note AAPL [--show] [--model gemini-2.0-flash]
    python sovereign.py delta AAPL [--year-new 2025] [--year-old 2024] [--show]
    python sovereign.py audit AAPL
    python sovereign.py query "What are the liquidity risks?" [--ticker AAPL] [--section risk_factors] [--n 5]
    python sovereign.py scan
    python sovereign.py verify AAPL [--note-id 3] [--save]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------------ #
#  Subcommand handlers                                                #
# ------------------------------------------------------------------ #

def cmd_init(args: argparse.Namespace) -> None:
    """Initialise all SQLite tables."""
    from scripts.init_db import init_db
    init_db()


def cmd_portfolio(args: argparse.Namespace) -> None:
    """Print the current portfolio holdings and summary."""
    from core.portfolio_engine import PortfolioManager
    pm = PortfolioManager()
    result = pm.run_pipeline()
    holdings = result["holdings"]
    summary  = result["summary"]

    print("\n" + "=" * 75)
    print("  PORTFOLIO HOLDINGS")
    print("=" * 75)
    if holdings.empty:
        print("  No holdings found. Add transactions first.")
    else:
        for ticker, row in holdings.iterrows():
            pnl_pct = (
                row["unrealised_pnl"] / row["capital_invested"] * 100
                if row.get("capital_invested") else 0
            )
            print(
                f"  {ticker:<6}  qty={row['quantity']:.4f}  "
                f"avg_cost=${row['avg_cost']:.2f}  "
                f"price=${row.get('current_price', 0):.2f}  "
                f"mkt=${row.get('market_value', 0):,.2f}  "
                f"P&L={pnl_pct:+.2f}%"
            )

    print("\n" + "-" * 75)
    s = summary.iloc[0]
    print(f"  Capital invested : ${s['total_capital_invested']:>12,.2f}")
    print(f"  Market value     : ${s['total_market_value']:>12,.2f}")
    print(f"  Unrealised P/L   : ${s['total_unrealised_pnl']:>12,.2f}  ({s['unrealised_pnl_pct']:+.2f}%)")
    print(f"  Realised P/L     : ${s['total_realised_pnl']:>12,.2f}")
    print(f"  Dividends        : ${s['total_dividends']:>12,.2f}")
    print(f"  Total P/L        : ${s['total_pnl_all_in']:>12,.2f}")
    print("=" * 75)


def cmd_fetch_filings(args: argparse.Namespace) -> None:
    """Download 10-K/10-Q filings for a ticker from SEC EDGAR."""
    from scripts.fetch_filings import fetch_filings_for_ticker, load_environment, get_user_agent
    from core.database import FilingMetadataStore

    load_environment()
    user_agent = get_user_agent()
    store = FilingMetadataStore()
    paths = fetch_filings_for_ticker(
        ticker=args.ticker.upper(),
        user_agent=user_agent,
        store=store,
        n_annual=args.annual,
        n_quarterly=args.quarterly,
    )
    print(f"\nFetched for {args.ticker.upper()}: {paths}")


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Preprocess raw SEC filings into clean Markdown + JSON chunks."""
    from scripts.preprocess_filings import main as preprocess_main
    preprocess_main(ticker_filter=args.ticker)


def cmd_embed(args: argparse.Namespace) -> None:
    """Generate Gemini embeddings and upsert into ChromaDB."""
    from scripts.embed_filings import main as embed_main
    embed_main(ticker_filter=args.ticker, dry_run=args.dry_run)


def cmd_signals(args: argparse.Namespace) -> None:
    """Fetch latest news and transcript signals for top holdings."""
    from scripts.fetch_signals import main as signals_main
    signals_main()


def cmd_note(args: argparse.Namespace) -> None:
    """Generate an AI analyst note for a ticker."""
    from scripts.generate_analyst_notes import generate_note
    generate_note(ticker=args.ticker.upper(), model=args.model, show=args.show)


def cmd_delta(args: argparse.Namespace) -> None:
    """Run a Surgical Delta between two annual 10-K Risk Factor sections."""
    from scripts.analyze_deltas import run_delta
    run_delta(
        ticker=args.ticker.upper(),
        year_new=args.year_new,
        year_old=args.year_old,
        model=args.model,
        show=args.show,
    )


def cmd_scan(args: argparse.Namespace) -> None:
    """Portfolio Health Report — one row per ticker, no API calls."""
    import json
    from datetime import date

    from core.database import AnalystNoteStore
    from core.portfolio_engine import PortfolioManager

    pm    = PortfolioManager()
    store = AnalystNoteStore()

    # -- Collect all distinct tickers ever traded --
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM transactions ORDER BY ticker"
        ).fetchall()
    tickers = [r[0] for r in rows]

    if not tickers:
        print("No transactions found. Add some trades first.")
        return

    # -- Portfolio data (current holdings) --
    holdings_df, _, _ = pm.recalculate_portfolio()
    pnl_map: dict[str, float | None] = {}
    for ticker in tickers:
        if ticker in holdings_df.index:
            row  = holdings_df.loc[ticker]
            cap  = row.get("capital_invested") or 0.0
            pnl  = row.get("unrealised_pnl") or 0.0
            pnl_map[ticker] = (pnl / cap * 100) if cap else None
        else:
            pnl_map[ticker] = None  # fully exited or no equity position

    # -- Build report rows --
    width = 79
    today = date.today().strftime("%Y-%m-%d")

    print("\n" + "=" * width)
    print(f"  PORTFOLIO HEALTH REPORT{today:>{width - 25}}")
    print("=" * width)
    print(f"  {'Ticker':<7}  {'P&L%':<9}  {'Sentiment':<12}  {'Conf':>5}  Delta Verdict")
    print(f"  {'------':<7}  {'--------':<9}  {'-----------':<12}  {'-----':>5}  {'-------------------'}")

    low_conf_tickers:  list[str] = []
    neg_sent_tickers:  list[str] = []

    for ticker in tickers:
        pnl_pct = pnl_map[ticker]
        pnl_str = f"{pnl_pct:+.2f}%" if pnl_pct is not None else "--"

        note = store.get_latest_note(ticker)
        if note:
            sentiment = note.get("sentiment") or "unknown"
            conf_raw  = note.get("confidence_score")
            conf_str  = f"{int(conf_raw)}/100" if conf_raw is not None else "--"
            if conf_raw is not None and conf_raw < 60:
                low_conf_tickers.append(ticker)
            if sentiment and sentiment.lower() == "negative":
                neg_sent_tickers.append(ticker)
        else:
            sentiment = "no note"
            conf_str  = "--"

        delta = store.get_latest_delta(ticker)
        if delta:
            try:
                ds   = json.loads(delta["delta_summary"] or "{}")
                verdict = ds.get("verdict") or ds.get("overall_verdict") or "—"
            except (json.JSONDecodeError, TypeError):
                verdict = "—"
            verdict_preview = verdict[:36].strip()
        else:
            verdict_preview = "no delta"

        print(
            f"  {ticker:<7}  {pnl_str:<9}  {sentiment:<12}  {conf_str:>5}  {verdict_preview}"
        )

    print("=" * width)
    low_label = ", ".join(low_conf_tickers) if low_conf_tickers else "none"
    neg_label = ", ".join(neg_sent_tickers) if neg_sent_tickers else "none"
    print(f"  Tickers with LOW confidence (<60) : {low_label}")
    print(f"  Tickers with NEGATIVE sentiment   : {neg_label}")
    print("=" * width)


def cmd_verify(args: argparse.Namespace) -> None:
    """Source-Trace audit: verify AI risk claims against ChromaDB."""
    from scripts.verify_notes import verify_note
    verify_note(ticker=args.ticker.upper(), note_id=args.note_id, save=args.save)


def cmd_audit(args: argparse.Namespace) -> None:
    """Audit the Sovereign View for a ticker and print the result."""
    from scripts.audit_sovereign_view import audit
    audit(ticker=args.ticker.upper())


def cmd_query(args: argparse.Namespace) -> None:
    """Semantic search over embedded SEC filing chunks."""
    from core.analysis import QueryEngine
    qe = QueryEngine()

    if qe.indexed_count == 0:
        print(
            "ChromaDB collection is empty. "
            "Run 'sovereign.py embed' first."
        )
        sys.exit(1)

    results = qe.query(
        question=args.question,
        ticker=args.ticker,
        section=args.section,
        n_results=args.n,
    )

    print(f"\nTop {len(results)} results for: \"{args.question}\"\n")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['ticker']}  {r['form_type']}  {r['year']}  "
              f"section={r['section']}  distance={r['distance']:.4f}")
        print(f"      {r['chunk'][:200].strip()}...")
        print()


# ------------------------------------------------------------------ #
#  Argument parser                                                    #
# ------------------------------------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sovereign",
        description="Sovereign Analyst — unified CLI for portfolio research.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sovereign.py init
  python sovereign.py portfolio
  python sovereign.py fetch-filings AAPL --annual 2 --quarterly 4
  python sovereign.py preprocess --ticker AAPL
  python sovereign.py embed --ticker AAPL
  python sovereign.py signals
  python sovereign.py note AAPL --show
  python sovereign.py delta AAPL --year-new 2025 --year-old 2024 --show
  python sovereign.py audit AAPL
  python sovereign.py scan
  python sovereign.py verify AAPL --save
  python sovereign.py query "What are the liquidity risks?" --ticker AAPL
        """,
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # init
    sub.add_parser("init", help="Initialise all SQLite tables.")

    # portfolio
    sub.add_parser("portfolio", help="Print current holdings and P/L summary.")

    # fetch-filings
    p_fetch = sub.add_parser("fetch-filings", help="Download 10-K/10-Q from SEC EDGAR.")
    p_fetch.add_argument("ticker", type=str, help="Ticker symbol e.g. AAPL")
    p_fetch.add_argument("--annual",    type=int, default=1, metavar="N", help="Number of 10-Ks to download (default: 1)")
    p_fetch.add_argument("--quarterly", type=int, default=4, metavar="N", help="Number of 10-Qs to download (default: 4)")

    # preprocess
    p_pre = sub.add_parser("preprocess", help="Preprocess raw filings into Markdown + chunks.")
    p_pre.add_argument("--ticker", type=str, default=None, metavar="TICKER", help="Restrict to this ticker.")

    # embed
    p_embed = sub.add_parser("embed", help="Embed filing chunks into ChromaDB.")
    p_embed.add_argument("--ticker",  type=str, default=None, metavar="TICKER", help="Restrict to this ticker.")
    p_embed.add_argument("--dry-run", action="store_true", help="Walk files without calling the API.")

    # signals
    sub.add_parser("signals", help="Fetch latest news & transcript signals for top holdings.")

    # note
    p_note = sub.add_parser("note", help="Generate an AI analyst note for a ticker.")
    p_note.add_argument("ticker", type=str, help="Ticker symbol e.g. AAPL")
    p_note.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini model (default: gemini-2.0-flash)")
    p_note.add_argument("--show",  action="store_true", help="Print the note to stdout.")

    # delta
    p_delta = sub.add_parser("delta", help="Surgical Delta between two annual 10-K Risk Factor sections.")
    p_delta.add_argument("ticker", type=str, help="Ticker symbol e.g. AAPL")
    p_delta.add_argument("--year-new", type=int, default=None, metavar="YEAR", help="Year of newer filing (default: most recent).")
    p_delta.add_argument("--year-old", type=int, default=None, metavar="YEAR", help="Year of older filing (default: second most recent).")
    p_delta.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini model (default: gemini-2.0-flash)")
    p_delta.add_argument("--show",  action="store_true", help="Print the delta report to stdout.")

    # audit
    p_audit = sub.add_parser("audit", help="Audit the Sovereign View for a ticker.")
    p_audit.add_argument("ticker", type=str, help="Ticker symbol e.g. AAPL")

    # scan
    sub.add_parser("scan", help="Portfolio Health Report: sentiment, confidence & delta for all tickers.")

    # verify
    p_verify = sub.add_parser("verify", help="Source-Trace audit: verify AI risk claims against ChromaDB.")
    p_verify.add_argument("ticker", type=str, help="Ticker symbol e.g. AAPL")
    p_verify.add_argument(
        "--note-id", type=int, default=None, metavar="ID",
        help="Specific analyst_notes row id to audit (default: latest note).",
    )
    p_verify.add_argument(
        "--save", action="store_true",
        help="Write the computed confidence_score back to analyst_notes.",
    )

    # query
    p_query = sub.add_parser("query", help="Semantic search over embedded filing chunks.")
    p_query.add_argument("question", type=str, help="Natural language question.")
    p_query.add_argument("--ticker",  type=str, default=None, metavar="TICKER", help="Restrict to this ticker.")
    p_query.add_argument("--section", type=str, default=None, choices=["risk_factors", "mda"], help="Restrict to section.")
    p_query.add_argument("--n", type=int, default=5, metavar="N", help="Number of results (default: 5).")

    return parser


# ------------------------------------------------------------------ #
#  Dispatch                                                           #
# ------------------------------------------------------------------ #

_HANDLERS = {
    "init":          cmd_init,
    "portfolio":     cmd_portfolio,
    "fetch-filings": cmd_fetch_filings,
    "preprocess":    cmd_preprocess,
    "embed":         cmd_embed,
    "signals":       cmd_signals,
    "note":          cmd_note,
    "delta":         cmd_delta,
    "audit":         cmd_audit,
    "scan":          cmd_scan,
    "verify":        cmd_verify,
    "query":         cmd_query,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    handler = _HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()
