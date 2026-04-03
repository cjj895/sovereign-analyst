from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from core.database import FilingMetadataStore, SignalStore


@dataclass
class PortfolioConfig:
    """Configuration for portfolio file locations."""

    db_path: Path = Path("data/sovereign.db")
    history_file: Path = Path("data/portfolio_history.csv")
    cache_file: Path = Path("data/.price_cache.json")


class PortfolioManager:
    """
    Manage portfolio with SQLite persistence and manual SPLIT transaction handling.
    
    Supported types:
        buy      - cash decreases, qty increases
        sell     - cash increases, qty decreases (validated against holdings)
        dividend - cash increases, qty unchanged
        income   - non-investment cash inflow
        expense  - non-investment cash outflow
        SPLIT    - multiplies current qty by ratio, divides avg_cost by ratio
    """

    INVESTMENT_TYPES: frozenset[str] = frozenset({"buy", "sell", "dividend", "SPLIT"})
    CACHE_TTL_SECONDS: int = 900  # 15 minutes

    def __init__(
        self,
        db_path: str | Path = "data/sovereign.db",
        history_file: str | Path = "data/portfolio_history.csv",
    ) -> None:
        self.config = PortfolioConfig(
            db_path=Path(db_path),
            history_file=Path(history_file),
        )
        self.snapshot_header: list[str] = [
            "month", "total_capital_invested", "total_market_value",
            "total_unrealised_pnl", "total_realised_pnl", "total_dividends",
            "unrealised_pnl_pct",
        ]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    # ------------------------------------------------------------------ #
    #  DB Operations                                                     #
    # ------------------------------------------------------------------ #

    def add_transaction(
        self,
        date: str,
        tx_type: str,
        asset: str | None = None,
        ticker: str | None = None,
        price: float | None = None,
        quantity: float | None = None,
        amount: float | None = None,
        description: str | None = None,
        ratio: float | None = None,
    ) -> int:
        """Execute a SQL INSERT for a new transaction."""
        sql = """
        INSERT INTO transactions (date, type, asset, ticker, price, quantity, amount, description, ratio)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, (
                date, tx_type.lower(), asset, ticker, price, quantity, amount, description, ratio
            ))
            return cursor.lastrowid

    def load_raw_transactions(self) -> pd.DataFrame:
        """Fetch all rows from the DB in chronological order."""
        with self._connect() as conn:
            df = pd.read_sql_query("SELECT * FROM transactions ORDER BY date ASC, id ASC", conn)
        return df

    # ------------------------------------------------------------------ #
    #  Portfolio Logic                                                   #
    # ------------------------------------------------------------------ #

    def recalculate_portfolio(self) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        """
        Fetch all rows from DB and loop through them to calculate 
        current quantity, avg_cost, and realised P/L.
        
        Handles SPLIT transactions by adjusting quantity and cost.
        Ensures no selling more than owned.
        """
        df = self.load_raw_transactions()
        
        # Running state
        holdings: dict[str, dict] = {}
        realised: dict[str, dict] = {}
        cash_flow = {"income": 0.0, "expense": 0.0}

        for _, row in df.iterrows():
            tx_type = row["type"].lower()
            ticker = row["ticker"]
            
            # Handle non-investment types
            if tx_type == "income":
                cash_flow["income"] += float(row["amount"] or 0)
                continue
            if tx_type == "expense":
                cash_flow["expense"] += abs(float(row["amount"] or 0))
                continue

            if not ticker: continue

            if ticker not in holdings:
                holdings[ticker] = {"asset": row["asset"], "qty": 0.0, "total_cost": 0.0}
            if ticker not in realised:
                realised[ticker] = {"pnl": 0.0, "divs": 0.0}

            qty = abs(float(row["quantity"] or 0))
            price = float(row["price"] or 0)
            amt = float(row["amount"] or 0)

            if tx_type == "buy":
                holdings[ticker]["qty"] += qty
                holdings[ticker]["total_cost"] += qty * price

            elif tx_type == "sell":
                current_qty = holdings[ticker]["qty"]
                # CRUCIAL: Check to ensure I don't sell more than I own
                if qty > current_qty + 1e-9:
                    print(f"⚠️ Warning: Attempted to sell {qty} {ticker}, but only own {current_qty}. Capping at {current_qty}.")
                    qty = current_qty

                if current_qty > 0:
                    avg_cost = holdings[ticker]["total_cost"] / current_qty
                    realised[ticker]["pnl"] += (price - avg_cost) * qty
                    holdings[ticker]["total_cost"] -= avg_cost * qty
                    holdings[ticker]["qty"] -= qty

            elif tx_type == "dividend":
                realised[ticker]["divs"] += abs(amt)

            elif tx_type == "split":
                # CRUCIAL: Multiply current quantity by ratio and divide cost
                ratio = float(row["ratio"] or 1.0)
                if ratio > 0:
                    holdings[ticker]["qty"] *= ratio
                    # total_cost stays same, so avg_cost (total/qty) implicitly divides by ratio
                    print(f"🔄 Applied {ratio}:1 split for {ticker}. New qty: {holdings[ticker]['qty']}")

        # Build DataFrames
        h_list = []
        for t, d in holdings.items():
            if d["qty"] > 1e-9:
                avg = d["total_cost"] / d["qty"]
                h_list.append({
                    "ticker": t, "asset": d["asset"], "quantity": d["qty"], 
                    "avg_cost": avg, "capital_invested": d["total_cost"],
                    "realised_pnl": realised[t]["pnl"], "dividends": realised[t]["divs"]
                })
        
        holdings_df = pd.DataFrame(h_list).set_index("ticker") if h_list else pd.DataFrame()
        
        r_list = [{"ticker": t, **v} for t, v in realised.items() if v["pnl"] != 0 or v["divs"] != 0]
        realised_df = pd.DataFrame(r_list).set_index("ticker") if r_list else pd.DataFrame()

        return holdings_df, realised_df, cash_flow

    # ------------------------------------------------------------------ #
    #  Price Caching & Pipeline                                          #
    # ------------------------------------------------------------------ #

    def _load_price_cache(self) -> dict:
        try: return json.loads(self.config.cache_file.read_text())
        except: return {}

    def _save_price_cache(self, cache: dict):
        self.config.cache_file.write_text(json.dumps(cache, indent=2))

    def get_live_prices(self, tickers: list[str]) -> list[float | None]:
        cache = self._load_price_cache()
        prices = []
        updated = False
        for t in tickers:
            entry = cache.get(t, {})
            fetched_at = datetime.fromisoformat(entry.get("fetched_at", "1970-01-01"))
            if (datetime.now() - fetched_at).total_seconds() < self.CACHE_TTL_SECONDS:
                prices.append(entry["price"])
            else:
                try:
                    p = round(float(yf.Ticker(t).fast_info.last_price), 2)
                    cache[t] = {"price": p, "fetched_at": datetime.now().isoformat()}
                    prices.append(p)
                    updated = True
                except:
                    prices.append(entry.get("price"))
        if updated: self._save_price_cache(cache)
        return prices

    # ------------------------------------------------------------------ #
    #  Filings                                                           #
    # ------------------------------------------------------------------ #

    def fetch_filings_for_top_holdings(
        self,
        n_holdings: int = 5,
        n_quarterly: int = 4,
    ) -> list[dict[str, Any]]:
        """
        For the top n_holdings positions by market value, download:
            - 1 latest 10-K annual report
            - Last n_quarterly 10-Q quarterly reports

        Each download is validated (size >= 10 KB + contains Table of Contents)
        and logged to the filings_metadata table on success.

        Returns a list of result dicts — one per ticker — with keys:
            ticker, 10-K (list of paths), 10-Q (list of paths)
        """
        from scripts.fetch_filings import fetch_filings_for_ticker, load_environment, get_user_agent

        load_environment()
        user_agent = get_user_agent()
        store = FilingMetadataStore(self.config.db_path)

        result = self.run_pipeline()
        holdings = result["holdings"]
        if holdings.empty:
            return []

        top_tickers: list[str] = (
            holdings.sort_values("market_value", ascending=False)
            .head(n_holdings)
            .index.tolist()
        )

        output = []
        for ticker in top_tickers:
            paths = fetch_filings_for_ticker(
                ticker=ticker,
                user_agent=user_agent,
                store=store,
                n_annual=1,
                n_quarterly=n_quarterly,
            )
            output.append({"ticker": ticker, **paths})

        return output

    def list_downloaded_filings(self, ticker: str | None = None) -> Any:
        """
        Return a DataFrame of all filings logged in the DB.
        Optionally filter by ticker.
        """
        store = FilingMetadataStore(self.config.db_path)
        if ticker:
            rows = store.get_filings_for_ticker(ticker.upper())
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        return store.get_all_filings()

    # ------------------------------------------------------------------ #
    #  Sovereign View                                                    #
    # ------------------------------------------------------------------ #

    def get_full_context(self, ticker: str) -> dict[str, Any]:
        """
        Return a unified 'Sovereign View' for a single ticker by joining
        data from three tables in the database:

            1. transactions    — holdings, cost basis, realised P/L
            2. filings_metadata — downloaded 10-K and 10-Q reports
            3. signals          — latest news headlines and transcript events

        Also fetches the current live price for the ticker.

        Returns a structured dict with the following keys:
            ticker              str
            position            dict  (quantity, avg_cost, capital_invested,
                                        market_value, unrealised_pnl,
                                        unrealised_pnl_pct, current_price)
                                None if ticker is not in current holdings
            filings             list[dict]  — all downloaded filings, newest first
            news                list[dict]  — latest 10 news signals (parsed JSON)
            transcripts         list[dict]  — latest 5 transcript signals (parsed JSON)
            generated_at        str  ISO-8601 timestamp of when this view was built
        """
        ticker = ticker.upper()

        # -- 1. Position data — compute only the requested ticker, not all holdings --
        holdings_df, _, _ = self.recalculate_portfolio()
        position: dict[str, Any] | None = None
        if not holdings_df.empty and ticker in holdings_df.index:
            prices = self.get_live_prices([ticker])
            current_price = prices.get(ticker)
            row = holdings_df.loc[ticker]
            capital = float(row.get("capital_invested", 0) or 0)
            qty = float(row.get("quantity", 0) or 0)
            market_val = qty * current_price if current_price else None
            unrealised = (market_val - capital) if market_val is not None else None
            position = {
                "quantity":           qty,
                "avg_cost":           float(row.get("avg_cost", 0) or 0),
                "capital_invested":   capital,
                "current_price":      current_price,
                "market_value":       market_val,
                "unrealised_pnl":     unrealised,
                "unrealised_pnl_pct": (
                    round(unrealised / capital * 100, 2)
                    if unrealised is not None and capital else None
                ),
            }

        # -- 2. Filings from filings_metadata table --
        filing_store = FilingMetadataStore(self.config.db_path)
        filings = filing_store.get_filings_for_ticker(ticker)

        # -- 3. Signals from signals table --
        signal_store = SignalStore(self.config.db_path)
        raw_news = signal_store.get_signals_for_ticker(ticker, signal_type="news", limit=10)
        raw_transcripts = signal_store.get_signals_for_ticker(
            ticker, signal_type="transcript", limit=5
        )

        def _parse_signals(raw: list[dict]) -> list[dict]:
            parsed = []
            for s in raw:
                try:
                    payload = json.loads(s["content"])
                except (json.JSONDecodeError, KeyError):
                    payload = {"raw": s.get("content", "")}
                parsed.append({
                    "timestamp": s["timestamp"],
                    **payload,
                })
            return parsed

        return {
            "ticker":       ticker,
            "position":     position,
            "filings":      filings,
            "news":         _parse_signals(raw_news),
            "transcripts":  _parse_signals(raw_transcripts),
            "generated_at": datetime.now().isoformat(),
        }

    def run_pipeline(self) -> dict[str, Any]:
        holdings, realised, cash = self.recalculate_portfolio()
        if not holdings.empty:
            holdings["current_price"] = self.get_live_prices(holdings.index.tolist())
            holdings["market_value"] = holdings["quantity"] * holdings["current_price"]
            holdings["unrealised_pnl"] = holdings["market_value"] - holdings["capital_invested"]
            holdings["weight"] = (holdings["market_value"] / holdings["market_value"].sum()) * 100
        
        # Summary metrics
        total_cap = holdings["capital_invested"].sum() if not holdings.empty else 0
        total_mkt = holdings["market_value"].sum() if not holdings.empty else 0
        total_unrealised = holdings["unrealised_pnl"].sum() if not holdings.empty else 0
        total_realised = realised["pnl"].sum() if not realised.empty else 0
        total_divs = realised["divs"].sum() if not realised.empty else 0
        
        summary = pd.DataFrame([{
            "total_capital_invested": total_cap,
            "total_market_value": total_mkt,
            "total_unrealised_pnl": total_unrealised,
            "total_realised_pnl": total_realised,
            "total_dividends": total_divs,
            "total_pnl_all_in": total_unrealised + total_realised + total_divs,
            "unrealised_pnl_pct": (total_unrealised / total_cap * 100) if total_cap else 0
        }])

        return {
            "holdings": holdings,
            "realised": realised,
            "cash": cash,
            "summary": summary
        }
