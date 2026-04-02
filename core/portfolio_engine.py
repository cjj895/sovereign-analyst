from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


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
