from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


@dataclass
class PortfolioConfig:
    """Configuration for portfolio file locations."""

    transactions_file: Path = Path("data/transactions_sample.csv")
    history_file: Path = Path("data/portfolio_history.csv")
    cache_file: Path = Path("data/.price_cache.json")


class PortfolioManager:
    """
    Manage portfolio processing, snapshots, validations, and charts.

    Supported transaction types in the CSV:
        buy      – cash decreases, quantity increases; grows cost basis.
        sell     – cash increases, quantity decreases; realised P/L recorded.
        dividend – cash increases, quantity unchanged; tracked separately.
        income   – non-investment cash inflow (salary, allowance, etc.).
        expense  – non-investment cash outflow (groceries, fees, etc.).

    Cost basis method: Average Cost (ACB).
        On each buy  → new avg_cost = (old_cost_total + qty * price) / new_total_qty
        On each sell → realised_pnl += (sell_price − avg_cost) × qty_sold
                       cost_total    -= avg_cost × qty_sold
    """

    INVESTMENT_TYPES: frozenset[str] = frozenset({"buy", "sell", "dividend"})
    CACHE_TTL_SECONDS: int = 900  # 15 minutes

    def __init__(
        self,
        transactions_file: str | Path = "data/transactions_sample.csv",
        history_file: str | Path = "data/portfolio_history.csv",
    ) -> None:
        self.config = PortfolioConfig(
            transactions_file=Path(transactions_file),
            history_file=Path(history_file),
        )
        self.snapshot_header: list[str] = [
            "month",
            "total_capital_invested",
            "total_market_value",
            "total_unrealised_pnl",
            "total_realised_pnl",
            "total_dividends",
            "unrealised_pnl_pct",
        ]

    # ------------------------------------------------------------------ #
    #  Data Loading                                                        #
    # ------------------------------------------------------------------ #

    def load_transactions(self) -> pd.DataFrame:
        """Load and coerce schema types from the transactions CSV."""
        df = pd.read_csv(self.config.transactions_file)
        df["date"] = pd.to_datetime(df["date"])
        df["type"] = df["type"].str.strip().str.lower()
        numeric_cols: list[str] = ["price", "quantity", "amount"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        return df.sort_values("date").reset_index(drop=True)

    def get_investment_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return rows that are buy, sell, or dividend activity."""
        return df[df["type"].isin(self.INVESTMENT_TYPES)].copy()

    def get_expense_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["type"] == "expense"].copy()

    def get_income_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["type"] == "income"].copy()

    def filter_by_date(
        self,
        df: pd.DataFrame,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()

    # ------------------------------------------------------------------ #
    #  Holdings & Cost Basis (Average Cost Method)                        #
    # ------------------------------------------------------------------ #

    def calculate_holdings(
        self,
        transactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Walk through every buy / sell / dividend row in date order and
        compute open positions + realised P/L using the Average Cost Basis.

        Returns
        -------
        holdings_df : pd.DataFrame  (indexed by ticker)
            Open positions with columns:
            asset, quantity, avg_cost, capital_invested, realised_pnl, dividends

        realised_df : pd.DataFrame  (indexed by ticker)
            Tickers that have at least one sell or dividend recorded:
            realised_pnl, dividends
        """
        inv_df = (
            transactions[transactions["type"].isin(self.INVESTMENT_TYPES)]
            .copy()
            .sort_values("date")
        )

        # Per-ticker running state
        lots: dict[str, dict] = {}      # open-position book
        realised: dict[str, dict] = {}  # closed / income book

        for _, row in inv_df.iterrows():
            ticker = str(row.get("ticker", "")).strip()
            if not ticker or ticker.lower() == "nan":
                continue

            if ticker not in lots:
                lots[ticker] = {
                    "asset": str(row.get("asset", "")).strip(),
                    "quantity": 0.0,
                    "total_cost": 0.0,
                }
            if ticker not in realised:
                realised[ticker] = {"realised_pnl": 0.0, "dividends": 0.0}

            tx_type = row["type"]
            qty = abs(float(row["quantity"])) if pd.notna(row["quantity"]) else 0.0
            price = float(row["price"]) if pd.notna(row["price"]) else 0.0

            if tx_type == "buy":
                lots[ticker]["quantity"] += qty
                lots[ticker]["total_cost"] += qty * price

            elif tx_type == "sell":
                current_qty = lots[ticker]["quantity"]
                if current_qty > 0 and qty > 0:
                    avg_cost = lots[ticker]["total_cost"] / current_qty
                    qty_sold = min(qty, current_qty)  # cap at what we own
                    realised[ticker]["realised_pnl"] += (price - avg_cost) * qty_sold
                    lots[ticker]["total_cost"] = max(
                        0.0, lots[ticker]["total_cost"] - avg_cost * qty_sold
                    )
                    lots[ticker]["quantity"] = max(0.0, current_qty - qty_sold)

            elif tx_type == "dividend":
                amount = float(row["amount"]) if pd.notna(row["amount"]) else 0.0
                realised[ticker]["dividends"] += abs(amount)

        # ---- Build open-positions DataFrame ----
        holdings_records = []
        for ticker, lot in lots.items():
            if lot["quantity"] > 1e-9:
                avg_cost = lot["total_cost"] / lot["quantity"]
                holdings_records.append(
                    {
                        "ticker": ticker,
                        "asset": lot["asset"],
                        "quantity": round(lot["quantity"], 6),
                        "avg_cost": round(avg_cost, 4),
                        "capital_invested": round(lot["total_cost"], 2),
                        "realised_pnl": round(
                            realised.get(ticker, {}).get("realised_pnl", 0.0), 2
                        ),
                        "dividends": round(
                            realised.get(ticker, {}).get("dividends", 0.0), 2
                        ),
                    }
                )

        holdings_df = (
            pd.DataFrame(holdings_records).set_index("ticker")
            if holdings_records
            else pd.DataFrame(
                columns=[
                    "asset", "quantity", "avg_cost",
                    "capital_invested", "realised_pnl", "dividends",
                ]
            )
        )

        # ---- Build realised-only DataFrame (all tickers with any activity) ----
        realised_records = [
            {"ticker": t, **r}
            for t, r in realised.items()
            if r["realised_pnl"] != 0.0 or r["dividends"] != 0.0
        ]
        realised_df = (
            pd.DataFrame(realised_records).set_index("ticker")
            if realised_records
            else pd.DataFrame(columns=["realised_pnl", "dividends"])
        )

        return holdings_df, realised_df

    # ------------------------------------------------------------------ #
    #  Price Cache                                                        #
    # ------------------------------------------------------------------ #

    def _load_price_cache(self) -> dict:
        """
        Read the on-disk JSON price cache.
        Returns an empty dict on any read or parse failure.
        """
        try:
            return json.loads(self.config.cache_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save_price_cache(self, cache: dict) -> None:
        """Write the price cache back to disk as formatted JSON."""
        self.config.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.cache_file.write_text(
            json.dumps(cache, indent=2), encoding="utf-8"
        )

    def _is_cache_fresh(self, entry: dict) -> bool:
        """
        Return True if a cache entry was fetched within the last
        CACHE_TTL_SECONDS seconds (default 15 minutes).
        """
        try:
            fetched_at = datetime.fromisoformat(entry["fetched_at"])
            age_seconds = (datetime.now() - fetched_at).total_seconds()
            return age_seconds < self.CACHE_TTL_SECONDS
        except (KeyError, ValueError):
            return False

    # ------------------------------------------------------------------ #
    #  Live Prices & Position Metrics                                     #
    # ------------------------------------------------------------------ #

    def get_live_prices(self, tickers: list[str]) -> list[float | None]:
        """
        Return the last traded price for each ticker.

        Cache behaviour
        ---------------
        - Prices are stored in data/.price_cache.json keyed by ticker.
        - An entry is reused as-is if it was fetched within CACHE_TTL_SECONDS.
        - If a network call fails but a stale entry exists, the stale price is
          returned as a fallback (preferable to returning None and crashing
          downstream calculations).
        - The cache file is only written when at least one entry was refreshed,
          avoiding unnecessary disk I/O on fully-fresh runs.
        """
        cache = self._load_price_cache()
        prices: list[float | None] = []
        cache_updated = False

        for ticker in tickers:
            entry = cache.get(ticker, {})

            if self._is_cache_fresh(entry):
                prices.append(entry["price"])
                continue

            # Entry is stale or missing — fetch a fresh price.
            fresh_price: float | None = None
            try:
                fresh_price = round(float(yf.Ticker(ticker).fast_info.last_price), 2)
            except Exception:
                # Network error or bad ticker: fall back to stale data if available.
                fresh_price = entry.get("price")

            cache[ticker] = {
                "price": fresh_price,
                "fetched_at": datetime.now().isoformat(),
            }
            cache_updated = True
            prices.append(fresh_price)

        if cache_updated:
            self._save_price_cache(cache)

        return prices

    def calculate_position_metrics(self, holdings: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich holdings with live market data:
        current_price, market_value, unrealised_pnl, unrealised_pnl_pct, weight.
        """
        enriched = holdings.copy()
        enriched["current_price"] = self.get_live_prices(enriched.index.tolist())
        enriched["market_value"] = enriched["quantity"] * enriched["current_price"]
        enriched["unrealised_pnl"] = (
            (enriched["current_price"] - enriched["avg_cost"]) * enriched["quantity"]
        )
        enriched["unrealised_pnl_pct"] = round(
            enriched["unrealised_pnl"] / enriched["capital_invested"] * 100,
            2,
        )
        total_market = enriched["market_value"].sum()
        enriched["weight"] = round(
            enriched["market_value"] / total_market * 100, 2
        )
        return enriched

    # ------------------------------------------------------------------ #
    #  Portfolio-Level Aggregation                                        #
    # ------------------------------------------------------------------ #

    def compute_portfolio_metrics(self, holdings: pd.DataFrame) -> pd.DataFrame:
        """
        Roll up per-position metrics into a single-row portfolio summary.

        Columns returned
        ----------------
        total_capital_invested  – sum of current cost bases (open positions only)
        total_market_value      – sum of market values
        total_unrealised_pnl    – mark-to-market gain/loss on open positions
        total_realised_pnl      – locked-in gains/losses from completed sells
        total_dividends         – cumulative dividend income received
        total_pnl_all_in        – unrealised + realised + dividends
        unrealised_pnl_pct      – unrealised P/L as % of capital invested
        """
        total_capital = holdings["capital_invested"].sum()
        total_market = holdings["market_value"].sum()
        total_unrealised = round(holdings["unrealised_pnl"].sum(), 2)
        total_realised = round(holdings["realised_pnl"].sum(), 2)
        total_dividends = round(holdings["dividends"].sum(), 2)
        total_pnl_all_in = round(total_unrealised + total_realised + total_dividends, 2)
        unrealised_pnl_pct = (
            round((total_unrealised / total_capital) * 100, 2) if total_capital else 0.0
        )

        return pd.DataFrame(
            {
                "total_capital_invested": [total_capital],
                "total_market_value": [total_market],
                "total_unrealised_pnl": [total_unrealised],
                "total_realised_pnl": [total_realised],
                "total_dividends": [total_dividends],
                "total_pnl_all_in": [total_pnl_all_in],
                "unrealised_pnl_pct": [unrealised_pnl_pct],
            }
        )

    def get_cash_summary(self, df: pd.DataFrame) -> dict[str, float]:
        """Summarise non-investment cash flows (income and expenses)."""
        income = df[df["type"] == "income"]["amount"].sum()
        expenses = abs(df[df["type"] == "expense"]["amount"].sum())
        return {
            "total_income": round(float(income), 2),
            "total_expenses": round(float(expenses), 2),
            "net_cash_flow": round(float(income) - float(expenses), 2),
        }

    # ------------------------------------------------------------------ #
    #  Snapshot History                                                   #
    # ------------------------------------------------------------------ #

    def ensure_history_file(self) -> None:
        """Create the history CSV with headers if it does not exist."""
        if self.config.history_file.exists():
            return
        self.config.history_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=self.snapshot_header).to_csv(
            self.config.history_file, index=False
        )

    def add_snapshot(
        self, snapshot_date: str, portfolio_summary: pd.DataFrame
    ) -> bool:
        """
        Append a monthly snapshot if one doesn't already exist for that date.
        Returns True when a new row is written, else False.
        """
        self.ensure_history_file()
        history_df = pd.read_csv(self.config.history_file)
        if "month" in history_df.columns:
            existing = set(history_df["month"].astype(str).str.strip())
            if snapshot_date.strip() in existing:
                return False

        row = [snapshot_date] + [
            portfolio_summary.loc[0, col]
            for col in self.snapshot_header[1:]
            if col in portfolio_summary.columns
        ]
        with self.config.history_file.open("a", encoding="utf-8", newline="") as f:
            f.write(",".join(map(str, row)) + "\n")
        return True

    def load_snapshots(self) -> pd.DataFrame:
        self.ensure_history_file()
        snapshots_df = pd.read_csv(self.config.history_file)
        if snapshots_df.empty:
            return snapshots_df
        snapshots_df["date"] = pd.to_datetime(snapshots_df["month"])
        return snapshots_df.sort_values("date")

    # ------------------------------------------------------------------ #
    #  Validations                                                        #
    # ------------------------------------------------------------------ #

    def run_validations(self, holdings: pd.DataFrame) -> dict[str, bool]:
        """Sanity-check the enriched holdings table."""
        checks: dict[str, bool] = {}
        checks["no_negative_avg_cost"] = bool((holdings["avg_cost"] > 0).all())
        checks["no_negative_holdings"] = bool((holdings["quantity"] >= 0).all())
        checks["market_value_correct"] = bool(
            (
                (
                    holdings["market_value"]
                    - holdings["quantity"] * holdings["current_price"]
                ).abs()
                < 0.01
            ).all()
        )
        checks["weights_sum_to_100"] = bool(
            abs(holdings["weight"].sum() - 100) < 0.01
        )
        return checks

    # ------------------------------------------------------------------ #
    #  Display Formatting                                                 #
    # ------------------------------------------------------------------ #

    def format_holdings_display(self, holdings: pd.DataFrame) -> pd.DataFrame:
        display_df = holdings.copy()
        dollar_cols = [
            "avg_cost", "current_price", "market_value",
            "capital_invested", "unrealised_pnl", "realised_pnl", "dividends",
        ]
        for col in dollar_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].map("${:,.2f}".format)
        if "unrealised_pnl_pct" in display_df.columns:
            display_df["unrealised_pnl_pct"] = display_df["unrealised_pnl_pct"].map(
                "{:.2f}%".format
            )
        if "weight" in display_df.columns:
            display_df["weight"] = display_df["weight"].map("{:.2f}%".format)
        return display_df

    # ------------------------------------------------------------------ #
    #  Visualisation                                                      #
    # ------------------------------------------------------------------ #

    def plot_allocation_pie(self, holdings: pd.DataFrame) -> None:
        plt.figure(figsize=(6, 6))
        plt.pie(
            holdings["market_value"],
            labels=holdings.index,
            autopct="%.1f%%",
            startangle=90,
            colors=plt.cm.tab20.colors,
        )
        plt.title("Portfolio Allocation by Asset")
        plt.show()

    def plot_unrealised_pnl_bar(self, holdings: pd.DataFrame) -> None:
        plt.figure(figsize=(8, 4))
        plt.bar(
            holdings.index,
            holdings["unrealised_pnl"],
            color="skyblue",
            edgecolor="black",
        )
        plt.xlabel("Asset")
        plt.ylabel("Unrealised P/L ($)")
        plt.title("Unrealised P/L by Asset")
        plt.grid(axis="y", linestyle="--")
        plt.show()

    def plot_realised_vs_unrealised(self, holdings: pd.DataFrame) -> None:
        """Grouped bar: realised vs unrealised P/L side-by-side per ticker."""
        tickers = holdings.index.tolist()
        x_pos = list(range(len(tickers)))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            [p - width / 2 for p in x_pos],
            holdings["unrealised_pnl"],
            width,
            label="Unrealised P/L",
            color="skyblue",
            edgecolor="black",
        )
        ax.bar(
            [p + width / 2 for p in x_pos],
            holdings["realised_pnl"],
            width,
            label="Realised P/L",
            color="salmon",
            edgecolor="black",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tickers)
        ax.set_xlabel("Ticker")
        ax.set_ylabel("P/L ($)")
        ax.set_title("Realised vs Unrealised P/L by Ticker")
        ax.legend()
        ax.grid(axis="y", linestyle="--")
        plt.tight_layout()
        plt.show()

    def plot_market_value_over_time(self, snapshots_df: pd.DataFrame) -> None:
        plt.figure(figsize=(8, 4))
        plt.plot(snapshots_df["month"], snapshots_df["total_market_value"], marker="o")
        plt.title("Portfolio Market Value Over Time")
        plt.xlabel("Month")
        plt.ylabel("Total Market Value")
        plt.grid(True, linestyle="--")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_capital_vs_market_value(self, snapshots_df: pd.DataFrame) -> None:
        plt.figure(figsize=(8, 4))
        plt.plot(
            snapshots_df["month"],
            snapshots_df["total_market_value"],
            marker="o",
            linestyle="-",
            color="green",
            label="Market Value",
        )
        plt.plot(
            snapshots_df["month"],
            snapshots_df["total_capital_invested"],
            marker="s",
            linestyle="--",
            color="blue",
            label="Capital Invested",
        )
        plt.title("Capital vs Market Value")
        plt.xlabel("Month")
        plt.ylabel("Value ($)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  Filings                                                           #
    # ------------------------------------------------------------------ #

    # Minimum acceptable byte size for a real SEC 10-K filing.
    # A genuine 10-K is typically several MB; anything smaller is likely
    # an error page, a redirect, or a truncated download.
    MIN_FILING_SIZE_BYTES: int = 10_240  # 10 KB

    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Convert a raw byte count to a human-readable string."""
        if size_bytes >= 1_048_576:
            return f"{size_bytes / 1_048_576:.2f} MB"
        if size_bytes >= 1_024:
            return f"{size_bytes / 1_024:.1f} KB"
        return f"{size_bytes} B"

    def validate_filing(self, path: Path) -> dict[str, Any]:
        """
        Validate that a filing file was actually downloaded and contains
        real content.

        Checks performed
        ----------------
        exists      – the file was created at the expected path.
        size_bytes  – raw byte count from the filesystem.
        size_readable – human-readable size (B / KB / MB).
        is_valid    – True only when the file exists AND its size exceeds
                      MIN_FILING_SIZE_BYTES (10 KB).  A real 10-K runs into
                      the megabytes; anything smaller is almost certainly an
                      HTML error page or an empty response.
        status      – plain-English description of the result.

        Returns a dict so the caller can log, print, or assert against it.
        """
        if not path.exists():
            return {
                "path": str(path),
                "exists": False,
                "size_bytes": 0,
                "size_readable": "0 B",
                "is_valid": False,
                "status": "File not found — download did not complete.",
            }

        size_bytes = path.stat().st_size
        size_readable = self._format_file_size(size_bytes)
        is_valid = size_bytes >= self.MIN_FILING_SIZE_BYTES

        if size_bytes == 0:
            status = "Empty file — download likely failed or returned no data."
        elif not is_valid:
            status = (
                f"Suspiciously small ({size_readable}) — "
                "may be an error response rather than a real filing."
            )
        else:
            status = f"OK — {size_readable}"

        return {
            "path": str(path),
            "exists": True,
            "size_bytes": size_bytes,
            "size_readable": size_readable,
            "is_valid": is_valid,
            "status": status,
        }

    def fetch_filings_for_top_holdings(self, n: int = 3) -> list[dict[str, Any]]:
        """
        Fetch the latest SEC 10-K filing for the top N holdings by market value,
        then validate each downloaded file.

        Steps
        -----
        1. Run the full pipeline to get enriched holdings with live prices.
        2. Sort by market_value descending and take the top N tickers.
        3. For each ticker call fetch_latest_10k() from scripts/fetch_filings.py.
        4. Validate the resulting file with validate_filing().
        5. Return a list of result dicts — one per ticker — containing the
           ticker name, success flag, and the full validation report.

        Requires a .env file at the project root with:
            USER_AGENT=Your Name (your@email.com)
        """
        from scripts.fetch_filings import fetch_latest_10k  # lazy import

        holdings = self.run_pipeline()["holdings"]
        top_tickers: list[str] = (
            holdings.sort_values("market_value", ascending=False)
            .head(n)
            .index.tolist()
        )

        results: list[dict[str, Any]] = []
        for ticker in top_tickers:
            record: dict[str, Any] = {"ticker": ticker}
            try:
                path = fetch_latest_10k(ticker)
                validation = self.validate_filing(path)
                record.update({"fetched": True, **validation})
            except Exception as exc:
                record.update({
                    "fetched": False,
                    "path": None,
                    "exists": False,
                    "size_bytes": 0,
                    "size_readable": "0 B",
                    "is_valid": False,
                    "status": f"Error: {exc}",
                })
            results.append(record)

        return results

    # ------------------------------------------------------------------ #
    #  Pipeline                                                           #
    # ------------------------------------------------------------------ #

    def run_pipeline(
        self,
        snapshot_date: str | None = None,
    ) -> dict[str, Any]:
        """
        End-to-end processing pipeline.

        Returns
        -------
        transactions        – full raw DataFrame
        holdings            – enriched open-position DataFrame
        realised            – realised P/L + dividends per ticker
        display_holdings    – human-readable formatted holdings
        portfolio_summary   – single-row portfolio metrics DataFrame
        cash_summary        – dict of income / expense totals
        snapshots           – historical snapshot DataFrame
        validations         – dict of bool sanity checks
        snapshot_written    – True if a new snapshot row was appended
        """
        df = self.load_transactions()
        holdings, realised_df = self.calculate_holdings(df)
        holdings = self.calculate_position_metrics(holdings)
        portfolio_summary = self.compute_portfolio_metrics(holdings)
        cash_summary = self.get_cash_summary(df)

        snapshot_written = False
        if snapshot_date:
            snapshot_written = self.add_snapshot(snapshot_date, portfolio_summary)

        snapshots_df = self.load_snapshots()
        checks = self.run_validations(holdings)
        display_df = self.format_holdings_display(holdings)

        return {
            "transactions": df,
            "holdings": holdings,
            "realised": realised_df,
            "display_holdings": display_df,
            "portfolio_summary": portfolio_summary,
            "cash_summary": cash_summary,
            "snapshots": snapshots_df,
            "validations": checks,
            "snapshot_written": snapshot_written,
        }

    def get_summary(self) -> pd.DataFrame:
        """Convenience method: return the current portfolio summary metrics."""
        return self.run_pipeline()["portfolio_summary"]
