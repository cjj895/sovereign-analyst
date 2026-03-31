from __future__ import annotations

from dataclasses import dataclass
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


class PortfolioManager:
    """Manage portfolio processing, snapshots, validations, and charts."""

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
            "portfolio_pnl_pct",
        ]

    def load_transactions(self) -> pd.DataFrame:
        """Load transactions and coerce schema types."""
        df = pd.read_csv(self.config.transactions_file)
        df["date"] = pd.to_datetime(df["date"])
        numeric_cols: list[str] = ["price", "quantity", "amount"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        return df

    def get_investment_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["type"] == "investment"].copy()

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

    def calculate_holdings(self, investment_df: pd.DataFrame) -> pd.DataFrame:
        """Build holdings table from investment transactions."""
        grouped = investment_df.groupby("ticker").agg(
            {"asset": "first", "quantity": "sum", "amount": "sum"}
        )
        holdings = grouped[grouped["quantity"] > 0].copy()
        holdings["avg_price"] = abs(holdings["amount"] / holdings["quantity"])

        capital_invested = (
            investment_df[investment_df["amount"] < 0].groupby("ticker")["amount"].sum()
        )
        holdings["capital_invested"] = abs(capital_invested.reindex(holdings.index))
        holdings = holdings.drop(columns=["amount"])
        return holdings

    def get_live_prices(self, tickers: list[str]) -> list[float | None]:
        """Fetch last traded prices for tickers."""
        prices: list[float | None] = []
        for ticker in tickers:
            try:
                obj = yf.Ticker(ticker)
                last_price = obj.fast_info["last_price"]
                prices.append(round(float(last_price), 2))
            except Exception:
                prices.append(None)
        return prices

    def calculate_position_metrics(self, holdings: pd.DataFrame) -> pd.DataFrame:
        """Add per-position valuation and P/L metrics."""
        enriched = holdings.copy()
        enriched["current_price"] = self.get_live_prices(enriched.index.tolist())
        enriched["market_value"] = enriched["quantity"] * enriched["current_price"]
        enriched["unrealised_pnl"] = (
            (enriched["current_price"] - enriched["avg_price"]) * enriched["quantity"]
        )
        enriched["unrealised_pnl_pct"] = round(
            enriched["unrealised_pnl"] / enriched["capital_invested"] * 100,
            2,
        )
        enriched["weight"] = round(
            enriched["market_value"] / enriched["market_value"].sum() * 100,
            2,
        )
        return enriched

    def compute_portfolio_metrics(self, holdings: pd.DataFrame) -> pd.DataFrame:
        total_capital = holdings["capital_invested"].sum()
        total_market = holdings["market_value"].sum()
        total_pnl = round(holdings["unrealised_pnl"].sum(), 2)
        pnl_pct = round((total_pnl / total_capital) * 100, 2) if total_capital else 0.0

        return pd.DataFrame(
            {
                "total_capital_invested": [total_capital],
                "total_market_value": [total_market],
                "total_unrealised_pnl": [total_pnl],
                "portfolio_pnl_pct": [pnl_pct],
            }
        )

    def ensure_history_file(self) -> None:
        """Create history CSV if it does not exist."""
        if self.config.history_file.exists():
            return

        self.config.history_file.parent.mkdir(parents=True, exist_ok=True)
        header_df = pd.DataFrame(columns=self.snapshot_header)
        header_df.to_csv(self.config.history_file, index=False)

    def add_snapshot(self, snapshot_date: str, portfolio_summary: pd.DataFrame) -> bool:
        """
        Add monthly snapshot if it doesn't already exist.

        Returns True when a new row is written, else False.
        """
        self.ensure_history_file()

        snapshot = [
            snapshot_date,
            portfolio_summary.loc[0, "total_capital_invested"],
            portfolio_summary.loc[0, "total_market_value"],
            portfolio_summary.loc[0, "total_unrealised_pnl"],
            portfolio_summary.loc[0, "portfolio_pnl_pct"],
        ]

        history_df = pd.read_csv(self.config.history_file)
        if "month" in history_df.columns:
            existing = set(history_df["month"].astype(str).str.strip().tolist())
            if snapshot_date.strip() in existing:
                return False

        with self.config.history_file.open("a", encoding="utf-8", newline="") as file:
            file.write(",".join(map(str, snapshot)) + "\n")
        return True

    def load_snapshots(self) -> pd.DataFrame:
        self.ensure_history_file()
        snapshots_df = pd.read_csv(self.config.history_file)
        if snapshots_df.empty:
            return snapshots_df
        snapshots_df["date"] = pd.to_datetime(snapshots_df["month"])
        snapshots_df = snapshots_df.sort_values("date")
        return snapshots_df

    def run_validations(
        self,
        investment_df: pd.DataFrame,
        holdings: pd.DataFrame,
    ) -> dict[str, bool]:
        checks: dict[str, bool] = {}
        checks["no_negative_prices"] = bool((investment_df["price"] > 0).all())
        checks["market_value_correct"] = bool(
            (
                holdings["market_value"]
                == holdings["quantity"] * holdings["current_price"]
            ).all()
        )
        checks["no_negative_holdings"] = bool((holdings["quantity"] >= 0).all())
        checks["weights_sum_to_100"] = bool(abs(holdings["weight"].sum() - 100) < 0.01)
        return checks

    def format_holdings_display(self, holdings: pd.DataFrame) -> pd.DataFrame:
        display_df = holdings.copy()
        display_df["avg_price"] = round(display_df["avg_price"], 2)
        display_df["unrealised_pnl_pct"] = display_df["unrealised_pnl_pct"].map(
            "{:.2f}%".format
        )
        display_df["weight"] = display_df["weight"].map("{:.2f}%".format)
        return display_df

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

    def plot_market_value_over_time(self, snapshots_df: pd.DataFrame) -> None:
        plt.figure(figsize=(8, 4))
        plt.plot(
            snapshots_df["month"],
            snapshots_df["total_market_value"],
            marker="o",
        )
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

    def run_pipeline(
        self,
        snapshot_date: str | None = None,
    ) -> dict[str, Any]:
        """
        End-to-end processing pipeline.

        Returns holdings, display table, summary, snapshots, and validation checks.
        """
        df = self.load_transactions()
        investment_df = self.get_investment_transactions(df)[
            ["asset", "ticker", "price", "quantity", "amount"]
        ]
        holdings = self.calculate_holdings(investment_df)
        holdings = self.calculate_position_metrics(holdings)
        portfolio_summary = self.compute_portfolio_metrics(holdings)

        snapshot_written = False
        if snapshot_date:
            snapshot_written = self.add_snapshot(snapshot_date, portfolio_summary)

        snapshots_df = self.load_snapshots()
        checks = self.run_validations(investment_df, holdings)
        display_df = self.format_holdings_display(holdings)

        return {
            "transactions": df,
            "investment_transactions": investment_df,
            "holdings": holdings,
            "display_holdings": display_df,
            "portfolio_summary": portfolio_summary,
            "snapshots": snapshots_df,
            "validations": checks,
            "snapshot_written": snapshot_written,
        }

    def get_summary(self) -> pd.DataFrame:
        """Convenience method to return current portfolio summary metrics."""
        result = self.run_pipeline()
        return result["portfolio_summary"]
