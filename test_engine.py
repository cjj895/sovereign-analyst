"""
Quick integration test for PortfolioManager.

Wipes and recreates the database, seeds sample transactions including a
SPLIT, runs the pipeline, and prints holdings + summary.

Run:  .venv/bin/python test_engine.py
"""
from pathlib import Path

from core.database import TransactionStore
from core.portfolio_engine import PortfolioManager

DB_PATH = Path("data/sovereign.db")

# Start fresh
if DB_PATH.exists():
    DB_PATH.unlink()

# Initialise schema via TransactionStore (single source of truth)
TransactionStore(DB_PATH)

pm = PortfolioManager(db_path=DB_PATH)

# Seed with test data including a SPLIT
print("--- Seeding Database ---")
pm.add_transaction("2024-01-01", "buy", "NVIDIA Corp", "NVDA", price=400.0, quantity=10, amount=-4000.0)
pm.add_transaction("2024-06-10", "split", "NVIDIA Corp", "NVDA", ratio=10.0, description="10-for-1 forward split")
pm.add_transaction("2024-07-01", "sell", "NVIDIA Corp", "NVDA", price=120.0, quantity=20, amount=2400.0)
pm.add_transaction("2024-08-01", "dividend", "NVIDIA Corp", "NVDA", amount=15.0)

# Run Pipeline
print("\n--- Running Portfolio Pipeline ---")
res = pm.run_pipeline()

print("\n" + "=" * 60)
print("HOLDINGS (Split-Adjusted)")
print("=" * 60)
h = res["holdings"]
if not h.empty:
    for t, r in h.iterrows():
        print(f"{t:<6} | Qty: {r['quantity']:>4.1f} | Avg: ${r['avg_cost']:>6.2f} | Mkt: ${r['market_value']:>7.2f} | Unrealised: ${r['unrealised_pnl']:>7.2f}")
else:
    print("No open positions.")

print("\n" + "=" * 60)
print("REALISED P/L & DIVIDENDS")
print("=" * 60)
r = res["realised"]
if not r.empty:
    for t, row in r.iterrows():
        print(f"{t:<6} | Realised: ${row['pnl']:>7.2f} | Dividends: ${row['divs']:>7.2f}")

print("\n" + "=" * 60)
print("PORTFOLIO SUMMARY")
print("=" * 60)
s = res["summary"].iloc[0]
print(f"  Total Capital Invested : ${s['total_capital_invested']:,.2f}")
print(f"  Total Market Value     : ${s['total_market_value']:,.2f}")
print(f"  Total P/L All-In       : ${s['total_pnl_all_in']:,.2f} ({s['unrealised_pnl_pct']:.2f}%)")
