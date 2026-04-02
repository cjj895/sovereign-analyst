from core.portfolio_engine import PortfolioManager
import sqlite3
from pathlib import Path

# 1. Initialize DB
DB_PATH = Path("data/sovereign.db")
if DB_PATH.exists(): DB_PATH.unlink() # Start fresh for this demo

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    type        TEXT    NOT NULL,
    asset       TEXT,
    ticker      TEXT,
    price       REAL,
    quantity    REAL,
    amount      REAL,
    description TEXT,
    ratio       REAL,
    created_at  TEXT    DEFAULT (datetime('now'))
);
"""
conn = sqlite3.connect(DB_PATH)
conn.executescript(SCHEMA)
conn.close()

pm = PortfolioManager()

# 2. Seed with some data including a SPLIT
print("--- Seeding Database ---")
pm.add_transaction("2024-01-01", "buy", "NVIDIA Corp", "NVDA", price=400.0, quantity=10, amount=-4000.0)
pm.add_transaction("2024-06-10", "split", "NVIDIA Corp", "NVDA", ratio=10.0, description="10-for-1 forward split")
pm.add_transaction("2024-07-01", "sell", "NVIDIA Corp", "NVDA", price=120.0, quantity=20, amount=2400.0)
pm.add_transaction("2024-08-01", "dividend", "NVIDIA Corp", "NVDA", amount=15.0)

# 3. Run Pipeline
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
