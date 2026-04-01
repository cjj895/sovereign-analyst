from core.portfolio_engine import PortfolioManager

pm = PortfolioManager("data/transactions_sample.csv")
result = pm.run_pipeline()

print("=" * 60)
print("OPEN POSITIONS")
print("=" * 60)
print(result["display_holdings"].to_string())

print("\n" + "=" * 60)
print("PORTFOLIO SUMMARY")
print("=" * 60)
summary = result["portfolio_summary"]
print(f"  Capital Invested : ${summary.loc[0, 'total_capital_invested']:,.2f}")
print(f"  Market Value     : ${summary.loc[0, 'total_market_value']:,.2f}")
print(f"  Unrealised P/L   : ${summary.loc[0, 'total_unrealised_pnl']:,.2f}  ({summary.loc[0, 'unrealised_pnl_pct']:.2f}%)")
print(f"  Realised P/L     : ${summary.loc[0, 'total_realised_pnl']:,.2f}")
print(f"  Dividends        : ${summary.loc[0, 'total_dividends']:,.2f}")
print(f"  Total P/L All-In : ${summary.loc[0, 'total_pnl_all_in']:,.2f}")

print("\n" + "=" * 60)
print("CASH FLOW SUMMARY (non-investment)")
print("=" * 60)
cash = result["cash_summary"]
print(f"  Total Income   : ${cash['total_income']:,.2f}")
print(f"  Total Expenses : ${cash['total_expenses']:,.2f}")
print(f"  Net Cash Flow  : ${cash['net_cash_flow']:,.2f}")

print("\n" + "=" * 60)
print("REALISED P/L & DIVIDENDS BY TICKER")
print("=" * 60)
print(result["realised"].to_string())

print("\n" + "=" * 60)
print("VALIDATION CHECKS")
print("=" * 60)
for check, passed in result["validations"].items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {check}")
