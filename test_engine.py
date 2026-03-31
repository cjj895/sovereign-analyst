from core.portfolio_engine import PortfolioManager

# Assume transactions_sample.csv is in the ./data directory relative to this script
csv_path = "data/transactions_sample.csv"

# Initialize PortfolioManager with the CSV file
pm = PortfolioManager(csv_path)

# Get the portfolio summary
summary = pm.get_summary()

# Print the summary to the console
print(summary)
