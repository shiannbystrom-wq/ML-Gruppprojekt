import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "crypto_dataset.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("DATA QUALITY VALIDATION")
print("=" * 40)

# 1. Check date range
print("\nDate range:")
print("Start:", df["Date"].min())
print("End:", df["Date"].max())

# 2. Check missing calendar days
all_dates = pd.date_range(
    start=df["Date"].min(),
    end=df["Date"].max(),
    freq="D"
)

missing_dates = all_dates.difference(df["Date"])

print("\nMissing calendar days:", len(missing_dates))

if len(missing_dates) > 0:
    print("First missing dates:")
    print(missing_dates[:10])

# 3. Check price statistics
print("\nPrice statistics:")
print(df[["ETH_Price", "BTC_Price"]].describe())

# 4. Check unusually large daily price changes
returns = df[["ETH_Price", "BTC_Price"]].pct_change()

print("\nDaily return statistics:")
print(returns.describe())

print("\nLargest absolute daily returns:")
print(returns.abs().max())

# 5. Show latest observations
print("\nLast 10 observations:")
print(df.tail(10))

# 6. Identify largest daily price movements

returns = df[["ETH_Price", "BTC_Price"]].pct_change()

for column in ["ETH_Price", "BTC_Price"]:
    largest_changes = returns[column].abs().nlargest(5)

    print(f"\nLargest daily movements for {column}:")

    for index in largest_changes.index:
        print(
            "Date:", df.loc[index, "Date"],
            "Return:", round(returns.loc[index, column] * 100, 2), "%"
        )

print("\nValidation completed.")