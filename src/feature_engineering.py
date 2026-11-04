import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "crypto_dataset.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "crypto_features.csv"

# Load cleaned dataset
df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# 1. Daily returns
df["ETH_Return"] = df["ETH_Price"].pct_change()
df["BTC_Return"] = df["BTC_Price"].pct_change()

# 2. Moving averages (daily)
df["ETH_MA30"] = df["ETH_Price"].rolling(window=30).mean()
df["ETH_MA200"] = df["ETH_Price"].rolling(window=200).mean()

# 3. ETH rolling volatility (30 days)
df["ETH_Volatility30"] = (
    df["ETH_Return"].rolling(window=30).std()
)

# 4. Rolling correlation (30 days)
df["BTC_ETH_Correlation30"] = (
    df["BTC_Return"]
    .rolling(window=30)
    .corr(df["ETH_Return"])
)

# 5. ETH 20-week Simple Moving Average (SMA)
# Daily data: 20 weeks = 140 observations
df["ETH_SMA20W"] = (
    df["ETH_Price"]
    .rolling(window=140, min_periods=140)
    .mean()
)

# 6. ETH 21-week Exponential Moving Average (EMA)
# Daily data: 21 weeks = 147 observations
df["ETH_EMA21W"] = (
    df["ETH_Price"]
    .ewm(span=147, adjust=False, min_periods=147)
    .mean()
)

# Save dataset
df.to_csv(OUTPUT_PATH, index=False)

# Summary
print("FEATURE ENGINEERING COMPLETED")
print("=" * 40)

print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nLast five rows:")
print(df.tail())

print("\nNew features:")
print(df[["Date", "ETH_Price", "ETH_SMA20W", "ETH_EMA21W"]].tail())

print("\nSaved to:")
print(OUTPUT_PATH)