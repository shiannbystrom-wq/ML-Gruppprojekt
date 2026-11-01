import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "crypto_dataset.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "crypto_targets.csv"

df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Ensure one observation per date
assert df["Date"].is_unique

# Create a lookup table for future ETH prices
price_lookup = df.set_index("Date")["ETH_Price"]

# Define target horizons
horizons = {
    "6M": pd.DateOffset(months=6),
    "12M": pd.DateOffset(months=12)
}

for label, offset in horizons.items():

    target_dates = df["Date"] + offset

    # Find the first available observation on or after target date
    future_prices = price_lookup.reindex(
        pd.DatetimeIndex(target_dates),
        method="bfill",
        tolerance=pd.Timedelta(days=3)
    ).to_numpy()

    df[f"ETH_Future_Price_{label}"] = future_prices

    df[f"ETH_Return_{label}"] = (
        future_prices / df["ETH_Price"] - 1
    )

# Summary
print("\nFUTURE RETURNS SUMMARY")
print("=" * 45)

for label in horizons:

    column = f"ETH_Return_{label}"

    print(f"\n{label} Future Returns")
    print("Valid observations:", df[column].notna().sum())
    print("Missing observations:", df[column].isna().sum())
    print("Mean return:", round(df[column].mean() * 100, 2), "%")
    print("Median return:", round(df[column].median() * 100, 2), "%")

# Save
df.to_csv(OUTPUT_PATH, index=False)

print("\nFirst five rows:")
print(df.head())

print("\nLast ten rows:")
print(df.tail(10))

print("\nSaved to:", OUTPUT_PATH)

# Return distribution checks

for column in ["ETH_Return_6M", "ETH_Return_12M"]:

    print(f"\n{column} distribution:")

    print(
        df[column].describe(
            percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
        )
    )

    print(
        "Positive returns:",
        (df[column] > 0).mean() * 100,
        "%"
    )

print("\nTarget creation completed!")