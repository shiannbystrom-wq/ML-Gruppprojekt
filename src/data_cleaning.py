import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Load API datasets
eth = pd.read_csv(RAW_DIR / "eth_api.csv")
btc = pd.read_csv(RAW_DIR / "btc_api.csv")

print("API datasets loaded successfully.")


# Select required columns
eth = eth[["time", "PriceUSD"]].copy()
btc = btc[["time", "PriceUSD"]].copy()


# Rename columns
eth = eth.rename(
    columns={"time": "Date", "PriceUSD": "ETH_Price"}
)

btc = btc.rename(
    columns={"time": "Date", "PriceUSD": "BTC_Price"}
)


# Clean data
for df in [eth, btc]:

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        utc=True
    ).dt.tz_convert(None)

    df.iloc[:, 1] = pd.to_numeric(
        df.iloc[:, 1],
        errors="coerce"
    )


eth = eth.dropna(subset=["Date", "ETH_Price"])
btc = btc.dropna(subset=["Date", "BTC_Price"])

eth = eth[eth["ETH_Price"] > 0]
btc = btc[btc["BTC_Price"] > 0]

eth = eth.drop_duplicates(subset=["Date"])
btc = btc.drop_duplicates(subset=["Date"])


# Merge ETH and BTC
df = pd.merge(
    eth,
    btc,
    on="Date",
    how="inner"
)

df = df.sort_values("Date").reset_index(drop=True)


# Summary
print("\nDATA CLEANING SUMMARY")
print("=" * 40)

print("ETH observations:", len(eth))
print("BTC observations:", len(btc))
print("Merged observations:", len(df))

print("Start date:", df["Date"].min())
print("End date:", df["Date"].max())

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate dates:", df["Date"].duplicated().sum())

# Save
output_path = PROCESSED_DIR / "crypto_dataset.csv"

df.to_csv(output_path, index=False)

print("\nSaved:", output_path)
print("\nData cleaning completed!")