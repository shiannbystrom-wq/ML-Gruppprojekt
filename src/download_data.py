import pandas as pd
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# Coin Metrics CSV sources
data_sources = {
    "eth": "https://raw.githubusercontent.com/coinmetrics/data/master/csv/eth.csv",
    "btc": "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
}

# Download datasets
for asset, url in data_sources.items():

    print(f"Downloading {asset.upper()} data...")

    df = pd.read_csv(url)

    output_path = RAW_DIR / f"{asset}.csv"

    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("-" * 50)

print("Download completed!")