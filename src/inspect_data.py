import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

# Load datasets
eth = pd.read_csv(RAW_DIR / "eth.csv")
btc = pd.read_csv(RAW_DIR / "btc.csv")

# Inspect each dataset
datasets = {
    "ETH": eth,
    "BTC": btc
}

for name, df in datasets.items():

    print("\n" + "=" * 50)
    print(f"{name} DATASET")
    print("=" * 50)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())