import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

for asset in ["eth", "btc"]:

    print(f"\n{'=' * 40}")
    print(f"{asset.upper()} API DATA VALIDATION")
    print("=" * 40)

    path = RAW_DIR / f"{asset}_api.csv"

    df = pd.read_csv(path)

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["PriceUSD"] = pd.to_numeric(
        df["PriceUSD"],
        errors="coerce"
    )

    print("Shape:", df.shape)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate dates:", df["time"].duplicated().sum())

    print("\nInvalid prices:", (df["PriceUSD"] <= 0).sum())

    print("\nFirst 3 rows:")
    print(df.head(3))

    print("\nLast 5 rows:")
    print(df.tail(5))

print("\nAPI validation completed.")