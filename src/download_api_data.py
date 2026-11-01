import requests
import pandas as pd
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

API_URL = (
    "https://community-api.coinmetrics.io/v4/"
    "timeseries/asset-metrics"
)

ASSETS = ["eth", "btc"]

METRIC = "PriceUSD"

START_TIME = "2015-01-01"


def download_asset_data(asset):

    print(f"\nDownloading {asset.upper()} from Coin Metrics API...")

    params = {
        "assets": asset,
        "metrics": METRIC,
        "frequency": "1d",
        "start_time": START_TIME,
        "page_size": 10000
    }

    all_data = []
    next_page_token = None

    while True:

        if next_page_token:
            params["page_token"] = next_page_token

        response = requests.get(
            API_URL,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        all_data.extend(result.get("data", []))

        next_page_token = result.get("next_page_token")

        if not next_page_token:
            break

        time.sleep(0.2)

    df = pd.DataFrame(all_data)

    if df.empty:
        raise ValueError(f"No data returned for {asset.upper()}")

    df = df[["time", METRIC]]

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")

    df = df.dropna(subset=["time"])
    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"])

    output_path = RAW_DIR / f"{asset}_api.csv"

    df.to_csv(output_path, index=False)

    print("Observations:", len(df))
    print("Start:", df["time"].min())
    print("End:", df["time"].max())
    print("Saved:", output_path)


for asset in ASSETS:
    download_asset_data(asset)

print("\nCoin Metrics API download completed!")