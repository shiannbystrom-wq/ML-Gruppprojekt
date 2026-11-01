
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "experiment_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

TRAIN_START = "2015-08-08"
TRAIN_END = "2022-12-31"

VALIDATION_START = "2023-01-01"
VALIDATION_END = "2024-12-31"

TEST_START = "2025-01-01"
TEST_END = "2026-10-31"

FEATURE_COLUMNS = [
    "ETH_Return",
    "BTC_Return",
    "ETH_MA30",
    "ETH_MA200",
    "ETH_Volatility30",
    "BTC_ETH_Correlation30"
]

def create_splits(df, horizon):

    target_column = f"ETH_Return_{horizon}"

    # Keep observations with valid target and features
    required_columns = FEATURE_COLUMNS + [target_column]

    data = df.dropna(subset=required_columns).copy()

    # Keep only relevant target columns
    target_price_column = f"ETH_Future_Price_{horizon}"

    columns_to_keep = [
        "Date",
        "ETH_Price",
        "BTC_Price"
    ] + FEATURE_COLUMNS + [
        target_price_column,
        target_column
    ]

    data = data[columns_to_keep]

    # Date-based splits
    train = data[
        (data["Date"] >= TRAIN_START) &
        (data["Date"] <= TRAIN_END)
    ].copy()

    validation = data[
        (data["Date"] >= VALIDATION_START) &
        (data["Date"] <= VALIDATION_END)
    ].copy()

    test = data[
        (data["Date"] >= TEST_START) &
        (data["Date"] <= TEST_END)
    ].copy()

    # Purge samples whose future target windows
    # cross into the next dataset period.
    train_cutoff = pd.Timestamp(TRAIN_END) - pd.DateOffset(months=int(horizon.replace("M", "")))

    validation_cutoff = (
        pd.Timestamp(VALIDATION_END)
        - pd.DateOffset(months=int(horizon.replace("M", "")))
    )

    train = train[train["Date"] <= train_cutoff]
    validation = validation[validation["Date"] <= validation_cutoff]

    return train, validation, test


def main():

    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for horizon in ["6M", "12M"]:

        train, validation, test = create_splits(df, horizon)

        print(f"\n{'=' * 40}")
        print(f"{horizon} EXPERIMENT")
        print(f"{'=' * 40}")

        for name, dataset in [
            ("Train", train),
            ("Validation", validation),
            ("Test", test)
        ]:

            print(
                f"{name}: {len(dataset)} observations | "
                f"{dataset['Date'].min()} - {dataset['Date'].max()}"
                if not dataset.empty
                else f"{name}: 0 observations"
            )

            output_path = OUTPUT_DIR / f"{horizon}_{name.lower()}.csv"
            dataset.to_csv(output_path, index=False)

    print("\nTIME SPLITS CREATED SUCCESSFULLY")


if __name__ == "__main__":
    main()

