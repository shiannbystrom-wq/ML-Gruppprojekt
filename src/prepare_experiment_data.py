
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "crypto_features.csv"
TARGETS_PATH = PROJECT_ROOT / "data" / "processed" / "crypto_targets.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "experiment_dataset.csv"


def main():

    print("PREPARING SHARED EXPERIMENT DATASET")
    print("=" * 45)

    # Load features and targets
    features = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    targets = pd.read_csv(TARGETS_PATH, parse_dates=["Date"])

    print(f"Feature observations: {len(features)}")
    print(f"Target observations: {len(targets)}")

    
    # Keep only target columns from the targets dataset
    target_columns = [
        "Date",
        "ETH_Future_Price_6M",
        "ETH_Return_6M",
        "ETH_Future_Price_12M",
        "ETH_Return_12M"
    ]

    targets = targets[target_columns]

    # Merge features and targets
    dataset = pd.merge(
        features,
        targets,
        on="Date",
        how="inner"
    )

    # Sort chronologically
    dataset = dataset.sort_values("Date").reset_index(drop=True)



    # Sort chronologically
    dataset = dataset.sort_values("Date").reset_index(drop=True)

    # Save dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print("\nEXPERIMENT DATASET CREATED")
    print("=" * 45)

    print(f"Dataset shape: {dataset.shape}")
    print(f"Start date: {dataset['Date'].min().date()}")
    print(f"End date: {dataset['Date'].max().date()}")

    print("\nMissing values:")
    print(dataset.isnull().sum())

    print("\nColumns:")
    print(dataset.columns.tolist())

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

