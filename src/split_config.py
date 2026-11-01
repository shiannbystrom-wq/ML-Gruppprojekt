from datetime import date

TRAIN_START = date(2015, 8, 8)
TRAIN_END = date(2022, 12, 31)

VALIDATION_START = date(2023, 1, 1)
VALIDATION_END = date(2024, 12, 31)

TEST_START = date(2025, 1, 1)
TEST_END = date(2026, 10, 31)

TARGET_HORIZONS = {
    "6M": 6,
    "12M": 12
}

print("Shared experiment configuration loaded.")