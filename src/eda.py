import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "crypto_features.csv"
PLOT_DIR = BASE_DIR / "reports" / "figures"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

# Figure 1: ETH price and moving averages
plt.figure(figsize=(14, 6))

plt.plot(df["Date"], df["ETH_Price"], label="ETH Price", alpha=0.6)
plt.plot(df["Date"], df["ETH_MA30"], label="MA30")
plt.plot(df["Date"], df["ETH_MA200"], label="MA200")

plt.title("Ethereum Price and Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_DIR / "eth_moving_averages.png", dpi=300)
plt.show()


# Figure 2: ETH and BTC daily returns
plt.figure(figsize=(14, 5))

plt.plot(df["Date"], df["ETH_Return"], label="ETH Daily Return", alpha=0.7)
plt.plot(df["Date"], df["BTC_Return"], label="BTC Daily Return", alpha=0.7)

plt.title("Daily Returns: ETH vs BTC")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_DIR / "daily_returns.png", dpi=300)
plt.show()


# Figure 3: Rolling correlation
plt.figure(figsize=(14, 5))

plt.plot(
    df["Date"],
    df["BTC_ETH_Correlation30"],
    label="30-Day Rolling Correlation"
)

plt.axhline(0, linestyle="--", alpha=0.5)

plt.title("30-Day Rolling Correlation: BTC and ETH")
plt.xlabel("Date")
plt.ylabel("Correlation")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_DIR / "rolling_correlation.png", dpi=300)
plt.show()


print("EDA plots created successfully.")
print("Saved in:", PLOT_DIR)