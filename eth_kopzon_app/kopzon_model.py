"""
Köpzonsmodell för Ethereum – tidsserie, glidande medelvärden och ETH/BTC.

All beräkning ligger i den här filen så att appen (app.py) bara behöver visa
resultaten. Modulen kan även importeras i en notebook:

    from kopzon_model import load_prices, build_features

Principer (enligt TEAM_GUIDE.md):
- Varje värde en viss dag bygger BARA på data fram till och med den dagen.
- Framtida avkastning används endast för utvärdering, aldrig som input.
- Gruppens tidsperioder: Träning 2015–2022, Validering 2023–2024, Test 2025–2026.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# --------------------------------------------------------------------------
# Inställningar
# --------------------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "crypto_dataset.csv"

DAYS_PER_YEAR = 365
DEFAULT_TREND_YEARS = 4      # ≈ en hel kryptocykel (Bitcoin-halvering vart 4:e år)
ZONE_Z = 1.0                 # köpzon: priset minst 1 standardavvikelse under trenden
RATIO_WINDOW = 365           # ETH/BTC jämförs mot sitt eget 1-årssnitt
CORR_WINDOW = 90             # rullande korrelation mellan ETH och BTC

HORIZONS = {"6 mån": 182, "12 mån": 365}

PERIODS = [
    ("Träning (2015–2022)", None, "2022-12-31"),
    ("Validering (2023–2024)", "2023-01-01", "2024-12-31"),
    ("Test (2025–2026)", "2025-01-01", None),
]

ZONE_ORDER = ["Starkt köpläge", "Köpläge", "Neutral", "Dyrt", "Ingen data"]
ZONE_COLORS = {
    "Starkt köpläge": "#008300",
    "Köpläge": "#0ca30c",
    "Neutral": "#8a8984",
    "Dyrt": "#d03b3b",
    "Ingen data": "#c9c8c3",
}


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------

def load_prices(path: str | Path | None = None) -> pd.DataFrame:
    """Läser gruppens tvättade dataset (Date, ETH_Price, BTC_Price)."""
    df = pd.read_csv(path or DATA_PATH, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").set_index("Date")
    df = df.rename(columns={"ETH_Price": "ETH", "BTC_Price": "BTC"})
    return df[["ETH", "BTC"]].astype(float)


# --------------------------------------------------------------------------
# 2. Tidsseriemodell: rullande log-linjär trend
# --------------------------------------------------------------------------

def rolling_trend(price: pd.Series, window_days: int) -> pd.DataFrame:
    """
    Anpassar för varje dag en rät linje till log(pris) över de senaste
    `window_days` dagarna (minsta kvadrat), utan att titta framåt.

    Returnerar:
        trend : trendvärdet idag, i USD
        sd    : spridningen runt trenden (std av residualerna, i log)
        z     : hur många standardavvikelser priset ligger från trenden
    """
    y = np.log(price)
    t = pd.Series(np.arange(len(y), dtype=float), index=y.index)

    roll = lambda s: s.rolling(window_days, min_periods=window_days).mean()
    mean_t, mean_y = roll(t), roll(y)
    var_t = roll(t * t) - mean_t**2
    var_y = roll(y * y) - mean_y**2
    cov_ty = roll(t * y) - mean_t * mean_y

    slope = cov_ty / var_t
    fitted = mean_y + slope * (t - mean_t)            # trendlinjen utvärderad idag
    resid_var = (var_y - slope**2 * var_t).clip(lower=1e-12)
    sd = np.sqrt(resid_var)

    return pd.DataFrame(
        {"trend": np.exp(fitted), "sd": sd, "z": (y - fitted) / sd, "slope_per_year": slope * DAYS_PER_YEAR},
        index=price.index,
    )


# --------------------------------------------------------------------------
# 3. Glidande medelvärden
# --------------------------------------------------------------------------

def moving_averages(price: pd.Series) -> pd.DataFrame:
    """SMA 50, SMA 200 och Bull Market Support Band (20W SMA + 21W EMA)."""
    out = pd.DataFrame(index=price.index)
    out["SMA50"] = price.rolling(50, min_periods=50).mean()
    out["SMA200"] = price.rolling(200, min_periods=200).mean()

    # Veckodata med söndag som veckoslut (samma som i gruppens notebook).
    weekly = price.resample("W-SUN").last()
    sma20w = weekly.rolling(20, min_periods=20).mean()
    ema21w = weekly.ewm(span=21, adjust=False, min_periods=21).mean()
    # Ett veckovärde blir känt på söndagen och gäller fram till nästa söndag.
    out["BMSB_SMA20W"] = sma20w.reindex(price.index, method="ffill")
    out["BMSB_EMA21W"] = ema21w.reindex(price.index, method="ffill")
    return out


# --------------------------------------------------------------------------
# 4. Allt ihop
# --------------------------------------------------------------------------

def classify_zone(eth_z: pd.Series, btc_z: pd.Series, z: float = ZONE_Z) -> pd.Series:
    """
    Starkt köpläge : ETH OCH BTC ligger båda minst z standardavvikelser under trend
    Köpläge        : ETH ligger minst z standardavvikelser under sin trend
    Dyrt           : ETH ligger minst z standardavvikelser över sin trend
    Neutral        : däremellan
    """
    zone = pd.Series("Neutral", index=eth_z.index, dtype=object)
    zone[eth_z > z] = "Dyrt"
    zone[eth_z < -z] = "Köpläge"
    zone[(eth_z < -z) & (btc_z < -z)] = "Starkt köpläge"
    zone[eth_z.isna()] = "Ingen data"
    return zone


def assign_period(index: pd.DatetimeIndex) -> pd.Series:
    period = pd.Series(pd.NA, index=index, dtype=object)
    for name, start, end in PERIODS:
        mask = np.ones(len(index), dtype=bool)
        if start:
            mask &= index >= pd.Timestamp(start)
        if end:
            mask &= index <= pd.Timestamp(end)
        period[mask] = name
    return period


def build_features(prices: pd.DataFrame, trend_years: int = DEFAULT_TREND_YEARS,
                   zone_z: float = ZONE_Z) -> pd.DataFrame:
    """Bygger hela analystabellen som appen visar."""
    window = trend_years * DAYS_PER_YEAR
    df = prices.copy()

    eth = rolling_trend(df["ETH"], window)
    btc = rolling_trend(df["BTC"], window)
    df["ETH_trend"], df["ETH_sd"], df["ETH_z"] = eth["trend"], eth["sd"], eth["z"]
    df["ETH_trend_growth"] = np.exp(eth["slope_per_year"]) - 1
    df["BTC_trend"], df["BTC_z"] = btc["trend"], btc["z"]

    # Gränserna för zonerna uttryckta som priser (för grafen)
    df["Köpzon_gräns"] = df["ETH_trend"] * np.exp(-zone_z * df["ETH_sd"])
    df["Dyrt_gräns"] = df["ETH_trend"] * np.exp(zone_z * df["ETH_sd"])

    df = df.join(moving_averages(df["ETH"]))
    df["BTC_SMA200"] = df["BTC"].rolling(200, min_periods=200).mean()

    # ETH mot BTC
    df["ETH_BTC"] = df["ETH"] / df["BTC"]
    log_ratio = np.log(df["ETH_BTC"])
    ratio_mean = log_ratio.rolling(RATIO_WINDOW, min_periods=RATIO_WINDOW).mean()
    ratio_std = log_ratio.rolling(RATIO_WINDOW, min_periods=RATIO_WINDOW).std()
    df["ETH_BTC_snitt_1år"] = np.exp(ratio_mean)
    df["ETH_BTC_z"] = (log_ratio - ratio_mean) / ratio_std

    ret_eth, ret_btc = df["ETH"].pct_change(), df["BTC"].pct_change()
    df["Korrelation_90d"] = ret_eth.rolling(CORR_WINDOW, min_periods=CORR_WINDOW).corr(ret_btc)

    # Zoner och period
    df["Zon"] = classify_zone(df["ETH_z"], df["BTC_z"], zone_z)
    df["Period"] = assign_period(df.index)

    # Framtida avkastning – ENDAST för utvärdering (raderna saknas i slutet)
    for label, days in HORIZONS.items():
        df[f"Avk_{label}"] = df["ETH"].shift(-days) / df["ETH"] - 1

    return df


# --------------------------------------------------------------------------
# 5. Utvärdering
# --------------------------------------------------------------------------

def zone_return_table(df: pd.DataFrame, horizon: str = "12 mån", by_period: bool = False,
                      group_col: str = "Zon") -> pd.DataFrame:
    """Median, andel positiva och antal dagar per zon (och ev. per period)."""
    col = f"Avk_{horizon}"
    d = df[df[col].notna() & (df[group_col] != "Ingen data")]
    keys = ["Period", group_col] if by_period else [group_col]
    out = d.groupby(keys, observed=True)[col].agg(
        Dagar="count",
        Median=lambda s: s.median(),
        Andel_upp=lambda s: (s > 0).mean(),
    )
    return out.reset_index()


def train_probability_model(df: pd.DataFrame, horizon: str = "12 mån") -> dict:
    """
    Logistisk regression: sannolikheten att ETH är högre om `horizon`,
    givet hur långt ETH (och BTC) ligger från sina trender idag.

    Träning: dagar vars utfall var känt innan valideringsperioden startade
    (2023-01-01) – annars skulle modellen ha sett "framtiden".
    Utvärdering: alla dagar från 2023 och framåt med känt utfall.
    """
    days = HORIZONS[horizon]
    features = ["ETH_z"]
    target_col = f"Avk_{horizon}"

    d = df[features + [target_col]].dropna()
    y = (d[target_col] > 0).astype(int)
    last_train_day = pd.Timestamp("2023-01-01") - pd.Timedelta(days=days + 1)
    train = d.index <= last_train_day
    test = d.index >= pd.Timestamp("2023-01-01")

    model = LogisticRegression().fit(d.loc[train, features], y[train])
    p_test = model.predict_proba(d.loc[test, features])[:, 1]

    all_rows = df[features].dropna()
    prob = pd.Series(model.predict_proba(all_rows)[:, 1], index=all_rows.index)

    return {
        "features": features,
        "coef": dict(zip(features, model.coef_[0])),
        "train_start": d.index[train].min(),
        "train_end": d.index[train].max(),
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "auc": roc_auc_score(y[test], p_test) if y[test].nunique() > 1 else np.nan,
        "accuracy": accuracy_score(y[test], p_test > 0.5),
        "base_rate": y[test].mean(),
        "prob": prob,
    }


def similar_days(df: pd.DataFrame, today_z: float, horizon: str = "12 mån", width: float = 0.25) -> pd.Series:
    """Utfallet för historiska dagar där ETH låg ungefär lika långt från trenden som idag."""
    col = f"Avk_{horizon}"
    d = df[df[col].notna() & df["ETH_z"].between(today_z - width, today_z + width)]
    return d[col]
