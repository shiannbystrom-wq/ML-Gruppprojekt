# Ethereum Accumulation Zones

### Regression-Based Price Bands and Historical Return Analysis

## 1. Project Overview

This project investigates whether regression-based price bands can identify historical Ethereum (ETH) price zones associated with more favorable long-term returns.

The analysis uses historical ETH and Bitcoin (BTC) market data to explore price trends, market relationships, and subsequent investment returns over 6-month and 12-month periods.

The project combines exploratory data analysis, regression methods, time-series analysis, and historical return evaluation.

## 2. Research Question

**Can regression-based price bands identify historical Ethereum price zones associated with more favorable long-term returns?**

The project aims to investigate:

* Long-term Ethereum price trends.
* Regression-based price bands.
* Relationships between Bitcoin and Ethereum.
* Historical returns following different Ethereum price zones.
* The stability of regression-based methods across different market periods.

## 3. Data Source

Historical cryptocurrency data is retrieved using the Coin Metrics Community API.

* Provider: Coin Metrics
* Assets: Ethereum (ETH) and Bitcoin (BTC)
* Metric: PriceUSD
* Frequency: Daily

API documentation:

https://docs.coinmetrics.io/api/v4/

The project uses a Python API integration script to retrieve historical observations and handle pagination.

Raw downloaded data is stored locally and excluded from Git version control.

## 4. Project Structure

```text
ML-Gruppprojekt/
│
├── app/
│
├── data/
│   ├── raw/
│   └── processed/
│       └── splits/
│
├── notebooks/
│
├── reports/
│   └── figures/
│
├── src/
│   ├── download_api_data.py
│   ├── validate_api_data.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── create_targets.py
│   ├── eda.py
│   ├── prepare_experiment_data.py
│   └── create_time_splits.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

## 5. Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Statsmodels
* Requests
* Streamlit

## 6. Installation

Clone the repository:

```bash
git clone https://github.com/shiannbystrom-wq/ML-Gruppprojekt.git
cd ML-Gruppprojekt
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install -r requirements-lock.txt
```

## 7. Data Pipeline

Run the scripts in the following order.

### Step 1: Download data

```bash
python src/download_api_data.py
```

### Step 2: Validate raw data

```bash
python src/validate_api_data.py
```

### Step 3: Clean and merge data

```bash
python src/data_cleaning.py
```

### Step 4: Generate features

```bash
python src/feature_engineering.py
```

### Step 5: Generate future returns

```bash
python src/create_targets.py
```

### Step 6: Generate exploratory analysis figures

```bash
python src/eda.py
```

### Step 7: Prepare the shared experiment dataset

```bash
python src/prepare_experiment_data.py
```

### Step 8: Create chronological experiment splits

```bash
python src/create_time_splits.py
```

## 8. Features and Targets

Current engineered features include:

* ETH daily returns.
* BTC daily returns.
* ETH 30-day moving average.
* ETH 200-day moving average.
* ETH 30-day rolling volatility.
* 30-day BTC–ETH rolling correlation.

Future return targets:

* ETH 6-month future return.
* ETH 12-month future return.

The future return columns are outcome variables and must not be used as model input features.

## 9. Experimental Design

The project uses chronological training, validation, and test datasets rather than random splitting.

Separate datasets are generated for 6-month and 12-month future return experiments.

Target-window purging is applied near split boundaries to reduce temporal leakage.

The final regression methods and evaluation framework will be agreed upon by the project team.

## 10. Team Collaboration

All team members should:

* Use the same data source.
* Follow the shared data pipeline.
* Use the agreed feature and target definitions.
* Preserve chronological splitting.
* Avoid future information leakage.
* Work on separate Git branches.
* Submit changes through Pull Requests.

See `reports/TEAM_GUIDE.md` for detailed instructions.

## 11. Project Status

Current completed work:

* Coin Metrics API integration.
* Historical ETH and BTC data retrieval.
* Data validation and cleaning.
* Feature engineering.
* Exploratory data analysis.
* Future return target generation.
* Shared experiment dataset.
* Chronological experiment splits.

Individual regression experiments, model comparison, backtesting, and the final application remain to be developed.

---

**Academic project – Machine Learning, Nackademin.**
