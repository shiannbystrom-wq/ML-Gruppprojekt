
# ============================================================
# THE OVERFITTERS
# Ethereum & Bitcoin – Machine Learning
# Streamlit Dashboard
# ============================================================

# STEG 1 – Importera bibliotek

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import requests

# ============================================================
# STEG 2 – APPENS GRUNDINSTÄLLNINGAR
# ============================================================

st.set_page_config(
    page_title="The Overfitters – Crypto ML",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# STEG 3 – LÄS IN DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PRICE_FILE = DATA_DIR / "crypto_historical_prices.csv"
ML_FILE = DATA_DIR / "crypto_ml_with_trends.csv"


@st.cache_data
def load_csv(file_path):
    data = pd.read_csv(file_path)

    data["time"] = pd.to_datetime(
        data["time"],
        utc=True
    )

    data = data.sort_values("time").reset_index(drop=True)

    return data


# Kontrollera att filerna finns
if not PRICE_FILE.exists():
    st.error(f"Prisfilen saknas: {PRICE_FILE}")
    st.stop()

if not ML_FILE.exists():
    st.error(f"ML-filen saknas: {ML_FILE}")
    st.stop()


# Läs in historiska priser
df = load_csv(PRICE_FILE)

# Läs in fullständig dataset
df_stats = load_csv(ML_FILE)

# Utökad trendhistorik från 2015
TREND_FILE = DATA_DIR / "crypto_trends_2015.csv"

df_trend_data = load_csv(TREND_FILE)


# ============================================================
# STEG 75 – HÄMTA SENASTE PRISER VIA COIN METRICS API
# ============================================================

@st.cache_data(ttl=3600)
def load_latest_api_prices():

    url = (
        "https://community-api.coinmetrics.io"
        "/v4/timeseries/asset-metrics"
    )

    params = {
        "assets": "btc,eth",
        "metrics": "PriceUSD",
        "frequency": "1d",
        "limit_per_asset": 2,
        "paging_from": "end"
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = pd.DataFrame(response.json()["data"])

    data["time"] = pd.to_datetime(
        data["time"],
        utc=True
    )

    data["PriceUSD"] = pd.to_numeric(
        data["PriceUSD"],
        errors="coerce"
    )

    data = data.dropna(subset=["PriceUSD"])

    if data.empty:
        raise ValueError("API:et returnerade inga giltiga priser.")

    return data.sort_values(["asset", "time"])


# API:et anropas bara vid behov i Dashboard-fliken.


# Kontrollera datan
required_columns = ["time", "ETH_Price", "BTC_Price"]

for col in required_columns:
    if col not in df.columns:
        st.error(f"Kolumn saknas: {col}")
        st.stop()


# ============================================================
# STEG 4 – TITEL OCH INTRODUKTION
# ============================================================

st.title("Ethereum & Bitcoin – Machine Learning")

st.subheader("The Overfitters")

st.write(
    "En interaktiv analys av Ethereum och Bitcoin "
    "med statistik, maskininlärning och backtesting."
)


# ============================================================
# STEG 5 – SKAPA APPENS FEM FLIKAR
# ============================================================

tabs = st.tabs([
    "Dashboard",
    "Statistisk analys",
    "Trendanalys",
    "Machine Learning",
    "Backtesting"
])


# ============================================================
# FLIK 1 – DASHBOARD
# ============================================================

with tabs[0]:

    st.header("Crypto Dashboard")
    
    # STEG 75 – API-priser

    st.subheader("Senaste tillgängliga API-priser")

    if st.button("Uppdatera API-priser"):
        load_latest_api_prices.clear()

    try:
        api_prices = load_latest_api_prices()

        latest_api = (
            api_prices.groupby("asset")
            .tail(1)
            .set_index("asset")
        )

        if not {"eth", "btc"}.issubset(latest_api.index):
            raise ValueError(
                "API-svaret saknar ETH eller BTC."
            )

        col_eth, col_btc = st.columns(2)

        with col_eth:
            st.metric(
                "Ethereum – API",
                f"${latest_api.loc['eth', 'PriceUSD']:,.2f}"
            )
            st.caption(
                "Observation: "
                f"{latest_api.loc['eth', 'time']:%Y-%m-%d} UTC"
            )

        with col_btc:
            st.metric(
                "Bitcoin – API",
                f"${latest_api.loc['btc', 'PriceUSD']:,.2f}"
            )
            st.caption(
                "Observation: "
                f"{latest_api.loc['btc', 'time']:%Y-%m-%d} UTC"
            )

        st.caption(
            "Källa: Coin Metrics Community API. "
            "Dagliga observationer – inte livekurser."
        )

    except (requests.RequestException, ValueError, KeyError) as e:
        st.warning(
            "Kunde inte hämta API-priser just nu. "
            "De historiska CSV-diagrammen fungerar fortfarande."
        )
        st.caption(f"Teknisk information: {e}")

    st.divider()


    # Senaste observerade priserna
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    eth_change = (
        latest["ETH_Price"] / previous["ETH_Price"] - 1
    ) * 100

    btc_change = (
        latest["BTC_Price"] / previous["BTC_Price"] - 1
    ) * 100

    # Nyckeltal
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Ethereum (ETH)",
            f"${latest['ETH_Price']:,.2f}",
            f"{eth_change:+.2f}%"
        )

    with col2:
        st.metric(
            "Bitcoin (BTC)",
            f"${latest['BTC_Price']:,.2f}",
            f"{btc_change:+.2f}%"
        )

    with col3:
        st.metric(
            "Historiska observationer",
            f"{len(df):,}"
        )

    st.caption(
        f"Senaste observation: "
        f"{latest['time'].strftime('%Y-%m-%d')} UTC. "
        "Historiska priser, inte livekurser."
    )

    st.divider()

    # --------------------------------------------------------
    # INTERAKTIVA PRISDIAGRAM
    # --------------------------------------------------------

    st.subheader("Historisk prisutveckling")

    # Välj tidsperiod
    period = st.selectbox(
        "Välj tidsperiod",
        [
            "1 månad",
            "3 månader",
            "1 år",
            "3 år",
            "Hela historiken"
        ],
        index=2
    )

    latest_date = df["time"].max()

    periods = {
        "1 månad": 30,
        "3 månader": 90,
        "1 år": 365,
        "3 år": 1095
    }

    if period == "Hela historiken":
        df_filtered = df.copy()

    else:
        start_date = latest_date - pd.Timedelta(
            days=periods[period]
        )

        df_filtered = df[
            df["time"] >= start_date
        ].copy()

    # Välj vilket diagram som ska visas
    chart_type = st.radio(
        "Välj diagram",
        [
            "Ethereum",
            "Bitcoin",
            "Jämför ETH och BTC"
        ],
        horizontal=True
    )

    # Ethereum
    if chart_type == "Ethereum":

        fig = px.line(
            df_filtered,
            x="time",
            y="ETH_Price",
            title="Ethereum – Historiskt pris i USD"
        )

        y_title = "Pris (USD)"

    # Bitcoin
    elif chart_type == "Bitcoin":

        fig = px.line(
            df_filtered,
            x="time",
            y="BTC_Price",
            title="Bitcoin – Historiskt pris i USD"
        )

        y_title = "Pris (USD)"

    # Jämförelse mellan ETH och BTC
    else:

        comparison = df_filtered.copy()

        comparison["Ethereum"] = (
            comparison["ETH_Price"]
            / comparison["ETH_Price"].iloc[0] * 100
        )

        comparison["Bitcoin"] = (
            comparison["BTC_Price"]
            / comparison["BTC_Price"].iloc[0] * 100
        )

        fig = px.line(
            comparison,
            x="time",
            y=["Ethereum", "Bitcoin"],
            title="ETH vs BTC – Normaliserad prisutveckling"
        )

        y_title = "Prisindex (start = 100)"

    # Diagraminställningar
    fig.update_layout(
        xaxis_title="Datum",
        yaxis_title=y_title,
        hovermode="x unified",
        legend_title_text="Kryptovaluta",
        height=600,
        dragmode="zoom"
    )

    # Datumreglage och zoomknappar
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[
                dict(
                    count=1,
                    label="1M",
                    step="month",
                    stepmode="backward"
                ),
                dict(
                    count=3,
                    label="3M",
                    step="month",
                    stepmode="backward"
                ),
                dict(
                    count=1,
                    label="1Y",
                    step="year",
                    stepmode="backward"
                ),
                dict(
                    count=3,
                    label="3Y",
                    step="year",
                    stepmode="backward"
                ),
                dict(
                    step="all",
                    label="ALL"
                )
            ]
        )
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.caption(
        "Historiska priser från Coin Metrics. "
        "Normaliserat prisindex används för att jämföra "
        "kryptovalutornas relativa utveckling."
    )


# ============================================================
# FLIK 2 – STATISTISK ANALYS
# ============================================================

with tabs[1]:

    st.header("Statistisk analys – ETH & BTC")

    st.write(
        "Utforska historiska data, statistiska samband "
        "och volatilitet för Ethereum och Bitcoin."
    )

    st.divider()

    # --------------------------------------------------------
    # DEL 1 – BESKRIVANDE STATISTIK
    # --------------------------------------------------------

    st.subheader("1. Beskrivande statistik (EDA)")

    # Hitta numeriska variabler
    numeric_columns = [
        col
        for col in df_stats.select_dtypes(
            include="number"
        ).columns
        if not col.startswith(
            ("ETH_Up_", "ETH_Future_")
        )
    ]

    selected_variables = st.multiselect(
        "Välj variabler för statistisk analys",
        options=numeric_columns,
        default=[
            "ETH_Price",
            "BTC_Price",
            "ETH_Return",
            "BTC_Return"
        ]
    )

    if selected_variables:

        summary = (
            df_stats[selected_variables]
            .describe()
            .T
        )

        # Lägg till median
        summary["Median"] = (
            df_stats[selected_variables].median()
        )

        summary = summary[
            [
                "count",
                "mean",
                "Median",
                "std",
                "min",
                "max"
            ]
        ]

        st.dataframe(
            summary.round(2),
            width="stretch"
        )

    else:
        st.info("Välj minst en variabel.")

    st.divider()

    # --------------------------------------------------------
    # DEL 2 – KORRELATIONSANALYS
    # --------------------------------------------------------

    st.subheader("2. Korrelationsanalys")

    st.write(
        "Undersök linjära samband mellan variabler "
        "med Pearsons korrelationskoefficient."
    )

    corr_options = [
        "ETH_Return",
        "BTC_Return",
        "ETH_RSI_14",
        "BTC_RSI_14",
        "ETH_Volatility_30",
        "BTC_Volatility_30",
        "ETH_Volume_USD",
        "BTC_Volume_USD",
        "Fear_Greed_Index",
        "VIX",
        "SP500"
    ]

    corr_options = [
        col for col in corr_options
        if col in df_stats.columns
    ]

    corr_selected = st.multiselect(
        "Välj variabler till korrelationsmatris",
        options=corr_options,
        default=corr_options[:6]
    )

    if len(corr_selected) >= 2:

        # Begränsa till historisk utvecklingsperiod
        corr_data = df_stats.loc[
            df_stats["time"] < pd.Timestamp(
                "2023-01-01",
                tz="UTC"
            ),
            corr_selected
        ]

        correlation_matrix = corr_data.corr(
            method="pearson"
        )

        # Skapa interaktiv heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Korrelationsmatris – träningsperiod"
        )

        fig_corr.update_layout(
            height=650
        )

        st.plotly_chart(
            fig_corr,
            width="stretch"
        )

        st.caption(
            "Korrelationer beräknade på perioden före 2023. "
            "Korrelation innebär inte orsakssamband."
        )

    else:
        st.info(
            "Välj minst två variabler för korrelationsanalys."
        )

    st.divider()

    # --------------------------------------------------------
    # DEL 3 – HISTORISK VOLATILITET
    # --------------------------------------------------------

    st.subheader("3. Historisk volatilitet")

    st.write(
        "Jämför 30-dagars volatilitet för Ethereum "
        "och Bitcoin över tid."
    )

    volatility_df = df_stats[
        [
            "time",
            "ETH_Volatility_30",
            "BTC_Volatility_30"
        ]
    ].copy()

    fig_vol = px.line(
        volatility_df,
        x="time",
        y=[
            "ETH_Volatility_30",
            "BTC_Volatility_30"
        ],
        title="30-dagars volatilitet – ETH vs BTC"
    )

    fig_vol.update_layout(
        xaxis_title="Datum",
        yaxis_title="Volatilitet (daglig avkastning, %)",
        hovermode="x unified",
        legend_title_text="Kryptovaluta",
        height=500
    )

    st.plotly_chart(
        fig_vol,
        width="stretch"
    )

    st.caption(
        "Volatiliteten beräknas som standardavvikelsen "
        "för daglig procentuell avkastning under 30 dagar."
    )


# ============================================================
# FLIK 3 – TRENDANALYS
# ============================================================


# ============================================================
# FLIK 3 – TRENDANALYS
# ============================================================

with tabs[2]:

    st.header("Trendanalys – ETH/BTC")

    st.write(
        "Undersök Ethereums utveckling relativt Bitcoin "
        "med hjälp av glidande medelvärden och en "
        "dynamisk regressionskanal."
    )

    # Välj historisk tidsperiod
    trend_period = st.selectbox(
        "Välj tidsperiod för trendanalys",
        ["1 år", "3 år", "5 år", "Hela historiken"],
        index=1
    )

    trend_days = {
        "1 år": 365,
        "3 år": 1095,
        "5 år": 1825
    }

    trend_df = df_trend_data.copy()

    if trend_period != "Hela historiken":
        start = trend_df["time"].max() - pd.Timedelta(
            days=trend_days[trend_period]
        )
        trend_df = trend_df[
            trend_df["time"] >= start
        ].copy()

    # NYCKELTAL
    latest_trend = trend_df.iloc[-1]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "ETH/BTC Ratio",
            f"{latest_trend['ETH_BTC_Ratio']:.5f}"
        )

    with c2:
        st.metric(
            "Avstånd från trend",
            f"{latest_trend['Trend_Distance_Pct']:+.2f}%"
        )

    with c3:
        st.metric(
            "Antal nedåtgående trendbrott",
            int(trend_df["Breakdown"].sum())
        )

    st.divider()

    # KONTROLLERA KOLUMNER
    required_trend = [
        "ETH_BTC_Ratio",
        "Rolling_Trend",
        "Trend_Zone",
        "Breakdown"
    ]

    if not all(
        col in trend_df.columns for col in required_trend
    ):
        st.error("Några trendkolumner saknas i datasetet.")
        st.stop()

    # Trendkanalens övre och nedre band beräknas
    # från originaldatasetet i notebooken senare.
    # Tills vidare används de faktiska banden om
    # de finns sparade i CSV-filen.

    if (
        "Upper_Band" in trend_df.columns
        and "Lower_Band" in trend_df.columns
    ):
        upper = trend_df["Upper_Band"]
        lower = trend_df["Lower_Band"]
    else:
        st.warning(
            "Övre och nedre trendband saknas i CSV-filen. "
            "Vi visar trendlinjen tills banden har exporterats."
        )
        upper = None
        lower = None

    import plotly.graph_objects as go

    fig_trend = go.Figure()

    # Faktisk ETH/BTC
    fig_trend.add_trace(go.Scatter(
        x=trend_df["time"],
        y=trend_df["ETH_BTC_Ratio"],
        mode="lines",
        name="ETH/BTC",
        line=dict(color="#5DADE2", width=1.5)
    ))

    # Dynamisk regression
    fig_trend.add_trace(go.Scatter(
        x=trend_df["time"],
        y=trend_df["Rolling_Trend"],
        mode="lines",
        name="Rolling Regression",
        line=dict(color="#2ECC71", width=2.5)
    ))

    # Gröna trendband
    if upper is not None and lower is not None:

        fig_trend.add_trace(go.Scatter(
            x=trend_df["time"],
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))

        fig_trend.add_trace(go.Scatter(
            x=trend_df["time"],
            y=lower,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(46,204,113,0.18)",
            line=dict(width=0),
            name="Trendkanal",
            hoverinfo="skip"
        ))

    # Markera nedåtgående trendbrott
    breakdown_points = trend_df[
        trend_df["Breakdown"] == 1
    ]

    fig_trend.add_trace(go.Scatter(
        x=breakdown_points["time"],
        y=breakdown_points["ETH_BTC_Ratio"],
        mode="markers",
        name="Trendbrott",
        marker=dict(
            color="#FF5B5B",
            size=8,
            symbol="triangle-down"
        )
    ))

    fig_trend.update_layout(
        title="ETH/BTC – Dynamisk trendkanal",
        xaxis_title="Datum",
        yaxis_title="ETH/BTC (logaritmisk skala)",
        yaxis_type="log",
        hovermode="x unified",
        height=650
    )

    fig_trend.update_xaxes(
        rangeslider_visible=True
    )

    st.plotly_chart(
        fig_trend,
        width="stretch"
    )

    st.caption(
        "Historiska trendlinjer och trendbrott. "
        "Indikatorerna är inte köp- eller säljsignaler."
    )

    # ========================================================
    # STEG 88 – ETH SMA 200 OCH BULL MARKET SUPPORT BAND
    # ========================================================

    st.divider()

    st.header("Ethereum – Långsiktig trendanalys")

    st.write(
        "Vi undersöker Ethereums långsiktiga prisutveckling "
        "med SMA 200 och Bull Market Support Band."
    )

    # Välj skala
    price_scale = st.radio(
        "Välj prisskala",
        ["Linjär", "Logaritmisk"],
        horizontal=True,
        key="eth_trend_scale"
    )

    use_log = price_scale == "Logaritmisk"

    # Använd vår historiska prisfil från 2015
    eth_df = df[
        ["time", "ETH_Price"]
    ].copy()

    eth_df = eth_df.sort_values("time")

    eth_df["ETH_Price"] = pd.to_numeric(
        eth_df["ETH_Price"],
        errors="coerce"
    )

    # Ta bort perioder utan giltiga ETH-priser
    eth_df = eth_df[
        eth_df["ETH_Price"] > 0
    ].copy()

    # --------------------------------------------------------
    # DEL 1 – SMA 200
    # --------------------------------------------------------

    st.subheader("1. Ethereum – SMA 200")

    # 200 dagars glidande medelvärde
    eth_df["SMA_200"] = (
        eth_df["ETH_Price"]
        .rolling(window=200, min_periods=200)
        .mean()
    )

    fig_sma = px.line(
        eth_df,
        x="time",
        y=["ETH_Price", "SMA_200"],
        title="Ethereum – Pris och 200 dagars SMA"
    )

    fig_sma.update_layout(
        xaxis_title="Datum",
        yaxis_title="ETH-pris (USD)",
        yaxis_type="log" if use_log else "linear",
        hovermode="x unified",
        height=550,
        legend_title_text="Indikator"
    )

    fig_sma.update_xaxes(
        rangeslider_visible=True
    )

    st.plotly_chart(
        fig_sma,
        width="stretch"
    )

    st.caption(
        "SMA 200 visar medelpriset under de senaste "
        "200 dagarna. De första 199 giltiga "
        "observationerna saknar SMA-värde."
    )

    # --------------------------------------------------------
    # DEL 2 – BULL MARKET SUPPORT BAND
    # --------------------------------------------------------

    st.divider()

    st.subheader("2. Bull Market Support Band")

    st.write(
        "Bull Market Support Band består av "
        "20 veckors SMA och 21 veckors EMA."
    )

    # Gör om dagliga priser till veckopriser
    weekly = (
        eth_df.set_index("time")["ETH_Price"]
        .resample("W-SUN")
        .last()
        .dropna()
        .to_frame()
    )

    # 20 veckors SMA
    weekly["SMA_20W"] = (
        weekly["ETH_Price"]
        .rolling(window=20, min_periods=20)
        .mean()
    )

    # 21 veckors EMA
    weekly["EMA_21W"] = (
        weekly["ETH_Price"]
        .ewm(
            span=21,
            adjust=False,
            min_periods=21
        )
        .mean()
    )

    # Bandets gränser
    weekly["Upper_Band"] = weekly[
        ["SMA_20W", "EMA_21W"]
    ].max(axis=1)

    weekly["Lower_Band"] = weekly[
        ["SMA_20W", "EMA_21W"]
    ].min(axis=1)

    weekly = weekly.reset_index()

    # Bygg interaktivt Plotly-diagram
    fig_bmsb = go.Figure()

    # Övre band
    fig_bmsb.add_trace(go.Scatter(
        x=weekly["time"],
        y=weekly["Upper_Band"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Nedre band med fyllning
    fig_bmsb.add_trace(go.Scatter(
        x=weekly["time"],
        y=weekly["Lower_Band"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(46, 204, 113, 0.22)",
        line=dict(width=0),
        name="Bull Market Support Band",
        hoverinfo="skip"
    ))

    # ETH-pris
    fig_bmsb.add_trace(go.Scatter(
        x=weekly["time"],
        y=weekly["ETH_Price"],
        mode="lines",
        name="Ethereum",
        line=dict(color="#5DADE2", width=1.5)
    ))

    # SMA 20 veckor
    fig_bmsb.add_trace(go.Scatter(
        x=weekly["time"],
        y=weekly["SMA_20W"],
        mode="lines",
        name="20W SMA",
        line=dict(color="#F39C12", width=2)
    ))

    # EMA 21 veckor
    fig_bmsb.add_trace(go.Scatter(
        x=weekly["time"],
        y=weekly["EMA_21W"],
        mode="lines",
        name="21W EMA",
        line=dict(color="#2ECC71", width=2)
    ))

    fig_bmsb.update_layout(
        title="Ethereum – Bull Market Support Band",
        xaxis_title="Datum",
        yaxis_title="ETH-pris (USD)",
        yaxis_type="log" if use_log else "linear",
        hovermode="x unified",
        height=600
    )

    fig_bmsb.update_xaxes(
        rangeslider_visible=True
    )

    st.plotly_chart(
        fig_bmsb,
        width="stretch"
    )

    st.caption(
        "BMSB beräknas från ETH-prisets sista "
        "tillgängliga observation varje vecka. "
        "Veckorna avslutas på söndagar (UTC). "
        "Indikatorn beskriver historisk trend och "
        "är inte en verifierad prognosmodell."
    )

    # --------------------------------------------------------
    # DEL 3 – NYCKELTAL
    # --------------------------------------------------------

    st.divider()

    st.subheader("3. Senaste trendindikatorer")

    latest_eth = eth_df.iloc[-1]
    latest_week = weekly.iloc[-1]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "ETH-pris",
            f"${latest_eth['ETH_Price']:,.2f}"
        )

    with col2:
        st.metric(
            "SMA 200",
            f"${latest_eth['SMA_200']:,.2f}"
        )

    with col3:
        st.metric(
            "20W SMA",
            f"${latest_week['SMA_20W']:,.2f}"
        )

    st.caption(
        f"Senaste veckobaserade observation: "
        f"{latest_week['time']:%Y-%m-%d} UTC"
    )



# ============================================================
# FLIK 4 – MACHINE LEARNING
# ============================================================


# ============================================================
# FLIK 4 – MACHINE LEARNING
# ============================================================

with tabs[3]:

    st.header("Machine Learning – Modelljämförelse")

    st.write(
        "Jämför våra maskininlärningsmodeller "
        "för att förutsäga Ethereums framtida prisriktning."
    )

    st.info(
        "Resultaten kommer från en historisk utvärdering "
        "2025–2026. Modellerna har inte visat stabil "
        "prognosförmåga över alla tidsperioder."
    )

    # Läs in exporterade modellresultat
    results_file = DATA_DIR / "ml_model_results.csv"

    if results_file.exists():

        results = load_csv_ml = pd.read_csv(results_file)

        # Välj prognoshorisont
        horizon = st.selectbox(
            "Välj prognoshorisont",
            ["7D", "30D", "6M", "12M"],
            index=1
        )

        selected = results[
            results["Horisont"] == horizon
        ].copy()

        st.subheader("Modellernas resultat")

        st.dataframe(
            selected.round(4),
            width="stretch",
            hide_index=True
        )

        # Jämför Accuracy
        st.subheader("Accuracy – Modelljämförelse")

        fig_accuracy = px.bar(
            selected,
            x="Modell",
            y="Accuracy",
            color="Modell",
            text_auto=".2%",
            title=f"Accuracy – {horizon}"
        )

        fig_accuracy.update_layout(
            yaxis_title="Accuracy",
            yaxis_tickformat=".0%",
            yaxis_range=[0, 1],
            showlegend=False,
            height=450
        )

        st.plotly_chart(
            fig_accuracy,
            width="stretch"
        )

        # Jämför ROC-AUC
        st.subheader("ROC-AUC – Modelljämförelse")

        fig_auc = px.bar(
            selected,
            x="Modell",
            y="ROC-AUC",
            color="Modell",
            text_auto=".3f",
            title=f"ROC-AUC – {horizon}"
        )

        fig_auc.add_hline(
            y=0.5,
            line_dash="dash",
            annotation_text="Referensnivå 0,50"
        )

        fig_auc.update_layout(
            yaxis_range=[0, 1],
            showlegend=False,
            height=450
        )

        st.plotly_chart(
            fig_auc,
            width="stretch"
        )

        st.caption(
            "Baseline använder träningsperiodens vanligaste "
            "klass. ROC-AUC för den konstanta baseline är 0,50. "
            "Resultaten är explorativa och prognosperioderna "
            "överlappar varandra."
        )

    else:
        st.warning(
            "Filen ml_model_results.csv saknas. "
            "Exportera resultaten från steg 69 i notebooken."
        )




# ============================================================
# FLIK 5 – BACKTESTING
# ============================================================

with tabs[4]:

    st.header("Backtesting – Random Forest vs Buy & Hold")

    st.write(
        "Historisk simulering av en ML-baserad strategi "
        "jämfört med att köpa och behålla Ethereum."
    )

    backtest_file = DATA_DIR / "crypto_backtest_results.csv"

    if backtest_file.exists():

        bt = load_csv(backtest_file)

        initial_capital = 10000

        # Beräkna avkastning
        strategy_final = bt["Strategy_Value"].iloc[-1]
        buyhold_final = bt["BuyHold_Value"].iloc[-1]

        strategy_return = (
            strategy_final / initial_capital - 1
        ) * 100

        buyhold_return = (
            buyhold_final / initial_capital - 1
        ) * 100

        # Beräkna maximal drawdown
        def calculate_drawdown(values):
            peak = values.cummax().clip(lower=initial_capital)
            return ((values / peak) - 1).min() * 100

        strategy_dd = calculate_drawdown(bt["Strategy_Value"])
        buyhold_dd = calculate_drawdown(bt["BuyHold_Value"])

        # Nyckeltal
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Random Forest – Avkastning",
                f"{strategy_return:+.2f}%"
            )

        with c2:
            st.metric(
                "Buy & Hold – Avkastning",
                f"{buyhold_return:+.2f}%"
            )

        with c3:
            st.metric(
                "Antal affärer",
                int(bt["Trade"].sum())
            )

        st.divider()

        # Kapitalutveckling
        st.subheader("Kapitalutveckling")

        fig_bt = px.line(
            bt,
            x="time",
            y=["Strategy_Value", "BuyHold_Value"],
            title="Random Forest vs Buy & Hold"
        )

        fig_bt.add_hline(
            y=initial_capital,
            line_dash="dash",
            annotation_text="Startkapital"
        )

        fig_bt.update_layout(
            xaxis_title="Datum",
            yaxis_title="Portföljvärde (USD)",
            hovermode="x unified",
            height=600
        )

        st.plotly_chart(fig_bt, width="stretch")

        # Resultattabell
        st.subheader("Resultat och risk")

        results_bt = pd.DataFrame({
            "Strategi": ["Random Forest", "Buy & Hold"],
            "Slutkapital (USD)": [
                strategy_final,
                buyhold_final
            ],
            "Total avkastning (%)": [
                strategy_return,
                buyhold_return
            ],
            "Max Drawdown (%)": [
                strategy_dd,
                buyhold_dd
            ]
        })

        st.dataframe(
            results_bt.round(2),
            hide_index=True,
            width="stretch"
        )

        st.metric(
            "Andel dagar investerad",
            f"{bt['Position'].mean() * 100:.2f}%"
        )

        st.warning(
            "Backtestingen är en förenklad historisk "
            "simulering. Strategin är inte verifierad "
            "för handel i realtid. Resultaten omfattar "
            "inte slippage eller skatt och bygger på "
            "antagna transaktionskostnader."
        )

    else:
        st.warning(
            "Filen crypto_backtest_results.csv saknas. "
            "Exportera resultaten från steg 70."
        )



# ============================================================
# SLUT PÅ STEG 64
# ============================================================
