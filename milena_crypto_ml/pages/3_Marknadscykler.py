
# ============================================================
# STEG 30 – MARKNADSCYKLER OCH TRENDANALYS
# Ethereum & Bitcoin
#
# Historiska Bear-episoder
# SMA-baserade Bull/Bear-faser
# Moving Averages och BMSB
# Historisk avkastning och risk
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots
from pathlib import Path


# ============================================================
# 1. SIDINSTÄLLNINGAR
# ============================================================

st.set_page_config(
    page_title="Marknadscykler & Trendanalys",
    page_icon="📈",
    layout="wide"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PHASE_COLORS = {
    "Bull Market": "#2ECC71",
    "Bear Market": "#FF5252",
    "Övergångsfas": "#A0A0A0",
    "Otillräcklig data": "#555555"
}

DRAWDOWN_THRESHOLD = 50
MIN_DECLINE_DAYS = 90


# ============================================================
# 2. LADDA CSV-FILER
# ============================================================

@st.cache_data
def load_market_data(asset):

    path = DATA_DIR / f"{asset.lower()}_market_analysis.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Filen saknas: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "time",
        f"{asset}_Price",
        "SMA_200",
        "SMA_20W",
        "EMA_21W",
        "BMSB_Upper",
        "BMSB_Lower",
        "BMSB_Position",
        "Market_Phase",
        "Confirmed_Phase"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Kolumner saknas: {missing}"
        )

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    numeric_cols = [
        f"{asset}_Price",
        "SMA_200",
        "SMA_20W",
        "EMA_21W",
        "BMSB_Upper",
        "BMSB_Lower"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = (
        df.sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 3. IDENTIFIERA HISTORISKA BEAR-EPISODER
# ============================================================

@st.cache_data
def find_historical_bear_cycles(
    source,
    asset,
    threshold=50,
    min_days=90
):

    df = source.copy()

    price_col = f"{asset}_Price"

    df = df[
        df[price_col].notna()
        & (df[price_col] > 0)
    ].copy()

    df = df.sort_values("time").reset_index(drop=True)

    if df.empty:
        return pd.DataFrame()

    # Historiskt högsta pris
    df["Historical_High"] = (
        df[price_col].cummax()
    )

    # Nedgång från tidigare topp
    df["Drawdown_Pct"] = (
        df[price_col] / df["Historical_High"] - 1
    ) * 100

    previous_high = df["Historical_High"].shift()

    # Datum där priset når eller återtar
    # ett historiskt högsta pris
    new_high = (
        df[price_col].ge(previous_high)
        | previous_high.isna()
    )

    peak_indices = np.flatnonzero(
        new_high.to_numpy()
    )

    episodes = []

    for i, peak_idx in enumerate(peak_indices):

        if i + 1 < len(peak_indices):
            recovery_idx = int(peak_indices[i + 1])
            end_idx = recovery_idx - 1
        else:
            recovery_idx = None
            end_idx = len(df) - 1

        if end_idx <= peak_idx:
            continue

        segment = df.iloc[
            peak_idx:end_idx + 1
        ]

        trough_idx = int(
            segment[price_col].idxmin()
        )

        peak_price = df.loc[
            peak_idx, price_col
        ]

        trough_price = df.loc[
            trough_idx, price_col
        ]

        peak_date = df.loc[
            peak_idx, "time"
        ]

        trough_date = df.loc[
            trough_idx, "time"
        ]

        decline = (
            trough_price / peak_price - 1
        ) * 100

        decline_days = (
            trough_date - peak_date
        ).days

        # Minst 50 % fall och 90 dagar
        if (
            decline > -threshold
            or decline_days < min_days
        ):
            continue

        if recovery_idx is not None:

            end_date = df.loc[
                recovery_idx, "time"
            ]

            status = "Återhämtad"

        else:

            end_date = df["time"].iloc[-1]

            status = "Ej återhämtad vid dataslut"

        episodes.append({

            "Valuta": asset,

            "Episod": len(episodes) + 1,

            "Start": peak_date,

            "Botten": trough_date,

            "Slut": end_date,

            "Topppris": peak_price,

            "Bottenpris": trough_price,

            "Max nedgång (%)": decline,

            "Dagar till botten": decline_days,

            "Varaktighet (dagar)": (
                end_date - peak_date
            ).days,

            "Status": status
        })

    return pd.DataFrame(
        episodes,
        columns=[
            "Valuta",
            "Episod",
            "Start",
            "Botten",
            "Slut",
            "Topppris",
            "Bottenpris",
            "Max nedgång (%)",
            "Dagar till botten",
            "Varaktighet (dagar)",
            "Status"
        ]
    )


# ============================================================
# 4. MARKERA HISTORISKA BEAR-PERIODER
# ============================================================

def add_historical_episode_column(df, episodes):

    result = df.copy()

    result["Historical_Bear"] = False

    for _, episode in episodes.iterrows():

        mask = (
            (result["time"] >= episode["Start"])
            & (result["time"] <= episode["Slut"])
        )

        result.loc[
            mask, "Historical_Bear"
        ] = True

    return result


# ============================================================
# 5. HISTORISK FRAMTIDA AVKASTNING
# ============================================================

def calculate_future_returns(df, asset, horizon):

    work = df.copy()

    work = work.set_index("time")

    price_col = f"{asset}_Price"

    price = work[price_col]

    # Exakt framtida kalenderdatum
    future = price.copy()

    future.index = (
        future.index
        - pd.Timedelta(days=horizon)
    )

    future_price = future.reindex(
        work.index
    )

    work["Future_Return"] = (
        future_price / price - 1
    ) * 100

    valid = work[
        work["BMSB_Position"].isin([
            "Över BMSB",
            "Inom BMSB",
            "Under BMSB"
        ])
        & work["Future_Return"].notna()
        & price.gt(0)
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    summary = (
        valid.groupby("BMSB_Position")
        .agg(
            Observationer=(
                "Future_Return", "size"
            ),

            Medianavkastning=(
                "Future_Return", "median"
            ),

            Medelavkastning=(
                "Future_Return", "mean"
            ),

            Positiva=(
                "Future_Return",
                lambda x: (x > 0).mean() * 100
            ),

            Stora_forluster=(
                "Future_Return",
                lambda x: (x < -20).mean() * 100
            )
        )
        .reset_index()
    )

    return summary.rename(columns={
        "BMSB_Position": "Position",
        "Medianavkastning": "Medianavkastning (%)",
        "Medelavkastning": "Medelavkastning (%)",
        "Positiva": "Positiva utfall (%)",
        "Stora_forluster": "Förlust över 20 % (%)"
    })


# ============================================================
# 6. RUBRIK
# ============================================================

st.title("Marknadscykler & Trendanalys")

st.markdown("""
### Ethereum och Bitcoin – historiska marknadsfaser

Vi jämför två metoder:

**Metod A – SMA-baserade trendfaser**

Klassificering med SMA 200,
dess riktning och valbar
14 dagars bekräftelse.

**Metod B – Större historiska Bear-episoder**

Historiska nedgångar på minst 50 %
med minst 90 dagar mellan topp och botten.

Vi kompletterar analyserna med
Moving Averages, BMSB och historisk avkastning.
""")

st.caption(
    "Historisk analys, inte livekurser "
    "eller investeringsråd."
)

st.divider()


# ============================================================
# 7. SIDOMENY
# ============================================================

st.sidebar.header("Analysinställningar")

asset = st.sidebar.selectbox(
    "Välj kryptovaluta",
    ["ETH", "BTC"],
    format_func=lambda x: {
        "ETH": "Ethereum",
        "BTC": "Bitcoin"
    }[x]
)

mode = st.sidebar.radio(
    "Visa marknadsanalys",
    [
        "Båda metoderna",
        "Historiska Bear-episoder",
        "SMA-baserade trendfaser"
    ]
)

phase_method = st.sidebar.radio(
    "SMA-klassificering",
    [
        "14 dagars bekräftelse",
        "Ursprunglig metod"
    ]
)

st.sidebar.subheader("Moving Averages")

show_sma200 = st.sidebar.checkbox(
    "SMA 200",
    value=True
)

show_sma20w = st.sidebar.checkbox(
    "20W SMA",
    value=False
)

show_ema21w = st.sidebar.checkbox(
    "21W EMA",
    value=False
)

show_bmsb = st.sidebar.checkbox(
    "Bull Market Support Band",
    value=True
)

st.sidebar.subheader("Historiska cykler")

show_peaks = st.sidebar.checkbox(
    "Visa historiska toppar och bottnar",
    value=True
)


# ============================================================
# 8. LADDA OCH FÖRBERED DATA
# ============================================================

try:
    df = load_market_data(asset)

except (FileNotFoundError, ValueError) as e:

    st.error(f"Kunde inte läsa data: {e}")

    st.info(
        "Kontrollera eth_market_analysis.csv "
        "och btc_market_analysis.csv i data-mappen."
    )

    st.stop()


price_col = f"{asset}_Price"

phase_col = (
    "Confirmed_Phase"
    if phase_method == "14 dagars bekräftelse"
    else "Market_Phase"
)

df["Selected_Phase"] = (
    df[phase_col]
    .fillna("Otillräcklig data")
)

episodes = find_historical_bear_cycles(
    df,
    asset,
    DRAWDOWN_THRESHOLD,
    MIN_DECLINE_DAYS
)

df = add_historical_episode_column(
    df,
    episodes
)

show_historical = mode in [
    "Båda metoderna",
    "Historiska Bear-episoder"
]

show_sma_phases = mode in [
    "Båda metoderna",
    "SMA-baserade trendfaser"
]


# ============================================================
# 9. ÖVERSIKT
# ============================================================

st.header("1. Marknadsöversikt")

latest = df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Senaste registrerade pris",
    f"${latest[price_col]:,.2f}"
)

c2.metric(
    "SMA-marknadsfas",
    latest["Selected_Phase"]
)

c3.metric(
    "BMSB-position",
    latest["BMSB_Position"]
)

c4.metric(
    "Historiska Bear-episoder",
    len(episodes)
)

st.caption(
    "Senaste datapunkt: "
    f"{df['time'].max().strftime('%Y-%m-%d')} UTC. "
    "Uppdatera CSV-filerna för nyare observationer."
)

st.divider()


# ============================================================
# 10. FÖRDELNING AV SMA-MARKNADSFASER
# ============================================================

st.header("2. SMA-baserade marknadsfaser")

phase_summary = (
    df["Selected_Phase"]
    .value_counts()
    .rename_axis("Marknadsfas")
    .reset_index(name="Antal dagar")
)

phase_summary["Andel (%)"] = (
    phase_summary["Antal dagar"] / len(df) * 100
)

left, right = st.columns([1, 2])

with left:

    st.dataframe(
        phase_summary.round(2),
        hide_index=True,
        use_container_width=True
    )

with right:

    fig_pie = px.pie(
        phase_summary,
        names="Marknadsfas",
        values="Antal dagar",
        hole=0.45,
        color="Marknadsfas",
        color_discrete_map=PHASE_COLORS,
        title=f"{asset} – Fördelning av SMA-trendfaser",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

st.divider()


# ============================================================
# 11. GEMENSAMT INTERAKTIVT DIAGRAM
# ============================================================

st.header("3. Historiska cykler och Moving Averages")

st.markdown("""
**Historiska Bear-episoder** markeras
med röda bakgrundsområden i prisgrafen.

**SMA-trendfaser** visas i ett separat
färgat band under prisgrafen.

På så sätt kan båda metoderna jämföras
utan att deras färger överlappar.
""")

# Separat rad för SMA-trendstatus
if show_sma_phases:

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.88, 0.12]
    )

else:

    fig = make_subplots(
        rows=1,
        cols=1
    )


# ------------------------------------------------------------
# 11.1 HISTORISKA BEAR-EPISODER
# ------------------------------------------------------------

if show_historical:

    for _, ep in episodes.iterrows():

        fig.add_vrect(
            x0=ep["Start"],
            x1=ep["Slut"],
            fillcolor="rgba(255,82,82,0.17)",
            opacity=1,
            line_width=0,
            layer="below",
            row=1,
            col=1
        )


# ------------------------------------------------------------
# 11.2 BMSB-BAND
# ------------------------------------------------------------

if show_bmsb:

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["BMSB_Lower"],
            mode="lines",
            name="Nedre BMSB",
            line=dict(
                color="#27AE60",
                width=1
            ),
            legendgroup="BMSB"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["BMSB_Upper"],
            mode="lines",
            name="Övre BMSB",
            line=dict(
                color="#2ECC71",
                width=1
            ),
            fill="tonexty",
            fillcolor="rgba(46,204,113,0.15)",
            legendgroup="BMSB"
        ),
        row=1,
        col=1
    )


# ------------------------------------------------------------
# 11.3 HISTORISKT PRIS
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df[price_col],
        mode="lines",
        name=f"{asset}-pris",
        line=dict(
            color="#5DADE2",
            width=2
        )
    ),
    row=1,
    col=1
)


# ------------------------------------------------------------
# 11.4 SMA 200, 20W SMA OCH 21W EMA
# ------------------------------------------------------------

indicators = [
    ("SMA_200", "SMA 200", "#F39C12", show_sma200),
    ("SMA_20W", "20W SMA", "#A569BD", show_sma20w),
    ("EMA_21W", "21W EMA", "#F4D03F", show_ema21w)
]

for col, name, color, enabled in indicators:

    if enabled:

        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[col],
                mode="lines",
                name=name,
                line=dict(
                    color=color,
                    width=2
                )
            ),
            row=1,
            col=1
        )


# ------------------------------------------------------------
# 11.5 TOPPAR OCH BOTTNAR
# ------------------------------------------------------------

if show_historical and show_peaks and not episodes.empty:

    fig.add_trace(
        go.Scatter(
            x=episodes["Start"],
            y=episodes["Topppris"],
            mode="markers",
            name="Historiska toppar",
            marker=dict(
                color="#FF5252",
                symbol="triangle-down",
                size=11
            ),
            customdata=episodes[["Max nedgång (%)"]],
            hovertemplate=(
                "Topp: %{x}<br>"
                "Pris: $%{y:,.2f}<br>"
                "Nedgång till botten: %{customdata[0]:.2f}%"
                "<extra></extra>"
            )
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=episodes["Botten"],
            y=episodes["Bottenpris"],
            mode="markers",
            name="Historiska bottnar",
            marker=dict(
                color="#00CC96",
                symbol="triangle-up",
                size=11
            )
        ),
        row=1,
        col=1
    )


# ------------------------------------------------------------
# 11.6 SEPARAT SMA-TRENDBAND
# ------------------------------------------------------------

if show_sma_phases:

    phase_values = df["Selected_Phase"]

    # Ett färgat märke per observationsdag
    fig.add_trace(
        go.Scattergl(
            x=df["time"],
            y=np.zeros(len(df)),
            mode="markers",
            name="SMA-trendstatus",
            marker=dict(
                symbol="square",
                size=9,
                color=phase_values.map(
                    PHASE_COLORS
                ).fillna("#555555")
            ),
            customdata=phase_values,
            hovertemplate=(
                "Datum: %{x}<br>"
                "SMA-fas: %{customdata}"
                "<extra></extra>"
            ),
            showlegend=False
        ),
        row=2,
        col=1
    )

    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[-0.5, 0.5],
        title_text="SMA-fas",
        row=2,
        col=1
    )


# ------------------------------------------------------------
# 11.7 DIAGRAMINSTÄLLNINGAR
# ------------------------------------------------------------

fig.update_yaxes(
    type="log",
    title_text="Pris (USD, logaritmisk skala)",
    row=1,
    col=1
)

fig.update_layout(
    title=(
        f"{asset} – Historiska cykler, "
        "SMA och BMSB"
    ),
    template="plotly_dark",
    height=720,
    hovermode="x unified",
    legend_title="Indikatorer"
)

last_row = 2 if show_sma_phases else 1

fig.update_xaxes(
    title_text="Datum",
    rangeslider_visible=True,
    row=last_row,
    col=1
)

st.plotly_chart(
    fig,
    use_container_width=True
)

if show_sma_phases:

    st.markdown(
        "🟢 SMA Bull Market &nbsp;&nbsp; "
        "🔴 SMA Bear Market &nbsp;&nbsp; "
        "⚪ Övergångsfas"
    )

st.info(
    "Historiska Bear-episoder identifieras i efterhand. "
    "SMA-klassificeringen är däremot beräknad med "
    "information från aktuella och tidigare datum. "
    "Metoderna mäter olika egenskaper hos marknaden."
)

st.divider()


# ============================================================
# 12. HISTORISKA BEAR-EPISODER
# ============================================================

st.header("4. Större historiska Bear-episoder")

st.markdown("""
Här visas nedgångar på minst 50 %
från tidigare pristopp, där det gått
minst 90 dagar från toppen till botten.

En episod avslutas när den tidigare
toppen återtas eller när datan tar slut.
""")

if episodes.empty:

    st.warning(
        "Inga större Bear-episoder hittades "
        "med den valda definitionen."
    )

else:

    episodes_display = episodes.copy()

    for col in ["Start", "Botten", "Slut"]:

        episodes_display[col] = (
            episodes_display[col]
            .dt.strftime("%Y-%m-%d")
        )

    st.dataframe(
        episodes_display.round(2),
        hide_index=True,
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Antal större Bear-episoder",
        len(episodes)
    )

    c2.metric(
        "Djupaste historiska nedgång",
        f"{episodes['Max nedgång (%)'].min():.2f}%"
    )

    c3.metric(
        "Median dagar till botten",
        f"{episodes['Dagar till botten'].median():.0f}"
    )

st.divider()


# ============================================================
# 13. JÄMFÖR HISTORISKA CYKLER MED SMA
# ============================================================

st.header("5. Jämförelse mellan metoderna")

comparison_data = df[
    df["Selected_Phase"] != "Otillräcklig data"
].copy()

comparison_data["Historisk episod"] = np.where(
    comparison_data["Historical_Bear"],
    "Inom större Bear-episod",
    "Utanför större Bear-episod"
)

comparison_summary = (
    comparison_data
    .groupby("Historisk episod")
    .agg(
        Observationer=("Selected_Phase", "size"),

        SMA_Bull=(
            "Selected_Phase",
            lambda x: (x == "Bull Market").mean() * 100
        ),

        SMA_Bear=(
            "Selected_Phase",
            lambda x: (x == "Bear Market").mean() * 100
        ),

        SMA_Transition=(
            "Selected_Phase",
            lambda x: (x == "Övergångsfas").mean() * 100
        )
    )
    .reset_index()
)

comparison_summary = comparison_summary.rename(
    columns={
        "SMA_Bull": "SMA Bull (%)",
        "SMA_Bear": "SMA Bear (%)",
        "SMA_Transition": "SMA Övergång (%)"
    }
)

st.dataframe(
    comparison_summary.round(2),
    hide_index=True,
    use_container_width=True
)

chart_data = comparison_summary.melt(
    id_vars=["Historisk episod"],
    value_vars=[
        "SMA Bull (%)",
        "SMA Bear (%)",
        "SMA Övergång (%)"
    ],
    var_name="SMA-klassificering",
    value_name="Andel (%)"
)

fig_compare = px.bar(
    chart_data,
    x="Historisk episod",
    y="Andel (%)",
    color="SMA-klassificering",
    barmode="group",
    text_auto=".1f",
    title=(
        f"{asset} – SMA-faser inom och "
        "utanför större Bear-episoder"
    ),
    template="plotly_dark",
    color_discrete_map={
        "SMA Bull (%)": "#2ECC71",
        "SMA Bear (%)": "#FF5252",
        "SMA Övergång (%)": "#A0A0A0"
    }
)

fig_compare.update_layout(
    height=480,
    yaxis_range=[0, 100]
)

st.plotly_chart(
    fig_compare,
    use_container_width=True
)

st.caption(
    "Procenttalen avser dagar i respektive grupp. "
    "Historiska episoder omfattar även återhämtningen "
    "efter botten."
)

st.divider()


# ============================================================
# 14. BMSB OCH HISTORISK FRAMTIDA AVKASTNING
# ============================================================

st.header("6. BMSB och historisk avkastning")

horizon = st.selectbox(
    "Välj horisont",
    [30, 180],
    format_func=lambda x: f"{x} dagar"
)

return_summary = calculate_future_returns(
    df,
    asset,
    horizon
)

if not return_summary.empty:

    st.dataframe(
        return_summary.round(2),
        hide_index=True,
        use_container_width=True
    )

    fig_return = px.bar(
        return_summary,
        x="Position",
        y="Medianavkastning (%)",
        color="Position",
        text_auto=".2f",
        category_orders={
            "Position": [
                "Under BMSB",
                "Inom BMSB",
                "Över BMSB"
            ]
        },
        color_discrete_map={
            "Över BMSB": "#2ECC71",
            "Inom BMSB": "#F1C40F",
            "Under BMSB": "#FF5252"
        },
        title=(
            f"{asset} – Medianavkastning "
            f"efter {horizon} dagar"
        ),
        template="plotly_dark"
    )

    fig_return.add_hline(
        y=0,
        line_dash="dash",
        line_color="white"
    )

    fig_return.update_layout(
        height=480,
        showlegend=False
    )

    st.plotly_chart(
        fig_return,
        use_container_width=True
    )

st.caption(
    "Framtida avkastningsperioder kan överlappa. "
    "Historiska samband är inte validerade "
    "framtida handelsprognoser."
)

st.divider()


# ============================================================
# 15. SENASTE OBSERVATIONERNA
# ============================================================

st.header("7. Senaste historiska observationerna")

latest_columns = [
    "time",
    price_col,
    "SMA_200",
    "SMA_20W",
    "EMA_21W",
    "BMSB_Position",
    "Selected_Phase",
    "Historical_Bear"
]

latest_table = df[
    latest_columns
].tail(30).copy()

latest_table["time"] = (
    latest_table["time"]
    .dt.strftime("%Y-%m-%d")
)

latest_table = latest_table.rename(columns={
    "time": "Datum",
    price_col: "Pris (USD)",
    "BMSB_Position": "BMSB-position",
    "Selected_Phase": "SMA-fas",
    "Historical_Bear": "Historisk Bear-episod"
})

st.dataframe(
    latest_table.round(2),
    hide_index=True,
    use_container_width=True
)

st.divider()


# ============================================================
# 16. EXPORTERA ANALYSDATA
# ============================================================

st.header("8. Exportera analysdata")

c1, c2 = st.columns(2)

with c1:

    csv = df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Ladda ner analysdata (CSV)",
        data=csv,
        file_name=f"{asset.lower()}_market_analysis_full.csv",
        mime="text/csv"
    )

with c2:

    if not episodes.empty:

        csv_episodes = episodes.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Ladda ner historiska Bear-episoder",
            data=csv_episodes,
            file_name=f"{asset.lower()}_bear_episodes.csv",
            mime="text/csv"
        )

st.divider()


# ============================================================
# 17. METOD OCH SLUTSATS
# ============================================================

st.header("9. Metod och slutsats")

st.markdown("""
### Metod A – SMA-baserad trendklassificering

Bull Market definieras som pris över
SMA 200 samtidigt som SMA 200 stiger.

Bear Market definieras som pris under
SMA 200 samtidigt som SMA 200 faller.

I den bekräftade versionen krävs
14 dagar med samma trendvillkor.

### Metod B – Historiska Bear-episoder

Vi identifierar historiska nedgångar
på minst 50 % från en tidigare pristopp.

Minst 90 dagar ska ha passerat mellan
pristoppen och botten.

Episoden omfattar tiden från toppen
till återhämtningen eller dataslutet.

### Moving Averages och BMSB

SMA 200 beskriver en långsiktig trend
baserad på dagliga priser.

20W SMA och 21W EMA bildar tillsammans
Bull Market Support Band.

BMSB kan användas som en kompletterande
indikator för marknadens trendförhållanden.

### Viktig skillnad mellan metoderna

Historiska Bear-episoder identifieras
med facit från senare prisutveckling.

SMA-metoden beskriver i stället
trendförhållanden med aktuell och
tidigare historisk information.

### Slutsats

Metoderna beskriver olika egenskaper
hos marknaden och kompletterar varandra.

De kan användas tillsammans med gruppens
Shared Curvature Quantile Regression
för att visualisera historisk marknadskontext.

Resultaten utgör inte investeringsråd
eller bevis för framtida prognosförmåga.
""")

with st.expander("Källor och definitioner"):

    st.markdown("""
    - [SEC – Bear Market](https://www.investor.gov/introduction-investing/investing-basics/glossary/bear-market)
    - [Coinbase Institutional Research](https://www.coinbase.com/en-au/institutional/research-insights/research/monthly-outlook/monthly-outlook-apr-2025)
    - [Galaxy Research – Bitcoin Market Cycles](https://www.galaxy.com/insights/research/bitcoin-50-week-moving-average-bear-market-ceiling-200-week-floor-analysis)

    Gränserna 50 % och 90 dagar är
    experimentella antaganden i denna analys
    och ska inte betraktas som en
    universell marknadsdefinition.
    """)

st.caption(
    "Projekt inom statistisk programmering "
    "och maskininlärning."
)
