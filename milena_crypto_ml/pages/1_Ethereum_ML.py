
# Ethereum – Machine Learning & Accumulation Zones

from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Ethereum – ML",
    page_icon="📊",
    layout="wide"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        st.error(f"Filen saknas: {path}")
        st.stop()
    return pd.read_csv(path)

# Läs in resultaten
daily = load_csv("eth_daily_accumulation_zones.csv")
history = load_csv("eth_backtest_history.csv")
models = load_csv("eth_final_model_comparison.csv")
zones = load_csv("eth_final_buy_zones.csv")
backtest = load_csv("eth_final_backtesting.csv")
yearly = load_csv("eth_final_yearly_zones.csv")

daily["time"] = pd.to_datetime(daily["time"], utc=True)
history["time"] = pd.to_datetime(history["time"], utc=True)

# Titel
st.title("Ethereum – Machine Learning & Accumulation Zones")

st.write(
    "Historisk analys av Ethereum med logaritmisk "
    "regression, maskininlärning och backtesting."
)

st.caption(
    "Historiska resultat och dagliga observationer, "
    "inte livekurser eller investeringsråd."
)

st.divider()

# Översikt
st.header("1. Översikt")

latest = daily.dropna(subset=["ETH_Price"]).iloc[-1]

c1, c2, c3 = st.columns(3)

c1.metric(
    "Senaste ETH-pris",
    f"${latest['ETH_Price']:,.2f}"
)

c2.metric(
    "Regressionszon",
    str(latest["Price_Zone"])
)

c3.metric(
    "Historiska observationer",
    f"{len(daily):,}"
)

st.caption(
    f"Senaste observation: "
    f"{latest['time'].strftime('%Y-%m-%d')} UTC"
)

# Diagram 1: Pris och regressionsband
st.divider()
st.header("2. ETH-pris och regressionszoner")

scale = st.radio(
    "Välj prisskala",
    ["Logaritmisk", "Linjär"],
    horizontal=True
)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=daily["time"],
    y=daily["Upper_Band"],
    name="Övre band",
    line=dict(color="#2ECC71", width=1)
))

fig.add_trace(go.Scatter(
    x=daily["time"],
    y=daily["Lower_Band"],
    name="Nedre band",
    fill="tonexty",
    fillcolor="rgba(46,204,113,0.15)",
    line=dict(color="#2ECC71", width=1)
))

fig.add_trace(go.Scatter(
    x=daily["time"],
    y=daily["Regression_Trend"],
    name="Expanding Regression",
    line=dict(color="#F39C12", width=2)
))

fig.add_trace(go.Scatter(
    x=daily["time"],
    y=daily["ETH_Price"],
    name="ETH-pris",
    line=dict(color="#5DADE2", width=1.5)
))

lower = daily[daily["Price_Zone"] == "Nedre zon"]

fig.add_trace(go.Scatter(
    x=lower["time"],
    y=lower["ETH_Price"],
    name="Nedre zon",
    mode="markers",
    marker=dict(color="#FF4B4B", size=4)
))

fig.update_layout(
    title="Ethereum – Historiska regressionszoner",
    template="plotly_dark",
    height=600,
    yaxis_type="log" if scale == "Logaritmisk" else "linear",
    yaxis_title="ETH-pris (USD)",
    xaxis_title="Datum",
    hovermode="x unified"
)

fig.update_xaxes(rangeslider_visible=True)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "Nedre zon betyder att ETH-priset ligger under "
    "modellens nedre regressionsband. "
    "Det innebär inte automatiskt en köprekommendation."
)

# Diagram 2: Modelljämförelse
st.divider()
st.header("3. Machine Learning – Modelljämförelse")

horizon = st.selectbox(
    "Välj prognoshorisont",
    ["6M", "12M"]
)

selected = models[
    models["Horisont"] == horizon
].copy()

st.dataframe(
    selected.round(4),
    use_container_width=True,
    hide_index=True
)

comparison = selected.melt(
    id_vars="Modell",
    value_vars=["Validering_Brier", "Test_Brier"],
    var_name="Period",
    value_name="Brier Score"
)

comparison["Period"] = comparison["Period"].replace({
    "Validering_Brier": "Validering",
    "Test_Brier": "Senare test"
})

fig_models = px.bar(
    comparison,
    x="Modell",
    y="Brier Score",
    color="Period",
    barmode="group",
    text_auto=".3f",
    title=f"Modellernas Brier Score – {horizon}",
    template="plotly_dark"
)

st.plotly_chart(fig_models, use_container_width=True)

st.caption(
    "Lägre Brier Score är bättre. Den senare "
    "testperioden har även undersökts tidigare "
    "och är inte ett helt oberoende sluttest."
)

# Diagram 3: Köpzoner
st.divider()
st.header("4. Historiska köpzoner och avkastning")

zone_data = zones[
    zones["Horisont"] == horizon
].copy()

st.dataframe(
    zone_data.round(2),
    use_container_width=True,
    hide_index=True
)

fig_return = px.bar(
    zone_data,
    x="Priszon",
    y="Medianavkastning (%)",
    color="Priszon",
    text_auto=".1f",
    title=f"Framtida medianavkastning – {horizon}",
    template="plotly_dark"
)

fig_return.add_hline(y=0, line_dash="dash")
st.plotly_chart(fig_return, use_container_width=True)

fig_risk = px.bar(
    zone_data,
    x="Priszon",
    y="Förlust över 30 % (%)",
    color="Priszon",
    text_auto=".1f",
    title=f"Andel utfall med förlust större än 30 % – {horizon}",
    template="plotly_dark"
)

fig_risk.update_yaxes(range=[0, 100])
st.plotly_chart(fig_risk, use_container_width=True)

# Diagram 4: Backtesting
st.divider()
st.header("5. Backtesting – Strategi vs Buy & Hold")

st.dataframe(
    backtest.round(2),
    use_container_width=True,
    hide_index=True
)

fig_bt = go.Figure()

fig_bt.add_trace(go.Scatter(
    x=history["time"],
    y=history["Strategy_Value"],
    name="Ackumulationsstrategi"
))

fig_bt.add_trace(go.Scatter(
    x=history["time"],
    y=history["BuyHold_Value"],
    name="Buy & Hold"
))

fig_bt.update_layout(
    title="Kapitalutveckling från 10 000 USD",
    template="plotly_dark",
    yaxis_title="Portföljvärde (USD)",
    height=550
)

st.plotly_chart(fig_bt, use_container_width=True)

# Slutsats
st.divider()
st.header("6. Sammanfattande slutsats")

st.markdown("""
**Expanding Regression** gav lägst Brier Score
under den senare historiska testperioden
för både 6 och 12 månader.

Den nedre regressionszonen hade historiskt
högre medianavkastning och färre stora förluster
än den neutrala zonen.

Däremot gav **Buy & Hold** betydligt högre
total avkastning än vår ackumulationsstrategi
under perioden 2019–2026.

Den långsiktiga logaritmiska prisprojektionen
visade dessutom stora historiska prognosfel.

**Slutsats:** Regressionszonerna visar
intressanta historiska samband, men vi har
inte visat att modellen kan identifiera
framtida lönsamma köpperioder med tillräcklig
tillförlitlighet.
""")

st.caption(
    "Projekt inom statistisk programmering och "
    "maskininlärning. Historisk explorativ analys."
)
