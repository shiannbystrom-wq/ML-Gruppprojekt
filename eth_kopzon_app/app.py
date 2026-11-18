"""
Ethereum – köpzoner och relationen till Bitcoin
================================================

Starta från projektets rotmapp:

    streamlit run eth_kopzon_app/app.py

All beräkning görs i kopzon_model.py. Den här filen visar bara resultaten och
förklarar dem på ett sätt som ska gå att förstå utan teknisk bakgrund.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import kopzon_model as km

st.set_page_config(page_title="ETH köpzoner", page_icon="📈", layout="wide")

# Färger (validerad referenspalett): ETH = blå, BTC = orange, trend = grå
C_ETH, C_BTC, C_TREND, C_SECOND = "#2a78d6", "#eb6834", "#52514e", "#6250d6"
C_UP, C_DOWN = "#2a78d6", "#e34948"
ZONE_ICON = {"Starkt köpläge": "🟢", "Köpläge": "🟢", "Neutral": "⚪", "Dyrt": "🔴", "Ingen data": "▫️"}


# --------------------------------------------------------------------------
# Hjälpfunktioner
# --------------------------------------------------------------------------

def usd(x: float) -> str:
    if x >= 100:
        return f"{x:,.0f} USD".replace(",", " ")
    return f"{x:,.2f} USD".replace(",", " ").replace(".", ",")


def pct(x: float, sign: bool = True) -> str:
    if pd.isna(x):
        return "–"
    s = f"{x * 100:+.0f} %" if sign else f"{x * 100:.0f} %"
    return s.replace("-", "−")


def num(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def style(fig: go.Figure, log: bool = False, height: int = 460, y_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", type="log" if log else "linear")
    return fig


def explain(title: str, body: str, expanded: bool = False) -> None:
    with st.expander(f"💡 {title}", expanded=expanded):
        st.markdown(body)


def outcome_table(df: pd.DataFrame, horizon: str, group_col: str = "Zon",
                  order: list[str] | None = None, by_period: bool = False) -> pd.DataFrame:
    t = km.zone_return_table(df, horizon, by_period=by_period, group_col=group_col)
    if order:
        t[group_col] = pd.Categorical(t[group_col], order, ordered=True)
    if by_period:
        t["Period"] = pd.Categorical(t["Period"], [p[0] for p in km.PERIODS], ordered=True)
    sort_cols = (["Period"] if by_period else []) + ([group_col] if order else [])
    if sort_cols:
        t = t.sort_values(sort_cols)
    t = t.rename(columns={group_col: "Läge", "Dagar": "Antal dagar",
                          "Median": f"Typisk avkastning efter {horizon}",
                          "Andel_upp": "Andel gånger priset steg"})
    t[f"Typisk avkastning efter {horizon}"] = t[f"Typisk avkastning efter {horizon}"].map(pct)
    t["Andel gånger priset steg"] = t["Andel gånger priset steg"].map(lambda v: pct(v, sign=False))
    return t


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@st.cache_data
def get_prices() -> pd.DataFrame:
    return km.load_prices()


@st.cache_data
def get_features(trend_years: int, zone_z: float) -> pd.DataFrame:
    return km.build_features(get_prices(), trend_years, zone_z)


@st.cache_data
def get_model(trend_years: int, zone_z: float, horizon: str) -> dict:
    return km.train_probability_model(get_features(trend_years, zone_z), horizon)


@st.cache_data
def window_comparison(zone_z: float, horizon: str) -> pd.DataFrame:
    rows = []
    for years in [1, 2, 3, 4]:
        f = get_features(years, zone_z)
        buy = f["Zon"].isin(["Köpläge", "Starkt köpläge"])
        for name, *_ in km.PERIODS:
            r = f.loc[buy & (f["Period"] == name), f"Avk_{horizon}"].dropna()
            rows.append({"Trendfönster": f"{years} år", "Period": name, "Köpdagar": len(r),
                         "Typisk avkastning": pct(r.median()) if len(r) else "inga köpdagar",
                         "Andel steg": pct((r > 0).mean(), sign=False) if len(r) else "–"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Sidomeny
# --------------------------------------------------------------------------

st.sidebar.header("Inställningar")
trend_years = st.sidebar.select_slider(
    "Hur många år bakåt trenden räknas på", options=[1, 2, 3, 4], value=km.DEFAULT_TREND_YEARS,
    help="4 år ≈ en hel kryptocykel. Kortare fönster följer priset tätare men fungerar sämre historiskt (se flik 3).",
)
zone_z = st.sidebar.select_slider(
    "Hur långt under trenden räknas som köpzon", options=[0.5, 0.75, 1.0, 1.25, 1.5], value=km.ZONE_Z,
    format_func=lambda v: f"{num(v, 2)} standardavvikelser",
    help="Högre värde = strängare krav = färre men 'billigare' köpdagar.",
)
horizon = st.sidebar.radio("Utvärdera avkastning efter", list(km.HORIZONS), index=1, horizontal=True)

df = get_features(trend_years, zone_z)
first_year, last_year = int(df.index.year.min()), int(df.index.year.max())
start_year = st.sidebar.slider("Visa graferna från år", first_year, last_year - 1, first_year)
view = df[df.index.year >= start_year]

st.sidebar.caption(
    f"Data: gruppens tvättade dataset (Coin Metrics), {df.index.min():%Y-%m-%d} – {df.index.max():%Y-%m-%d}. "
    "Historisk analys – inte investeringsrådgivning."
)

today = df.iloc[-1]
today_date = df.index[-1]

# --------------------------------------------------------------------------
# Rubrik och dagens läge
# --------------------------------------------------------------------------

st.title("Ethereum – köpzoner och relationen till Bitcoin")
st.markdown(
    "Appen undersöker en enkel fråga: **finns det lägen då Ethereum (ETH) historiskt har varit billigt, "
    "och har det i så fall lönat sig att köpa då?** Den visar ETH:s prishistorik, en trendmodell som pekar "
    "ut köpzoner, glidande medelvärden och hur ETH rör sig jämfört med Bitcoin (BTC)."
)

c1, c2, c3, c4 = st.columns([1, 1.5, 1, 1])
c1.metric(f"ETH-pris {today_date:%Y-%m-%d}", usd(today["ETH"]))
c2.metric("Läge idag", f"{ZONE_ICON[today['Zon']]} {today['Zon']}")
c3.metric("Avstånd till trendlinjen", pct(today["ETH"] / today["ETH_trend"] - 1),
          help="Hur mycket över (+) eller under (−) ETH:s långsiktiga trend priset ligger.")
c4.metric("ETH jämfört med BTC", pct(today["ETH_BTC"] / today["ETH_BTC_snitt_1år"] - 1),
          help="Hur mycket ETH/BTC-kvoten skiljer sig från sitt snitt det senaste året. Negativt = ETH har gått sämre än BTC.")

tabs = st.tabs(["🏠 Översikt", "1 · Historik", "2 · Glidande medelvärden", "3 · Trend & köpzoner",
                "4 · ETH mot BTC", "5 · Köpläge & prognos", "📖 Ordlista & metod"])

# ==========================================================================
# Översikt
# ==========================================================================
with tabs[0]:
    hist_zone = df[df["Zon"] == today["Zon"]][f"Avk_{horizon}"].dropna()
    st.subheader(f"Är det köpläge idag? {ZONE_ICON[today['Zon']]} {today['Zon']}")

    zone_text = {
        "Starkt köpläge": "Både ETH och BTC ligger ovanligt långt under sina långsiktiga trender – "
                          "hela kryptomarknaden är billig jämfört med de senaste åren.",
        "Köpläge": "ETH ligger ovanligt långt under sin långsiktiga trend.",
        "Neutral": "ETH ligger inom det normala spannet runt sin trend – varken särskilt billigt eller dyrt.",
        "Dyrt": "ETH ligger ovanligt långt över sin långsiktiga trend. Historiskt har det varit ett dåligt läge att köpa.",
        "Ingen data": "Det finns inte tillräckligt med historik för att räkna ut trenden.",
    }[today["Zon"]]

    st.markdown(
        f"""
- ETH kostar **{usd(today['ETH'])}**. Den långsiktiga trendlinjen ligger på **{usd(today['ETH_trend'])}**.
- Priset ligger alltså **{pct(today['ETH'] / today['ETH_trend'] - 1)}** från trenden.
  Gränsen för köpzonen ligger just nu på **{usd(today['Köpzon_gräns'])}**.
- {zone_text}
"""
    )
    if len(hist_zone):
        st.success(
            f"**Vad hände tidigare?** Det har funnits {len(hist_zone)} dagar i läget *{today['Zon']}* där vi vet "
            f"utfallet. {horizon} senare hade ETH stigit i **{pct((hist_zone > 0).mean(), sign=False)}** av fallen, "
            f"och den typiska (median-) avkastningen var **{pct(hist_zone.median())}**."
        )
    st.warning(
        "**Viktigt:** Modellen bygger på drygt tio års historik och bara ett fåtal stora marknadscykler. "
        "Att något fungerat tidigare är ingen garanti för framtiden. Det här är ett skolprojekt, inte investeringsråd."
    )

    st.markdown("#### Så är appen uppbyggd – steg för steg")
    st.markdown(
        """
| Steg | Flik | Vad du får svar på |
|---|---|---|
| 1 | **Historik** | Hur har ETH-priset utvecklats sedan 2015? |
| 2 | **Glidande medelvärden** | Vad säger klassiska trendindikatorer (SMA 200, Bull Market Support Band)? |
| 3 | **Trend & köpzoner** | Vår tidsseriemodell: när har ETH varit ovanligt billigt, och vad hände sedan? |
| 4 | **ETH mot BTC** | Hur hänger ETH och Bitcoin ihop, och hjälper BTC till att hitta bättre köplägen? |
| 5 | **Köpläge & prognos** | Allt ihop: dagens läge, liknande dagar i historien och en enkel ML-prognos. |
"""
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=view["ETH"], name="ETH-pris", line=dict(color=C_ETH, width=2)))
    fig.add_trace(go.Scatter(x=view.index, y=view["ETH_trend"], name="Trendlinje", line=dict(color=C_TREND, width=2, dash="dash")))
    buy = view[view["Zon"].isin(["Köpläge", "Starkt köpläge"])]
    fig.add_trace(go.Scatter(x=buy.index, y=buy["ETH"], name="Köpzon", mode="markers",
                             marker=dict(color=km.ZONE_COLORS["Köpläge"], size=5)))
    st.plotly_chart(style(fig, log=True, height=420, y_title="Pris (USD, logaritmisk skala)"))
    st.caption("Blå linje = ETH-priset. Streckad linje = långsiktig trend. Gröna prickar = dagar i köpzon.")

# ==========================================================================
# 1. Historik
# ==========================================================================
with tabs[1]:
    st.subheader("1 · Ethereums prishistorik")
    st.markdown(
        "Här ser du ETH-priset dag för dag. Det är grunden för allt annat i appen."
    )
    log_hist = st.toggle("Logaritmisk skala", value=True, key="log_hist")
    fig = go.Figure(go.Scatter(x=view.index, y=view["ETH"], name="ETH-pris", line=dict(color=C_ETH, width=2)))
    st.plotly_chart(style(fig, log=log_hist, y_title="Pris (USD)"))

    ath_date = df["ETH"].idxmax()
    drawdown = today["ETH"] / df["ETH"].max() - 1
    a, b, c, d = st.columns(4)
    a.metric("Första pris i datat", usd(df["ETH"].iloc[0]), help=f"{df.index[0]:%Y-%m-%d}")
    b.metric("Högsta pris någonsin", usd(df["ETH"].max()), help=f"{ath_date:%Y-%m-%d}")
    c.metric("Från toppen till idag", pct(drawdown))
    d.metric("Antal dagar i datat", f"{len(df):,}".replace(",", " "))

    yearly = df["ETH"].resample("YE").last().pct_change().dropna()
    yearly.index = yearly.index.year
    fig = go.Figure(go.Bar(
        x=yearly.index, y=yearly.values * 100,
        marker_color=[C_UP if v > 0 else C_DOWN for v in yearly.values],
        hovertemplate="%{x}: %{y:.0f} %<extra></extra>",
    ))
    fig.update_layout(title="Avkastning per kalenderår (%)")
    st.plotly_chart(style(fig, height=320, y_title="%"))
    st.caption(f"{df.index.year.max()} är ett ofullständigt år (data till {today_date:%Y-%m-%d}). Blå = uppgång, röd = nedgång.")

    explain("Varför logaritmisk skala?", """
På en **vanlig skala** är avståndet mellan 10 och 20 dollar lika litet som mellan 4 000 och 4 010 dollar.
Då syns ETH:s första år knappt alls.

På en **logaritmisk skala** är varje *fördubbling* lika stor. Ett steg från 10 till 20 dollar ser ut precis
som ett steg från 2 000 till 4 000 dollar. Det gör att man kan jämföra procentuella rörelser över hela
historiken – och det är därför trendmodellen räknar på logaritmerade priser.
""", expanded=True)
    explain("Vad ska jag lägga märke till?", """
- ETH rör sig i **cykler**: kraftiga uppgångar (2017, 2021) följs av stora fall (2018, 2022).
- Fallen från toppen har ofta varit **70–90 %**. Det betyder att tajming spelar stor roll.
- Tillväxten har blivit **långsammare** med tiden – de första åren var uppgångarna mycket större.
""")

# ==========================================================================
# 2. Glidande medelvärden
# ==========================================================================
with tabs[2]:
    st.subheader("2 · Glidande medelvärden")
    st.markdown(
        "Ett **glidande medelvärde** är genomsnittspriset de senaste N dagarna. Varje dag flyttas fönstret "
        "en dag framåt – därav \"glidande\". Det jämnar ut dagliga svängningar och visar riktningen."
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=view["BMSB_SMA20W"], name="20-veckors snitt",
                             line=dict(color=C_SECOND, width=1), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=view.index, y=view["BMSB_EMA21W"], name="Bull Market Support Band",
                             line=dict(color=C_SECOND, width=1), fill="tonexty",
                             fillcolor="rgba(98,80,214,0.25)"))
    fig.add_trace(go.Scatter(x=view.index, y=view["ETH"], name="ETH-pris", line=dict(color=C_ETH, width=2)))
    fig.add_trace(go.Scatter(x=view.index, y=view["SMA200"], name="SMA 200 (200 dagars snitt)",
                             line=dict(color=C_BTC, width=2)))
    st.plotly_chart(style(fig, log=True, y_title="Pris (USD, logaritmisk skala)"))

    above = (df["ETH"] > df["SMA200"])[df["SMA200"].notna()]
    a, b, c = st.columns(3)
    a.metric("Pris jämfört med SMA 200 idag", pct(today["ETH"] / today["SMA200"] - 1))
    b.metric("Andel dagar över SMA 200", pct(above.mean(), sign=False))
    in_band = min(today["BMSB_SMA20W"], today["BMSB_EMA21W"]) <= today["ETH"] <= max(today["BMSB_SMA20W"], today["BMSB_EMA21W"])
    band_pos = "inom bandet" if in_band else ("över bandet" if today["ETH"] > max(today["BMSB_SMA20W"], today["BMSB_EMA21W"]) else "under bandet")
    c.metric("Läge mot Bull Market Support Band", band_pos)

    st.markdown(f"#### Vad hände efter att priset legat under respektive över SMA 200? ({horizon})")
    tmp = df.assign(SMA_läge=np.where(df["SMA200"].isna(), "Ingen data",
                                      np.where(df["ETH"] < df["SMA200"], "Under SMA 200", "Över SMA 200")))
    st.dataframe(outcome_table(tmp, horizon, "SMA_läge", ["Under SMA 200", "Över SMA 200"]), hide_index=True)

    explain("Så läser du grafen", """
- **Orange linje – SMA 200:** genomsnittspriset de senaste 200 dagarna (ungefär 7 månader). Ligger priset
  *över* linjen brukar marknaden beskrivas som stark, *under* som svag.
- **Lila band – Bull Market Support Band:** området mellan 20-veckors och 21-veckors medelvärde. I stigande
  marknader har priset ofta studsat på bandet; i fallande marknader har bandet fungerat som ett "tak".
""", expanded=True)
    explain("Vad lär vi oss?", """
Glidande medelvärden är bra på att visa **vilken riktning** marknaden har, men de reagerar **sent** –
medelvärdet hinner inte med när priset vänder snabbt. De säger heller inget om priset är *billigt* i
förhållande till den långsiktiga utvecklingen. Därför bygger vi i nästa steg en trendmodell som mäter just det.
Tabellen ovan visar att "under SMA 200" ensamt inte är en särskilt träffsäker köpsignal.
""")

# ==========================================================================
# 3. Trend & köpzoner
# ==========================================================================
with tabs[3]:
    st.subheader("3 · Tidsseriemodellen: trendlinje och köpzoner")
    st.markdown(
        f"""
Modellen gör tre saker **varje dag**, och använder bara priser fram till den dagen:

1. **Ritar en trendlinje** genom ETH-priserna de senaste **{trend_years} åren** (på logaritmisk skala).
2. **Mäter hur långt priset ligger från linjen**, jämfört med hur mycket priset brukar avvika.
   Måttet kallas *z-värde*: 0 = precis på trenden, −1 = ovanligt långt under, +1 = ovanligt långt över.
3. **Sätter ett läge** utifrån z-värdet:
"""
    )
    st.markdown(
        f"""
| Läge | Regel | Tolkning |
|---|---|---|
| 🟢 **Starkt köpläge** | ETH *och* BTC under −{num(zone_z, 2)} | ETH är billigt och hela kryptomarknaden likaså |
| 🟢 **Köpläge** | ETH under −{num(zone_z, 2)} | ETH är ovanligt billigt mot sin trend |
| ⚪ **Neutral** | mellan −{num(zone_z, 2)} och +{num(zone_z, 2)} | Normalt prisläge |
| 🔴 **Dyrt** | ETH över +{num(zone_z, 2)} | ETH är ovanligt dyrt mot sin trend |
"""
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=view["Dyrt_gräns"], name="Gräns: dyrt", line=dict(color=km.ZONE_COLORS["Dyrt"], width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=view.index, y=view["Köpzon_gräns"], name="Gräns: köpzon", line=dict(color=km.ZONE_COLORS["Köpläge"], width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(138,137,132,0.12)"))
    fig.add_trace(go.Scatter(x=view.index, y=view["ETH_trend"], name="Trendlinje", line=dict(color=C_TREND, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=view.index, y=view["ETH"], name="ETH-pris", line=dict(color=C_ETH, width=2)))
    for zone in ["Köpläge", "Starkt köpläge", "Dyrt"]:
        z = view[view["Zon"] == zone]
        fig.add_trace(go.Scatter(x=z.index, y=z["ETH"], mode="markers", name=zone,
                                 marker=dict(color=km.ZONE_COLORS[zone], size=8 if zone == "Starkt köpläge" else 6,
                                             symbol="diamond" if zone == "Starkt köpläge" else "circle",
                                             line=dict(width=1, color="white"))))
    st.plotly_chart(style(fig, log=True, height=520, y_title="Pris (USD, logaritmisk skala)"))
    st.caption(
        "Det grå fältet är det *normala* prisspannet runt trenden. Hamnar priset under den gröna prickade linjen "
        "är vi i köpzon; över den röda prickade linjen räknas ETH som dyrt. Trenden börjar först när det finns "
        f"{trend_years} års historik."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=view["ETH_z"], name="ETH z-värde", line=dict(color=C_ETH, width=2)))
    fig.add_hline(y=-zone_z, line=dict(color=km.ZONE_COLORS["Köpläge"], dash="dot"), annotation_text="köpzon", annotation_position="bottom left")
    fig.add_hline(y=zone_z, line=dict(color=km.ZONE_COLORS["Dyrt"], dash="dot"), annotation_text="dyrt", annotation_position="top left")
    fig.add_hline(y=0, line=dict(color=C_TREND, width=1))
    fig.update_layout(title="Avstånd till trenden (z-värde)")
    st.plotly_chart(style(fig, height=300, y_title="z-värde"))

    st.markdown(f"#### Fungerade det? Avkastning {horizon} efter respektive läge")
    split = st.toggle("Dela upp på gruppens perioder (träning / validering / test)", value=True)
    st.dataframe(outcome_table(df, horizon, "Zon", km.ZONE_ORDER, by_period=split), hide_index=True)
    st.caption(
        "**Typisk avkastning** = medianen (hälften av dagarna gav mer, hälften mindre). "
        "**Andel gånger priset steg** = hur ofta ETH var högre efter perioden. "
        "De sista månaderna saknas eftersom vi ännu inte vet hur det gick."
    )

    explain("Varför en rullande trend och inte en enda linje för hela historiken?", """
ETH:s tillväxt har **bromsat in** över tid. En enda linje för hela perioden (som i en klassisk
log-regression) hamnar därför för högt de senaste åren – då ser ETH aldrig billigt ut längre.
En *rullande* trend räknas om varje dag på de senaste åren och anpassar sig efter att tillväxten förändras.
Den använder dessutom bara data som fanns tillgänglig den dagen, så historiken blir en ärlig simulering.
""", expanded=True)
    explain("Varför just 4 år?", f"""
Kryptomarknaden har rört sig i ungefär **fyraåriga cykler** (kopplat till Bitcoins "halvering" vart fjärde år).
Ett fönster på fyra år täcker därför en hel cykel med både uppgång och nedgång.

Kortare fönster följer priset för tätt: under en lång nedgång följer trendlinjen med nedåt, och priset ser
"billigt" ut fast det fortsätter att falla. Tabellen visar köpdagarna (köpläge + starkt köpläge) för olika fönster.
Valet av 4 år gjordes utifrån **träningsperioden**; validering och test visar om det håller på ny data.
""")
    with st.expander("📊 Jämförelse av trendfönster"):
        st.dataframe(window_comparison(zone_z, horizon), hide_index=True)

# ==========================================================================
# 4. ETH mot BTC
# ==========================================================================
with tabs[4]:
    st.subheader("4 · Ethereum jämfört med Bitcoin")
    st.markdown(
        "Bitcoin är kryptomarknadens största tillgång och påverkar ofta resten av marknaden. "
        "Här undersöker vi hur ETH och BTC hänger ihop – och om BTC kan hjälpa oss att hitta bättre köplägen."
    )

    st.markdown("#### Utveckling sedan startdatum (start = 100)")
    norm = view[["ETH", "BTC"]] / view[["ETH", "BTC"]].iloc[0] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=norm.index, y=norm["ETH"], name="ETH", line=dict(color=C_ETH, width=2)))
    fig.add_trace(go.Scatter(x=norm.index, y=norm["BTC"], name="BTC", line=dict(color=C_BTC, width=2)))
    st.plotly_chart(style(fig, log=True, height=400, y_title="Index (start = 100, log-skala)"))
    st.caption("Båda börjar på 100 det första visade datumet, så att man kan jämföra procentuell utveckling trots helt olika priser.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ETH/BTC – hur många BTC kostar en ETH?")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view.index, y=view["ETH_BTC"], name="ETH/BTC", line=dict(color=C_ETH, width=2)))
        fig.add_trace(go.Scatter(x=view.index, y=view["ETH_BTC_snitt_1år"], name="Snitt senaste året",
                                 line=dict(color=C_TREND, width=2, dash="dash")))
        st.plotly_chart(style(fig, height=340, y_title="BTC per ETH"))
        st.caption("Stiger linjen går ETH bättre än BTC. Under den streckade linjen är ETH billigare än vanligt *mätt i BTC*.")
    with col2:
        st.markdown("#### Hur lika rör de sig? (90 dagars korrelation)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view.index, y=view["Korrelation_90d"], name="Korrelation", line=dict(color=C_SECOND, width=2)))
        fig.add_hline(y=0, line=dict(color=C_TREND, width=1))
        fig.update_yaxes(range=[-1, 1])
        st.plotly_chart(style(fig, height=340, y_title="Korrelation (−1 till 1)"))
        st.caption("1 = rör sig exakt lika, 0 = inget samband, −1 = rör sig åt motsatt håll.")

    st.markdown("#### Är båda billiga samtidigt? Avstånd till respektive trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=view["ETH_z"], name="ETH", line=dict(color=C_ETH, width=2)))
    fig.add_trace(go.Scatter(x=view.index, y=view["BTC_z"], name="BTC", line=dict(color=C_BTC, width=2)))
    fig.add_hline(y=-zone_z, line=dict(color=km.ZONE_COLORS["Köpläge"], dash="dot"), annotation_text="köpzon", annotation_position="bottom left")
    fig.add_hline(y=0, line=dict(color=C_TREND, width=1))
    st.plotly_chart(style(fig, height=340, y_title="z-värde"))
    st.caption("När båda linjerna ligger under den gröna prickade linjen räknas det som **starkt köpläge**.")

    st.markdown(f"#### Hjälper BTC? Avkastning för ETH {horizon} senare")
    buy_days = df[df["Zon"].isin(["Köpläge", "Starkt köpläge"])].copy()
    buy_days["BTC_läge"] = np.where(buy_days["Zon"] == "Starkt köpläge",
                                    "ETH i köpzon + BTC i köpzon", "ETH i köpzon, BTC inte")
    btc_split = st.toggle("Dela upp på gruppens perioder", value=False, key="btc_split")
    st.markdown("**a) ETH är i köpzon – spelar det roll om BTC också är det?**")
    btc_order = ["ETH i köpzon + BTC i köpzon", "ETH i köpzon, BTC inte"]
    st.dataframe(outcome_table(buy_days, horizon, "BTC_läge", btc_order, by_period=btc_split), hide_index=True)
    rel_order = ["ETH ovanligt billig mot BTC", "Övriga dagar"]
    rel = df.assign(Rel=np.where(df["ETH_BTC_z"].isna(), "Ingen data",
                                 np.where(df["ETH_BTC_z"] < -1, rel_order[0], rel_order[1])))
    st.markdown("**b) Är det ett bra läge när ETH har gått ovanligt dåligt jämfört med BTC det senaste året?**")
    st.dataframe(outcome_table(rel, horizon, "Rel", rel_order, by_period=btc_split), hide_index=True)

    def compare(frame: pd.DataFrame, col: str, a: str, b: str) -> str:
        r = frame[f"Avk_{horizon}"]
        ra, rb = r[frame[col] == a].dropna(), r[frame[col] == b].dropna()
        if not len(ra) or not len(rb):
            return "för få dagar för att jämföra"
        diff = ra.median() - rb.median()
        if abs(diff) < 0.10:
            return f"ungefär samma typiska avkastning ({pct(ra.median())} mot {pct(rb.median())})"
        better = "bättre" if diff > 0 else "sämre"
        return f"{better} typisk avkastning ({pct(ra.median())} mot {pct(rb.median())})"

    btc_result = compare(buy_days, "BTC_läge", *btc_order)
    rel_result = compare(rel, "Rel", *rel_order)
    btc_takeaway = (
        "Att hela marknaden är billig samtidigt har alltså gett ett ännu bättre läge."
        if btc_result.startswith("bättre") else
        "Att hela marknaden är billig samtidigt gör alltså inte automatiskt läget bättre – köpzonen för ETH "
        "är den viktigaste delen. Starkt köpläge betyder framför allt att signalen gäller hela kryptomarknaden."
    )

    corr_all = df["ETH"].pct_change().corr(df["BTC"].pct_change())
    explain("Vad lär vi oss om ETH och BTC?", f"""
- ETH och BTC **rör sig i stort sett tillsammans**: korrelationen mellan deras dagliga förändringar är
  **{num(corr_all)}** över hela perioden och just nu **{num(today['Korrelation_90d'])}**. När BTC faller brukar ETH falla också.
- **Tabell a:** när ETH är i köpzon och BTC *också* är det gav det {btc_result} jämfört med när bara ETH var
  i köpzon. {btc_takeaway}
- **Tabell b:** dagar när ETH hade gått ovanligt dåligt mot BTC (ETH/BTC mer än en standardavvikelse under sitt
  årssnitt) gav {rel_result} jämfört med övriga dagar. Slå på uppdelningen per period för att se om det håller
  i alla perioder – ett samband som bara finns i en period är svagt.
- Korrelation betyder inte orsak: BTC *orsakar* inte nödvändigtvis ETH:s rörelser, de påverkas av samma marknad.
""", expanded=True)

# ==========================================================================
# 5. Köpläge & prognos
# ==========================================================================
with tabs[5]:
    st.subheader("5 · Köpläge och prognos")
    st.markdown("Här sätter vi ihop allt till ett svar: **hur ser läget ut idag, och vad har hänt tidigare i liknande lägen?**")

    st.markdown("#### Läge dag för dag")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=view["ETH"], name="ETH-pris", line=dict(color=C_TREND, width=1), showlegend=False))
    for zone in ["Starkt köpläge", "Köpläge", "Neutral", "Dyrt"]:
        z = view[view["Zon"] == zone]
        fig.add_trace(go.Scatter(x=z.index, y=z["ETH"], mode="markers", name=f"{ZONE_ICON[zone]} {zone}",
                                 marker=dict(color=km.ZONE_COLORS[zone], size=6 if zone == "Starkt köpläge" else 4,
                                             symbol="diamond" if zone == "Starkt köpläge" else "circle")))
    st.plotly_chart(style(fig, log=True, height=420, y_title="Pris (USD, logaritmisk skala)"))

    st.markdown("#### Metod 1: Hur gick det de dagar som liknade idag?")
    sim = km.similar_days(df, today["ETH_z"], horizon)
    if len(sim) and not pd.isna(today["ETH_z"]):
        st.markdown(
            f"Idag ligger ETH **{num(today['ETH_z'])}** standardavvikelser från trenden. "
            f"Vi letar upp alla historiska dagar som låg ungefär lika långt bort (±0,25) och kollar vad som hände {horizon} senare."
        )
        a, b, c = st.columns(3)
        a.metric("Liknande dagar", len(sim))
        b.metric("Andel gånger priset steg", pct((sim > 0).mean(), sign=False))
        c.metric("Typisk avkastning (median)", pct(sim.median()))
        fig = go.Figure(go.Histogram(x=sim * 100, nbinsx=30, marker_color=C_ETH,
                                     hovertemplate="%{x:.0f} %: %{y} dagar<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=C_TREND, width=1))
        fig.update_layout(title=f"Fördelning av avkastning {horizon} efter liknande dagar")
        st.plotly_chart(style(fig, height=300, y_title="Antal dagar").update_layout(hovermode="closest").update_xaxes(title="Avkastning (%)"))
        first, last = sim.index.min(), sim.index.max()
        st.caption(f"De liknande dagarna ligger mellan {first:%Y-%m} och {last:%Y-%m}. Många av dem ligger tätt intill varandra och "
                   "räknas därför inte som helt oberoende händelser.")
    else:
        st.info("Det finns inga jämförbara dagar med känt utfall för dagens läge.")

    st.markdown("#### Metod 2: Maskininlärning – logistisk regression")
    model = get_model(trend_years, zone_z, horizon)
    st.markdown(
        f"""
Vi tränar en **logistisk regression** – en enkel ML-modell som räknar ut en sannolikhet. Den får en enda
uppgift: *givet hur långt under eller över trenden ETH ligger idag, hur sannolikt är det att priset är högre om {horizon}?*

- **Träning:** {model['train_start']:%Y-%m-%d} – {model['train_end']:%Y-%m-%d} ({model['n_train']} dagar). Vi stannar där för att
  modellen inte ska få se något utfall från valideringsperioden (2023 och framåt).
- **Test:** alla dagar från 2023 där vi vet utfallet ({model['n_test']} dagar) – data modellen aldrig sett.
"""
    )
    a, b, c, d = st.columns(4)
    a.metric("Modellens uppskattning idag", pct(model["prob"].iloc[-1], sign=False),
             help="Är för optimistisk som exakt sannolikhet – se förklaringen under grafen.")
    b.metric("Rangordningsförmåga (AUC)", num(model["auc"]),
             help="0,5 = som att singla slant, 1,0 = perfekt. Mäter om modellen ger högre värden till dagar som faktiskt följdes av uppgång.")
    c.metric("Träffsäkerhet på testdata", pct(model["accuracy"], sign=False))
    d.metric("Att alltid gissa 'upp'", pct(model["base_rate"], sign=False),
             help="Jämförelsevärde: andelen testdagar där priset faktiskt steg.")

    fig = go.Figure()
    p = model["prob"][model["prob"].index.year >= start_year]
    fig.add_trace(go.Scatter(x=p.index, y=p * 100, name="Sannolikhet för uppgång", line=dict(color=C_ETH, width=2)))
    fig.add_vrect(x0=pd.Timestamp("2023-01-01"), x1=p.index.max(), fillcolor="rgba(128,128,128,0.12)", line_width=0,
                  annotation_text="testperiod", annotation_position="top left")
    fig.add_hline(y=50, line=dict(color=C_TREND, width=1, dash="dot"))
    st.plotly_chart(style(fig, height=320, y_title="%"))

    explain("Hur ska jag tolka ML-resultatet?", f"""
- **AUC {num(model['auc'])}** betyder ungefär: om man tar en dag som följdes av uppgång och en som följdes av
  nedgång, så gav modellen den "bra" dagen högst värde i {pct(model['auc'], sign=False)} av fallen. Modellen är alltså
  bra på att **rangordna** dagar – ju längre under trenden, desto bättre läge.
- **Men** själva procentsatsen är för optimistisk. Träningsperioden innehöll främst stigande marknader, så modellen
  tror nästan alltid på uppgång. Det syns på att träffsäkerheten ({pct(model['accuracy'], sign=False)}) inte är bättre än att
  alltid gissa "upp" ({pct(model['base_rate'], sign=False)}).
- Använd därför sannolikheten för att **jämföra** lägen (högre = historiskt bättre), inte som en exakt chans.
""", expanded=True)

    st.markdown("#### Begränsningar")
    st.markdown(
        """
- **Få cykler:** ETH har bara gått igenom ungefär tre stora cykler. Det är få "riktiga" exempel att lära sig av.
- **Överlappande perioder:** dagar som ligger nära varandra har nästan samma framtid, så antalet *oberoende*
  observationer är mycket mindre än antalet dagar.
- **Marknaden förändras:** nya aktörer (t.ex. ETF:er), regler och teknik kan göra att gamla mönster slutar fungera.
- **Inga avgifter:** analysen räknar inte med courtage, spread eller skatt.
- **Ingen rådgivning:** appen beskriver historiska samband. Den kan inte förutsäga framtiden med säkerhet.
"""
    )

# ==========================================================================
# Ordlista & metod
# ==========================================================================
with tabs[6]:
    st.subheader("📖 Ordlista")
    st.markdown(
        """
| Ord | Förklaring |
|---|---|
| **ETH / BTC** | Ethereum respektive Bitcoin – de två största kryptovalutorna. |
| **Logaritmisk skala** | En skala där varje fördubbling är lika stor. Gör procentuella rörelser jämförbara över tid. |
| **Trendlinje** | En rak linje som beskriver den långsiktiga riktningen i priset de senaste åren. |
| **Rullande** | Räknas om varje dag med de senaste N dagarnas data. |
| **Standardavvikelse** | Ett mått på hur mycket något brukar variera. |
| **z-värde** | Hur många standardavvikelser priset ligger från trenden. −1 = ovanligt lågt, +1 = ovanligt högt. |
| **Köpzon** | Dagar då ETH legat ovanligt långt under sin trend. |
| **Glidande medelvärde (SMA)** | Genomsnittspriset de senaste N dagarna. SMA 200 = 200 dagar. |
| **EMA** | Ett glidande medelvärde som väger senaste dagarna tyngre. |
| **Bull Market Support Band** | Området mellan 20-veckors SMA och 21-veckors EMA, en populär trendindikator. |
| **ETH/BTC** | Priset på en ETH uttryckt i BTC. Visar om ETH går bättre eller sämre än BTC. |
| **Korrelation** | Hur lika två saker rör sig, från −1 (motsatt) till +1 (exakt lika). |
| **Median** | Mittenvärdet. Påverkas inte av enstaka extrema utfall som ett medelvärde gör. |
| **Logistisk regression** | En enkel maskininlärningsmodell som räknar ut sannolikheten för ett ja/nej-utfall. |
| **AUC** | Mäter hur bra en modell rangordnar fall: 0,5 = slumpen, 1,0 = perfekt. |
| **Träning / validering / test** | Modellen lär sig på äldre data och prövas på nyare data som den inte har sett. |
"""
    )
    st.subheader("Metoden steg för steg")
    st.markdown(
        f"""
1. **Data:** gruppens tvättade dataset med dagliga ETH- och BTC-priser i USD från Coin Metrics
   ({df.index.min():%Y-%m-%d} – {df.index.max():%Y-%m-%d}, {len(df)} dagar, inga saknade dagar).
2. **Logaritmera priserna** så att procentuella förändringar blir jämförbara.
3. **Rullande trend:** för varje dag anpassas en rät linje (minsta kvadratmetoden) till de senaste
   {trend_years} årens log-priser. Linjens värde idag är trenden.
4. **z-värde:** (log-pris − trend) / spridningen runt trenden under samma fönster.
5. **Zoner:** fasta gränser på ±{num(zone_z, 2)}. BTC behandlas på samma sätt för att hitta *starkt köpläge*.
6. **Glidande medelvärden och ETH/BTC** beräknas som kompletterande indikatorer.
7. **Utvärdering:** för varje dag räknas ETH:s avkastning 6 och 12 månader framåt. Den används *bara* för att
   kontrollera zonerna, aldrig som input. Resultaten redovisas per gruppens perioder
   (träning 2015–2022, validering 2023–2024, test 2025–2026).
8. **ML-prognos:** logistisk regression på ETH:s z-värde, tränad före 2023 och testad på 2023 och framåt.

Ingen beräkning använder information från framtiden – varje dags signal hade kunnat räknas ut den dagen.
"""
    )
    st.caption("Koden finns i `eth_kopzon_app/kopzon_model.py` (beräkningar) och `eth_kopzon_app/app.py` (visning).")
