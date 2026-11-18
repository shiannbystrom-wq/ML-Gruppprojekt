# ETH köpzoner – tidsserie, glidande medelvärden och ETH/BTC

**🔗 Live-app:** [eth-kopzoner.streamlit.app](https://eth-kopzoner.streamlit.app/)

En Streamlit-app som visar Ethereums prishistorik, pekar ut historiska köpzoner med en rullande
trendmodell och undersöker hur ETH hänger ihop med Bitcoin. Alla vyer förklaras så att de går att
förstå utan teknisk bakgrund.

Bygger på gruppens tvättade dataset `data/processed/crypto_dataset.csv`
(samma data som i `2_ethereum_accumulation_zones`-notebooken).

## Kör appen lokalt

Från projektets rotmapp, med projektets `.venv` aktiverad:

```powershell
pip install -r eth_kopzon_app/requirements.txt
streamlit run eth_kopzon_app/app.py
```

## Filer

| Fil | Innehåll |
|---|---|
| `kopzon_model.py` | All beräkning: data, rullande trend, z-värde, zoner, glidande medelvärden, ETH/BTC, utvärdering och logistisk regression. Kan importeras i en notebook. |
| `app.py` | Streamlit-appen. Visar resultaten och förklarar dem. |
| `requirements.txt` | Paket som appen behöver (plotly är det enda som inte redan finns i projektets requirements). |

## Modellen i korthet

1. **Rullande log-linjär trend.** För varje dag anpassas en rät linje till log(pris) över de senaste
   4 åren (minsta kvadrat). Bara data fram till och med dagen används.
2. **z-värde** = (log-pris − trend) / standardavvikelsen för residualerna i samma fönster.
3. **Zoner** med fasta gränser:
   - *Starkt köpläge*: ETH och BTC båda under −1
   - *Köpläge*: ETH under −1
   - *Neutral*: −1 till +1
   - *Dyrt*: ETH över +1
4. **Kompletterande indikatorer:** SMA 50/200, Bull Market Support Band (20W SMA + 21W EMA),
   ETH/BTC-kvot jämfört med sitt 1-årssnitt, 90 dagars rullande korrelation.
5. **Utvärdering:** ETH:s avkastning 6 och 12 månader framåt per zon, uppdelat på gruppens perioder
   (träning 2015–2022, validering 2023–2024, test 2025–2026). Framtida avkastning används aldrig som input.
6. **ML-prognos:** logistisk regression på ETH:s z-värde → sannolikhet att priset är högre efter
   6/12 månader. Tränas på dagar vars utfall var känt före 2023, testas på 2023 och framåt.

### Varför 4 år?

Ungefär en hel kryptocykel. Fönstret valdes utifrån träningsperioden, där kortare fönster (1–3 år)
gav köpsignaler mitt i långa nedgångar. Appen visar jämförelsen under flik 3.

## Begränsningar

- Få oberoende marknadscykler och kraftigt överlappande avkastningsperioder.
- Trendmodellen behöver 4 års historik, så zoner finns först från augusti 2019.
- Testperioden innehåller få köpdagar.
- ML-modellens sannolikheter är för optimistiska (träningsperioden var mest stigande); den rangordnar
  dagar bra (hög AUC) men slår inte "gissa alltid upp" i träffsäkerhet.
- Inga transaktionskostnader. Historisk analys – inte investeringsrådgivning.
