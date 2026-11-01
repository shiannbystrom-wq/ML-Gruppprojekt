# Projektguide – Ethereum Accumulation Zones

## 1. Om projektet

**Projekttitel:** Ethereum Accumulation Zones: Regression-Based Price Bands and Historical Return Analysis

**Forskningsfråga:**

*Can regression-based price bands identify historical Ethereum price zones associated with more favorable long-term returns?*

Syftet med projektet är att undersöka om regressionsbaserade prisband kan identifiera historiska prisnivåer för Ethereum (ETH) som har varit förknippade med mer gynnsam långsiktig avkastning.

Vi ska jämföra olika statistiska och maskininlärningsbaserade metoder. Alla experiment ska utgå från samma forskningsfråga och använda en gemensam datamängd och utvärderingsmetod.

Projektet är ett historiskt dataanalysprojekt och ska inte betraktas som investeringsrådgivning.

## 2. Projektstruktur

Projektet är organiserat enligt följande:

```text
ML-Gruppprojekt/
│
├── app/                    # Applikation
├── data/
│   ├── raw/                # Ursprunglig data från API
│   └── processed/          # Rensad data och experimentdata
│       └── splits/         # Train-, validation- och testdata
│
├── notebooks/              # Jupyter Notebooks för analys
│
├── reports/
│   ├── figures/            # Diagram och visualiseringar
│   └── TEAM_GUIDE.md       # Den här guiden
│
├── src/                    # Python-skript för datapipeline
│
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

## 3. Kom igång med projektet

### Steg 1 – Klona GitHub-repositoriet

Öppna PowerShell och kör:

```powershell
git clone https://github.com/shiannbystrom-wq/ML-Gruppprojekt.git
cd ML-Gruppprojekt
```

### Steg 2 – Skapa en virtuell miljö

```powershell
python -m venv .venv
```

Aktivera miljön:

```powershell
.venv\Scripts\Activate.ps1
```

Om PowerShell blockerar aktiveringen kan följande kommando användas för den aktuella sessionen:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Aktivera därefter miljön igen.

### Steg 3 – Installera beroenden

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Kontrollera att Python kan importera projektets viktigaste bibliotek:

```powershell
python -c "import pandas, numpy, sklearn, statsmodels, requests, streamlit; print('Dependencies OK')"
```

## 4. Data och datakälla

Projektet använder historiska prisdata för Ethereum (ETH) och Bitcoin (BTC).

Datakälla: **Coin Metrics Community API**

API-dokumentation: https://docs.coinmetrics.io/api/v4/

Vi använder dagliga USD-priser (`PriceUSD`).

Rådata hämtas genom:

```powershell
python src/download_api_data.py
```

Skriptet hämtar data och sparar filerna i:

```text
data/raw/eth_api.csv
data/raw/btc_api.csv
```

Rådata laddas inte upp till GitHub eftersom de kan hämtas på nytt från API:et.

## 5. Datapipeline

Kör följande skript i angiven ordning när rådata behöver hämtas och hela datasetet byggas om.

### Steg 1 – Hämta data

```powershell
python src/download_api_data.py
```

### Steg 2 – Validera API-data

```powershell
python src/validate_api_data.py
```

### Steg 3 – Rensa data

```powershell
python src/data_cleaning.py
```

### Steg 4 – Skapa features

```powershell
python src/feature_engineering.py
```

### Steg 5 – Skapa EDA-visualiseringar

```powershell
python src/eda.py
```

### Steg 6 – Skapa framtida avkastningsmål

```powershell
python src/create_targets.py
```

Vi undersöker framför allt framtida avkastning över 6 respektive 12 månader.

### Steg 7 – Skapa gemensamt experimentdataset

```powershell
python src/prepare_experiment_data.py
```

### Steg 8 – Skapa tidsbaserade dataset

```powershell
python src/create_time_splits.py
```

De gemensamma experimentdataseten sparas under:

```text
data/processed/splits/
```

## 6. Gemensamma experimentregler

För att resultaten ska kunna jämföras måste alla gruppmedlemmar följa samma grundläggande regler.

* Använd samma gemensamma datakälla.
* Använd samma definition av framtida avkastning.
* Använd samma tidsbaserade train-, validation- och testperioder.
* Använd inte framtida information som modellinput.
* Anpassa eventuell skalning och preprocessing på träningsdata, inte på hela datasetet.
* Testdata ska inte användas för att välja modell eller justera parametrar.
* Dokumentera modellens antaganden, begränsningar och resultat.

Vi ska undersöka både 6- och 12-månadersperspektivet.

Eftersom framtida avkastningsperioder överlappar varandra måste detta beaktas vid modellutvärdering och statistiska tester.

## 7. Möjliga experimentområden

Gruppen kan undersöka olika metoder, exempelvis:

1. **Logarithmic Regression** – undersöker det långsiktiga logaritmiska förhållandet mellan tid och ETH-pris.
2. **Quantile Regression** – undersöker olika kvantiler och kan användas för att konstruera prisband.
3. **Multiple Regression with BTC** – undersöker om information från Bitcoin kan bidra till att förklara ETH:s prisutveckling.
4. **Rolling Regression / Time Series Analysis** – undersöker hur sambanden förändras över tid.

Det slutliga metodvalet bestäms gemensamt av gruppen.

Alla metoder ska bidra till att besvara samma forskningsfråga.

## 8. Git och samarbete

Vi använder GitHub för versionshantering och samarbete.

Repository:

https://github.com/shiannbystrom-wq/ML-Gruppprojekt

### Arbetsflöde

Vi arbetar med separata feature branches.

Exempel:

```powershell
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

När du har gjort ändringar:

```powershell
git status
git add .
git commit -m "Describe your changes"
git push -u origin feature/your-feature-name
```

Skapa sedan en Pull Request på GitHub.

Pull Request ska granskas innan ändringarna slås ihop med `main`.

### Viktiga regler

* Arbeta inte direkt i `main`.
* Hämta de senaste ändringarna från `main` innan du börjar ett nytt arbete.
* Använd tydliga branch- och commitnamn.
* Ladda inte upp `.venv`, hemliga nycklar eller rådatafiler.
* Kontrollera alltid `git status` innan du gör en commit.
* Kommunicera med gruppen om ändringar i gemensamma skript och dataset.

## 9. Datahantering

Följande ska normalt versionshanteras:

* Python-skript i `src/`
* Jupyter Notebooks i `notebooks/`
* Dokumentation
* Diagram i `reports/figures/`
* Gemensamma bearbetade dataset i `data/processed/`

Följande ska inte versionshanteras:

* Virtuella Python-miljöer (`.venv/`)
* Rådata som kan hämtas från API:et (`data/raw/*.csv`)
* Lokala miljövariabler och hemliga nycklar

## 10. Dokumentation och rapportering

Varje gruppmedlem ansvarar för att dokumentera:

* Vilken metod som används.
* Varför metoden är relevant för forskningsfrågan.
* Vilka variabler och features som används.
* Hur modellen tränas och utvärderas.
* Vilka resultat som erhålls.
* Vilka begränsningar metoden har.

Resultat ska presenteras tydligt med relevanta diagram, tabeller och statistiska mått.

Vi ska inte enbart jämföra medelvärden. Median, fördelningar, andelen positiva framtida avkastningar och riskmått kan också vara relevanta.

## 11. Viktiga metodprinciper

För att undvika missvisande slutsatser gäller följande:

* Korrelation innebär inte kausalitet.
* Historiska samband garanterar inte framtida resultat.
* Prisband ska definieras tydligt och konsekvent.
* Framtida avkastning får endast användas som utvärderingsmål, inte som inputvariabel.
* En historisk simulering får inte använda information som inte hade varit tillgänglig vid den aktuella tidpunkten.
* Eventuella backtester ska ta hänsyn till transaktionskostnader och slippage.
* Begränsningar och osäkerheter ska redovisas öppet.

## 12. Projektets nästa steg

Gruppen ska gemensamt:

1. Bekräfta forskningsfrågan och den slutliga metodplanen.
2. Fördela experimentområden mellan gruppmedlemmarna.
3. Säkerställa att alla använder samma dataset och utvärderingsregler.
4. Genomföra och dokumentera experimenten.
5. Jämföra resultaten och diskutera begränsningarna.
6. Sammanställa slutsatserna i rapporten och applikationen.
