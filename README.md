# GB Power Forecasting Suite

> Forecasting GB intra-day electricity prices, with practical use cases built on top — starting with battery dispatch optimisation and an interactive dashboard.

The **core** of this project is a model that forecasts the Great Britain intra-day electricity price. On top of that forecast, the repo hosts **use cases** that turn the prediction into a concrete decision — the first being a **battery dispatch optimiser** with an interactive dashboard. The structure is designed to grow with new use cases over time.

```
forecasting model  →  forecasted ID prices  →  use case (e.g. battery dispatch optimiser)
```

---

# Core — Intra-Day Price Forecasting

## Objective

Each hour of the GB power market has two key prices:

- **Day-Ahead (DA)** — published the day before delivery (known in advance)
- **Intra-Day (ID)** — updated ~2 hours before delivery, as the supply/demand balance changes

The goal is to **forecast the Intra-Day price**, treating the Day-Ahead price as a known input and learning the *correction* the market applies closer to delivery.

## Data

Hourly data spanning **2020–2024** (GB grid, 17 transmission zones).

| File | Content |
|------|---------|
| `prices.csv` | Day-Ahead (`price_DA`) and Intra-Day (`price_ID`) prices |
| `INDGEN_DA / ID.csv` | Indicative generation forecast, per zone |
| `MELNGC_DA / ID.csv` | Indicative margin (available generation − demand), per zone |
| `TSDF_DA / ID.csv` | Transmission system demand forecast, per zone |

`_DA` = forecast made the day before · `_ID` = forecast updated intra-day. *Source: [Elexon BMRS](https://bmrs.elexon.co.uk/).*

## Approach

1. **Data preparation** — aligned the half-hourly prices to the hourly features, merged all sources, forward-filled missing values.
2. **EDA** — studied the **spread** (`price_ID − price_DA`): distribution, daily/seasonal patterns, link with margin, and the two market regimes (calm years vs the 2021–22 energy crisis).
3. **Feature engineering (13 features)** — `price_DA` (the anchor), **deltas** (`ID − DA`) for generation/margin/demand (the intra-day *surprise*), raw intra-day indicators, and calendar features.
4. **Modelling** — **XGBoost**, chosen for the non-linear, outlier-heavy relationships seen in the EDA, benchmarked against a **naive** baseline (`ID = DA`) and a **linear regression**.
5. **Validation** — chronological split: train on 2020–2023, test on 2024.

## Results

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Naive (ID = DA) | 10.16 | 15.94 | 0.737 |
| Linear Regression | 11.01 | 16.41 | 0.722 |
| XGBoost v1 (untuned) | 11.09 | 17.12 | 0.697 |
| **XGBoost v2 (regularised)** | **10.33** | **15.90** | **0.739** |

The regularised **XGBoost v2** is the best model. Feature importance confirms the economic intuition: the model **anchors on the day-ahead price** and **corrects it with the supply/demand surprise**. It captures the direction of moves well but smooths the extreme spikes, which are often driven by events outside the dataset (plant outages, balancing actions).

> 📓 Notebook: `Forecast-GB-Power-ID.ipynb` — it exports the 2024 forecasts to `dataset/forecast_ID_2024.csv`, used by the use cases below.

---

# Use case — Battery (BESS) Dispatch Optimisation

## Idea

A **Battery Energy Storage System (BESS)** makes money through **time arbitrage**: charge (buy) when electricity is cheap, discharge (sell) when it's expensive. Using the forecasted ID prices as input, we find the optimal hourly charge/discharge schedule that **maximises daily revenue** under the battery's physical constraints.

Specs are inspired by **Engie's Cathkin BESS** (East Kilbride, Scotland); the optimiser is spec-agnostic and works for any capacity/power.

## Method

The problem is framed as a **linear program** and solved with **PuLP** (CBC solver):

- **Decision variables** — `charge`, `discharge`, and `soc` (state of charge) for each hour
- **Objective** — maximise `Σ price[t] × (discharge[t] − charge[t])`
- **Constraints** — SOC evolution with round-trip efficiency, capacity and power limits, and a return to the starting SOC at the end of the day

The optimised strategy is benchmarked against a **naive rule** (charge during the cheapest hours, discharge during the most expensive) to quantify the value the optimisation adds.

> 📓 Notebook: `BESS-Dispatch-Optimisation.ipynb`

## Interactive dashboard

An interactive **Dash** app lets you explore the optimiser live: pick any 2024 day, tune the battery parameters (capacity, power, efficiency, initial SOC) or use presets, and watch the dispatch schedule, state of charge, and KPIs (revenue, average buy/sell price, cycles) update in real time.

```bash
cd bess-dashboard
python app.py
# then open http://localhost:7860
```

---

## Repository

```
.
├── Forecast-GB-Power-ID.ipynb         # Core — forecasting notebook
├── BESS-Dispatch-Optimisation.ipynb   # Use case — battery dispatch notebook
├── bess-dashboard/                    # Interactive dashboard (Dash)
│   ├── app.py
│   └── assets/style.css
├── dataset/                           # Input CSV files + forecast_ID_2024.csv
└── README.md
```

### Run

```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn pulp dash dash-bootstrap-components plotly
```

1. Run `Forecast-GB-Power-ID.ipynb` (produces `dataset/forecast_ID_2024.csv`)
2. Run `BESS-Dispatch-Optimisation.ipynb`, or launch the dashboard (`cd bess-dashboard && python app.py`)

---

> *AI was used to generate the plots, build the dashboard, and polish the markdown. The analysis, modelling and interpretation are my own.*
