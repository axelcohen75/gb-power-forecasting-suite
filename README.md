# GB Power: Intra-Day Price Forecasting & Battery Dispatch

An end-to-end project on the Great Britain power market, in two parts:

1. **Forecasting** the Intra-Day (ID) electricity price from public market data.
2. **Optimising** a battery's charge/discharge schedule using those forecasts to maximise revenue.

The two parts form a realistic pipeline — the forecast is not the end goal in itself, it's the **input to a decision**:

```
XGBoost forecast  →  forecasted ID prices  →  PuLP optimiser  →  battery dispatch strategy
```

---

# Part 1 — Intra-Day Price Forecasting

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

The regularised **XGBoost v2** is the best model. Feature importance confirms the economic intuition: the model **anchors on the day-ahead price** and **corrects it with the supply/demand surprise**. The model captures the direction of moves well but smooths the extreme spikes, which are often driven by events outside the dataset (plant outages, balancing actions).

---

# Part 2 — Battery (BESS) Dispatch Optimisation

## Objective

A **Battery Energy Storage System (BESS)** makes money through **time arbitrage**: charge (buy) when electricity is cheap, discharge (sell) when it's expensive. Using the forecasted ID prices as input, we find the optimal hourly charge/discharge schedule that **maximises daily revenue** under the battery's physical constraints.

Specs are inspired by **Engie's Cathkin BESS** (East Kilbride, Scotland), a 2-hour lithium iron phosphate site; the optimiser is spec-agnostic and works for any capacity/power.

## Method

The problem is framed as a **linear program** and solved with **PuLP**:

- **Decision variables** — `charge`, `discharge`, and `soc` (state of charge) for each hour
- **Objective** — maximise `Σ price[t] × (discharge[t] − charge[t])`
- **Constraints**:
  - SOC evolution with round-trip efficiency applied on charge
  - Capacity and power limits
  - Battery ends the day at its starting SOC (fair, repeatable cycle)

The optimised strategy is benchmarked against a **naive rule** (charge during the cheapest hours, discharge during the most expensive) to quantify the value the optimisation adds.

## Output

- Total daily revenue (£) and number of arbitrage cycles
- Hour-by-hour schedule (price vs charge/discharge vs SOC)
- On a volatile example day, the optimiser runs two arbitrage cycles — buying at the overnight low and selling into the evening peak — while respecting all physical limits.

---

## Repository

```
.
├── Forecast-GB-Power-ID.ipynb         # Part 1 — forecasting notebook
├── BESS-Dispatch-Optimisation.ipynb   # Part 2 — battery dispatch notebook
├── BESS Dispatch Dashboard/           # Interactive dashboard
├── dataset/                           # Input CSV files + forecast_ID_2024.csv (output of Part 1, input of Part 2)
└── README.md
```

### Run

```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn pulp
jupyter notebook
```

Run `Forecast-GB-Power-ID.ipynb` first (it produces `forecast_ID_2024.csv`), then `BESS-Dispatch-Optimisation.ipynb`.

---

> *AI was used to generate the plots and polish the markdown. The analysis, modelling and interpretation are my own.*
