# Forecasting Intra-Day Prices in the GB Power Market

Forecasting the **Intra-Day (ID)** electricity price in the Great Britain power market from publicly available market data, using the **Day-Ahead (DA)** price and supply/demand forecasts.


## Objective

In the GB power market, each hour has two key prices:

- **Day-Ahead (DA)** — published the day before delivery (known in advance)
- **Intra-Day (ID)** — updated ~2 hours before delivery, as the supply/demand balance changes

The goal is to **forecast the Intra-Day price**, treating the Day-Ahead price as a known input and learning the *correction* the market applies closer to delivery.


## Data

All data is hourly and spans **2020–2024** (GB grid, 17 transmission zones).

| File | Content |
|------|---------|
| `prices.csv` | Day-Ahead (`price_DA`) and Intra-Day (`price_ID`) prices |
| `INDGEN_DA / ID.csv` | Indicative generation forecast, per zone |
| `MELNGC_DA / ID.csv` | Indicative margin (available generation − demand), per zone |
| `TSDF_DA / ID.csv` | Transmission system demand forecast, per zone |

`_DA` = forecast made the day before · `_ID` = forecast updated intra-day.

*Source: [Elexon BMRS](https://bmrs.elexon.co.uk/).*


## Approach

1. **Data preparation** — aligned the half-hourly prices to the hourly features, merged all sources, handled missing values (forward-fill).
2. **EDA** — studied the **spread** (`price_ID − price_DA`): its distribution, daily and seasonal patterns, the link with margin, and the two market regimes (calm years vs the 2021–22 energy crisis).
3. **Feature engineering (13 features)**:
   - `price_DA` — the anchor
   - **Deltas** (`ID − DA`) for generation, margin and demand — the intra-day *surprise*
   - Raw intra-day indicators (national totals)
   - Calendar features (hour, month, season, weekend, peak-hour)
4. **Modelling** — **XGBoost**, chosen for its ability to capture the non-linear, outlier-heavy relationships seen in the EDA, benchmarked against:
   - a **naive** baseline (`ID = DA`)
   - a **linear regression**
5. **Validation** — chronological split: train on 2020–2023, test on 2024.


## Results

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Naive (ID = DA) | 10.16 | 15.94 | 0.737 |
| Linear Regression | 11.01 | 16.41 | 0.722 |
| XGBoost v1 (untuned) | 11.09 | 17.12 | 0.697 |
| **XGBoost v2 (regularised)** | **10.33** | **15.90** | **0.739** |

The regularised **XGBoost v2** is the best model — lowest RMSE and highest R², just above the naive baseline. Feature importance confirms the economic intuition: the model **anchors on the day-ahead price** and **corrects it with the supply/demand surprise**.

### Key findings

- The DA price explains ~90% of the ID price — the model's value lies in the **volatile periods**.
- The model captures the **direction** of moves well but **smooths the extreme spikes**, which are often driven by events outside the dataset (plant outages, balancing actions).
- 2024 was a calm year, which limits the gain over the naive baseline.

### Limitations & next steps

- Predict the **spread directly** (`ID − DA`) instead of the price.
- Add richer data (wind/solar forecasts, fuel mix, gas prices).
- Test **deep-learning models (LSTM)**, which tend to lead on intra-day forecasting.
- Move from a point forecast to a **prediction interval** for trading use.
