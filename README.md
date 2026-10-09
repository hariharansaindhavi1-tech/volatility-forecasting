# Volatility Forecasting: HAR-RV vs XGBoost

Comparing econometric and machine learning approaches to forecasting realized volatility on the S&P 500 (SPY) using 10 years of daily data (2015–2024).

---

## Research Question

Can XGBoost, augmented with implied volatility (VIX), outperform the HAR-RV model — the standard econometric benchmark — in forecasting next-day realized volatility?

---

## Models

| Model | Features | Method |
|---|---|---|
| **HAR-RV** | RV daily, weekly (5d avg), monthly (22d avg) lags | OLS regression |
| **XGB_HAR** | Same as HAR-RV | XGBoost (apples-to-apples comparison) |
| **XGB_FULL** | HAR features + VIX, VIX change, absolute return | XGBoost with implied volatility |

HAR-RV (Corsi 2009) is the industry benchmark. It captures the multi-scale clustering of volatility across daily, weekly, and monthly trader horizons using a simple 3-parameter OLS regression. XGB_HAR tests whether nonlinear ML can beat a linear baseline on the same features. XGB_FULL tests whether adding market-implied expectations (VIX) provides additional predictive signal.

---

## Data

- **SPY**: Daily OHLCV prices, 2015-01-01 to 2025-01-01 (~2,516 trading days)
- **VIX**: Daily close, same date range
- **Source**: Yahoo Finance via `yfinance`
- **Realized Volatility proxy**: Annualized squared daily log return → `RV_t = (ln(P_t / P_{t-1}))² × 252`

---

## Methodology

### Feature Engineering

```
log_return_t = ln(Close_t / Close_{t-1})
RV_t         = log_return_t² × 252        ← annualized

rv_d  = RV_t                              ← daily component
rv_w  = mean(RV_{t-4} ... RV_t)          ← 5-day average (weekly)
rv_m  = mean(RV_{t-21} ... RV_t)         ← 22-day average (monthly)

target = RV_{t+1}                         ← next-day RV (what we predict)
```

### Evaluation

Two evaluation frameworks:

- **Static split**: Train on first 80% (2015–2022), test on last 20% (2023–2024)
- **Walk-forward (expanding window)**: Re-train on all available data at each step, predict one step ahead. Step size = 21 trading days. Minimum initial window = 500 days. This simulates real deployment and avoids lookahead bias.

### Metrics

| Metric | Why |
|---|---|
| RMSE | Standard point forecast error; penalizes large misses |
| MAE | Robust to outliers; average absolute error |
| QLIKE | Standard volatility loss (Patton 2011); penalizes under-prediction of large spikes |
| OOS R² | Out-of-sample explanatory power vs. naive mean forecast |
| Diebold-Mariano | Statistical significance of accuracy differences between model pairs |

---

## Results

### Static Evaluation (80/20 Split)

| Model | RMSE | MAE | QLIKE | OOS R² |
|---|---|---|---|---|
| HAR-RV | 0.0280 | 0.0196 | 1.407 | -0.089 |
| XGB_HAR | 0.0275 | 0.0187 | 1.379 | -0.051 |
| XGB_FULL | 0.0267 | 0.0172 | 1.346 | **+0.010** |

### Walk-Forward Evaluation (Expanding Window, Step=21)

| Model | RMSE | MAE | QLIKE | OOS R² |
|---|---|---|---|---|
| HAR-RV | 0.0509 | 0.0272 | 1.114 | +0.005 |
| XGB_HAR | 0.0505 | 0.0262 | 1.155 | +0.020 |
| XGB_FULL | 0.0479 | 0.0263 | **1.047** | **+0.116** |

### Diebold-Mariano Tests

| Comparison | Static Split (p-value) | Walk-Forward (p-value) |
|---|---|---|
| HAR-RV vs XGB_HAR | 0.072 (not significant) | 0.966 (not significant) |
| HAR-RV vs XGB_FULL | **0.0004** (XGB_FULL significantly more accurate) | 0.356 (not significant) |
| XGB_HAR vs XGB_FULL | 0.022 (XGB_FULL significantly more accurate) | 0.740 (not significant) |

In the static split, XGB_FULL's edge over HAR-RV is statistically significant at the 1% level. In the walk-forward test — the more realistic, lookahead-bias-free evaluation — that same edge persists in magnitude but is no longer statistically significant.

---

## Key Findings

1. **XGB_FULL has the best point-forecast accuracy in both evaluations.** It posts the lowest RMSE, QLIKE, and highest OOS R² in both the static split (R² = +0.010) and the walk-forward backtest (R² = +0.116), where HAR-RV is barely positive (R² = +0.005).

2. **HAR-RV is hard to beat statistically.** Despite XGB_FULL's numerical lead, the Diebold-Mariano test only confirms a significant difference in the static 80/20 split (p = 0.0004). Under walk-forward evaluation, the same comparison is no longer significant (p = 0.356) — the apparent edge shrinks once the model is forced to re-train sequentially on expanding windows instead of a single fixed split.

3. **The static vs. walk-forward comparison is the reason this project runs two backtests.** A single train/test split overstates confidence in XGBoost's advantage; the walk-forward design, which better simulates real deployment and avoids lookahead bias, is more conservative and shows the gap is not yet statistically reliable.

4. **VIX is the main source of XGBoost's edge.** XGB_FULL (HAR features + VIX) consistently outperforms XGB_HAR (HAR features only) in both backtests, suggesting implied volatility carries real incremental signal over the HAR lags alone — though per finding #2, that signal isn't strong enough yet to statistically dominate HAR-RV.

5. **Sample size limits statistical power.** The walk-forward evaluation produces roughly 95 predictions (step=21 over ~2,500 trading days). A longer history, higher-frequency re-fitting (step=1), or higher-frequency intraday data would sharpen these significance tests.

---

## Project Structure

```
volatility-forecasting/
├── data/
│   └── raw/                  # SPY and VIX CSVs from yfinance
├── results/                  # All output files
│   ├── static_metrics.csv
│   ├── static_dm_tests.csv
│   ├── static_forecasts.png
│   ├── rolling_metrics.csv
│   ├── rolling_dm_tests.csv
│   ├── rolling_predictions.csv
│   ├── rolling_forecasts.png
│   ├── rolling_rmse_over_time.png
│   ├── rolling_scatter.png
│   └── feature_importance.png
├── src/
│   ├── data_loader.py        # yfinance data fetching
│   ├── features.py           # RV computation, HAR lags, feature matrix
│   ├── rolling_eval.py       # Walk-forward evaluation engine
│   ├── evaluation.py         # RMSE, MAE, QLIKE, OOS R², Diebold-Mariano
│   ├── main.py               # Full pipeline entry point
│   └── models/
│       ├── har_rv.py         # HAR-RV (OLS via statsmodels)
│       └── xgboost_model.py  # XGBoost v1 and v2
├── requirements.txt
└── README.md
```

---

## Setup and Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/main.py
```

Results are saved automatically to `results/`. The pipeline runs in under 2 minutes on a standard laptop.

To enable XGBoost hyperparameter tuning (adds ~5–10 minutes):
```python
# In src/main.py, set:
TUNE_XGB = True
```

---


