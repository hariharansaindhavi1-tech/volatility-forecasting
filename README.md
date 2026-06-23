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
| XGB_HAR | 0.0283 | 0.0182 | 1.411 | -0.107 |
| XGB_FULL | 0.0274 | 0.0170 | 1.672 | -0.040 |

### Walk-Forward Evaluation (Expanding Window, Step=21)

| Model | RMSE | MAE | QLIKE | OOS R² |
|---|---|---|---|---|
| HAR-RV | 0.0509 | 0.0272 | **1.114** | **+0.005** |
| XGB_HAR | 0.0521 | 0.0296 | 14.98 | -0.043 |
| XGB_FULL | 0.0527 | 0.0281 | 14.78 | -0.069 |

### Diebold-Mariano Tests

All pairwise DM tests show **no statistically significant difference** between models (p > 0.05) under both evaluation frameworks. The accuracy differences are real but not large enough to reject the null of equal predictive accuracy given the sample size.

---

## Key Findings

1. **HAR-RV is the best model by QLIKE and OOS R²** in the walk-forward evaluation, which is the more rigorous test. It achieves a positive OOS R² (0.005), meaning it beats the naive mean forecast — while both XGBoost variants are negative.

2. **XGBoost does not beat HAR-RV** despite having more flexibility and an additional feature (VIX). This is consistent with the established literature: the volatility clustering structure captured by HAR-RV's linear lags is the dominant signal, and XGBoost overfits to noise in the training data rather than learning a more general pattern.

3. **VIX adds marginal value in the static split** (XGB_FULL has the best RMSE and MAE in the static test) but hurts in the walk-forward evaluation. This suggests the VIX relationship is non-stationary and the model learns a regime-specific association that doesn't generalize across time.

4. **XGBoost systematically underestimates volatility on quiet days** (QLIKE of ~15 vs HAR-RV's ~1.1 in rolling eval). QLIKE heavily penalizes underestimation of large RV events, which is exactly what matters in risk management. This is XGBoost's key failure mode here.

5. **No statistically significant winner** — the Diebold-Mariano tests cannot reject equal predictive accuracy. The sample is ~95 walk-forward predictions, which limits statistical power. A longer evaluation window or higher-frequency data would sharpen these tests.

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

## References

- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Welch, I., & Goyal, A. (2008). A Comprehensive Look at the Empirical Performance of Equity Premium Prediction. *Review of Financial Studies*, 21(4), 1455–1508.
