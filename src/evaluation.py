"""
evaluation.py
Forecast evaluation metrics and the Diebold-Mariano test.

Metrics:
    - RMSE  : Root Mean Squared Error
    - MAE   : Mean Absolute Error
    - QLIKE : Quasi-likelihood loss (Patton 2011) — standard in RV literature
    - OOS R²: Out-of-sample R² (Welch & Goyal 2008)

Statistical test:
    - Diebold-Mariano (1995): tests whether forecast accuracy difference is significant
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Point forecast metrics
# ---------------------------------------------------------------------------

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(actual - predicted))


def qlike(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    QLIKE loss (Patton 2011).
    Standard loss function for realized volatility evaluation.
    QLIKE = mean(RV / RV_hat - ln(RV / RV_hat) - 1)

    Implementation notes:
    - Both series are floored at their joint 1st percentile (minimum meaningful RV).
      This is more principled than a fixed epsilon and handles varying data scales.
    - Annualized daily RV for SPY is rarely below ~0.001 in practice.
    - The floor prevents division-by-zero and log(0) when a model predicts
      near-zero volatility on a quiet day.
    """
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    # Floor: use the 1st percentile of actual RV as the minimum sensible value
    floor = max(np.percentile(actual, 1), 1e-4)
    actual    = np.maximum(actual,    floor)
    predicted = np.maximum(predicted, floor)

    ratio = actual / predicted
    return np.mean(ratio - np.log(ratio) - 1)


def oos_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Out-of-sample R² (Welch & Goyal 2008).
    Benchmarks model against the historical mean forecast.
    Negative values mean the model is worse than just predicting the mean.

    OOS-R² = 1 - MSE_model / MSE_benchmark
    where benchmark is the mean of actual values.
    """
    benchmark = np.full_like(actual, np.mean(actual))
    mse_model = np.mean((actual - predicted) ** 2)
    mse_benchmark = np.mean((actual - benchmark) ** 2)
    return 1 - mse_model / mse_benchmark


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------

def diebold_mariano(
    actual: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    loss: str = "mse",
    h: int = 1,
) -> dict:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    H0: The two forecasts have equal predictive accuracy.
    H1: The forecasts have different predictive accuracy.

    Parameters
    ----------
    actual : np.ndarray  — true values
    pred1  : np.ndarray  — forecast from model 1
    pred2  : np.ndarray  — forecast from model 2
    loss   : str         — 'mse' or 'mae'
    h      : int         — forecast horizon (1 for 1-step-ahead)

    Returns
    -------
    dict with keys: dm_stat, p_value, conclusion
    Negative DM stat: model 1 is more accurate.
    Positive DM stat: model 2 is more accurate.
    """
    if loss == "mse":
        e1 = (actual - pred1) ** 2
        e2 = (actual - pred2) ** 2
    elif loss == "mae":
        e1 = np.abs(actual - pred1)
        e2 = np.abs(actual - pred2)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    d = e1 - e2
    n = len(d)
    d_bar = np.mean(d)

    # Newey-West variance estimator to account for autocorrelation up to lag h-1
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k], ddof=1)[0, 1]
        gamma_sum += 2 * gamma_k

    var_d = (gamma_0 + gamma_sum) / n
    dm_stat = d_bar / np.sqrt(var_d)

    # Two-sided p-value using t-distribution with n-1 degrees of freedom
    p_value = 2 * stats.t.sf(np.abs(dm_stat), df=n - 1)

    conclusion = (
        "Model 1 significantly more accurate"
        if dm_stat < 0 and p_value < 0.05
        else "Model 2 significantly more accurate"
        if dm_stat > 0 and p_value < 0.05
        else "No significant difference"
    )

    return {"dm_stat": round(dm_stat, 4), "p_value": round(p_value, 4), "conclusion": conclusion}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def evaluate_all(results: dict, actual: np.ndarray) -> pd.DataFrame:
    """
    Compute all metrics for multiple models and return a summary DataFrame.

    Parameters
    ----------
    results : dict — {model_name: predictions_array}
    actual  : np.ndarray — true RV values

    Returns
    -------
    pd.DataFrame with rows = models, columns = metrics
    """
    rows = []
    for name, preds in results.items():
        rows.append({
            "Model": name,
            "RMSE":   round(rmse(actual, preds), 6),
            "MAE":    round(mae(actual, preds), 6),
            "QLIKE":  round(qlike(actual, preds), 6),
            "OOS R²": round(oos_r2(actual, preds), 4),
        })
    return pd.DataFrame(rows).set_index("Model")


def run_dm_tests(results: dict, actual: np.ndarray) -> pd.DataFrame:
    """
    Run all pairwise Diebold-Mariano tests and return a summary DataFrame.

    Parameters
    ----------
    results : dict — {model_name: predictions_array}
    actual  : np.ndarray

    Returns
    -------
    pd.DataFrame with one row per model pair
    """
    model_names = list(results.keys())
    rows = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name1, name2 = model_names[i], model_names[j]
            dm = diebold_mariano(actual, results[name1], results[name2])
            rows.append({
                "Model 1": name1,
                "Model 2": name2,
                "DM Stat": dm["dm_stat"],
                "p-value": dm["p_value"],
                "Conclusion": dm["conclusion"],
            })
    return pd.DataFrame(rows)
