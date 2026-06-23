"""
rolling_eval.py
Expanding window (walk-forward) out-of-sample evaluation.

Strategy:
- Start with an initial training window (min_train_size days)
- Each step: train on all data up to t, predict t+1, then expand window by 1
- This simulates real-world usage — the model only ever sees past data
- Produces a time series of errors, not just a single aggregate metric

For XGBoost: hyperparameter tuning is run once on the first window via
TimeSeriesSplit CV, then the best params are reused for all subsequent windows.
This avoids re-tuning thousands of times while still being principled.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm

from models.har_rv import HARModel
from models.xgboost_model import XGBoostVolModel
from evaluation import rmse, mae, qlike, oos_r2


HAR_COLS  = ["rv_d", "rv_w", "rv_m"]
FULL_COLS = ["rv_d", "rv_w", "rv_m", "vix", "vix_chg", "ret_abs"]


def _fit_har(X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """Fit HAR-RV via OLS and return prediction for the next row (X_test)."""
    X = sm.add_constant(X_train[HAR_COLS], has_constant="add")
    result = sm.OLS(y_train, X).fit()
    return result


def rolling_evaluation(
    features: pd.DataFrame,
    min_train_size: int = 500,
    tune_xgb: bool = False,
    step: int = 1,
) -> pd.DataFrame:
    """
    Expanding window walk-forward evaluation.

    Parameters
    ----------
    features      : full feature DataFrame (output of build_full_features, dropna applied)
    min_train_size: minimum number of days for the first training window
    tune_xgb      : if True, tune XGBoost hyperparams on the first window
    step          : number of days to advance per iteration (1 = daily re-fit)
                    Use step > 1 (e.g., 5 or 21) for faster approximate evaluation

    Returns
    -------
    pd.DataFrame with columns: date, actual, har_pred, xgb_har_pred, xgb_full_pred
    """
    har_cols  = HAR_COLS
    full_cols = FULL_COLS
    target    = "rv_target"

    n = len(features)
    records = []

    # --- Tune XGBoost once on the first window ---
    xgb_best_params = None
    if tune_xgb:
        print(f"Tuning XGBoost on first {min_train_size} training samples...")
        X_init = features.iloc[:min_train_size][full_cols]
        y_init = features.iloc[:min_train_size][target]
        tuner = XGBoostVolModel(name="tuner")
        tuner.fit(X_init, y_init, tune=True)
        xgb_best_params = tuner.model.get_params()
        print(f"Best XGB params: { {k: xgb_best_params[k] for k in ['n_estimators','max_depth','learning_rate','subsample','min_child_weight']} }")

    print(f"\nRunning expanding window evaluation...")
    print(f"Initial window: {min_train_size} days | Total predictions: {(n - min_train_size) // step}")

    indices = range(min_train_size, n - 1, step)

    for t in tqdm(indices, desc="Walk-forward"):
        train = features.iloc[:t]
        test_row = features.iloc[[t]]

        X_train_har  = train[har_cols]
        X_test_har   = test_row[har_cols]
        X_train_full = train[full_cols]
        X_test_full  = test_row[full_cols]
        y_train      = train[target]
        actual_val   = test_row[target].values[0]

        # --- HAR-RV ---
        ols_result = _fit_har(X_train_har, y_train)
        X_test_const = sm.add_constant(X_test_har, has_constant="add")
        har_pred = float(np.maximum(ols_result.predict(X_test_const).values[0], 1e-6))

        # --- XGB_HAR ---
        params_har = xgb_best_params.copy() if xgb_best_params else None
        xgb1 = XGBoostVolModel(name="XGB_HAR", params=params_har)
        xgb1.fit(X_train_har, y_train, tune=False)
        xgb_har_pred = float(xgb1.predict(X_test_har)[0])

        # --- XGB_FULL ---
        params_full = xgb_best_params.copy() if xgb_best_params else None
        xgb2 = XGBoostVolModel(name="XGB_FULL", params=params_full)
        xgb2.fit(X_train_full, y_train, tune=False)
        xgb_full_pred = float(xgb2.predict(X_test_full)[0])

        records.append({
            "date":          features.index[t],
            "actual":        actual_val,
            "har_pred":      har_pred,
            "xgb_har_pred":  xgb_har_pred,
            "xgb_full_pred": xgb_full_pred,
        })

    results_df = pd.DataFrame(records).set_index("date")
    return results_df


def summarize_rolling(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregate metrics from rolling evaluation results.
    """
    actual = results_df["actual"].values
    models = {
        "HAR-RV":   results_df["har_pred"].values,
        "XGB_HAR":  results_df["xgb_har_pred"].values,
        "XGB_FULL": results_df["xgb_full_pred"].values,
    }
    rows = []
    for name, preds in models.items():
        rows.append({
            "Model":  name,
            "RMSE":   round(rmse(actual, preds), 6),
            "MAE":    round(mae(actual, preds), 6),
            "QLIKE":  round(qlike(actual, preds), 6),
            "OOS R²": round(oos_r2(actual, preds), 4),
        })
    return pd.DataFrame(rows).set_index("Model")
