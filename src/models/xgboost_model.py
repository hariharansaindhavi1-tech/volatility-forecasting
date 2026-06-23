"""
xgboost_model.py
XGBoost models for realized volatility forecasting.

Model 1 (XGB_HAR): uses only HAR features (rv_d, rv_w, rv_m) — apples-to-apples with HAR-RV
Model 2 (XGB_FULL): uses HAR features + VIX features — tests whether implied vol adds signal
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# Default hyperparameters — reasonable starting point for daily financial data
DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


class XGBoostVolModel:
    """
    XGBoost volatility forecasting model.

    Parameters
    ----------
    name : str
        Label for this model variant (e.g., 'XGB_HAR' or 'XGB_FULL')
    params : dict, optional
        XGBoost hyperparameters. Defaults to DEFAULT_PARAMS.
    """

    def __init__(self, name: str = "XGBoost", params: dict = None):
        self.name = name
        self.params = params or DEFAULT_PARAMS.copy()
        self.model = XGBRegressor(**self.params)
        self.feature_cols = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, tune: bool = False):
        """
        Fit XGBoost model.

        Parameters
        ----------
        X_train : pd.DataFrame
        y_train : pd.Series
        tune    : bool
            If True, runs a time-series cross-validated grid search for hyperparameters.
            Adds runtime but improves results.
        """
        self.feature_cols = list(X_train.columns)

        if tune:
            print(f"\nTuning {self.name} hyperparameters via TimeSeriesCV...")
            self.model = self._tune(X_train, y_train)
        else:
            print(f"\nFitting {self.name}...")
            self.model.fit(X_train, y_train)

        print(f"{self.name} fitted on {len(X_train)} samples with features: {self.feature_cols}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions. Clips to small positive floor since RV is non-negative."""
        preds = self.model.predict(X[self.feature_cols])
        return np.maximum(preds, 1e-4)

    def feature_importance(self) -> pd.Series:
        """Return feature importances sorted descending."""
        importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_cols
        ).sort_values(ascending=False)
        return importance

    def _tune(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
        """
        Manual grid search with TimeSeriesSplit cross-validation.
        Avoids sklearn GridSearchCV compatibility issues with newer sklearn versions.
        """
        from sklearn.model_selection import TimeSeriesSplit
        from itertools import product

        tscv = TimeSeriesSplit(n_splits=5)

        param_grid = {
            "n_estimators":    [300, 500],
            "max_depth":       [3, 4, 5],
            "learning_rate":   [0.01, 0.05],
            "subsample":       [0.7, 0.8],
            "min_child_weight":[3, 5],
        }

        keys   = list(param_grid.keys())
        values = list(param_grid.values())
        best_score  = np.inf
        best_params = {}

        total = 1
        for v in values:
            total *= len(v)
        print(f"  Evaluating {total} parameter combinations × {tscv.n_splits} folds...")

        for combo in product(*values):
            params = dict(zip(keys, combo))
            fold_mses = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr = X_train.iloc[train_idx]
                y_tr = y_train.iloc[train_idx]
                X_val = X_train.iloc[val_idx]
                y_val = y_train.iloc[val_idx]

                m = XGBRegressor(
                    **params,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                    reg_lambda=1.0,
                    verbosity=0,
                )
                m.fit(X_tr, y_tr, verbose=False)
                preds = m.predict(X_val)
                fold_mses.append(np.mean((y_val.values - preds) ** 2))

            score = np.mean(fold_mses)
            if score < best_score:
                best_score  = score
                best_params = params

        print(f"  Best params: {best_params}  (CV MSE: {best_score:.8f})")

        best_model = XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            reg_lambda=1.0,
            verbosity=0,
        )
        best_model.fit(X_train, y_train, verbose=False)
        return best_model


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from data_loader import load_data
    from features import prepare_datasets

    df = load_data()
    X_train_har, X_test_har, X_train_full, X_test_full, y_train, y_test, idx = prepare_datasets(df)

    xgb1 = XGBoostVolModel(name="XGB_HAR")
    xgb1.fit(X_train_har, y_train)
    preds1 = xgb1.predict(X_test_har)
    print(f"\nXGB_HAR first 5 predictions: {preds1[:5]}")

    xgb2 = XGBoostVolModel(name="XGB_FULL")
    xgb2.fit(X_train_full, y_train)
    preds2 = xgb2.predict(X_test_full)
    print(f"XGB_FULL first 5 predictions: {preds2[:5]}")

    print("\nXGB_FULL feature importance:")
    print(xgb2.feature_importance())
