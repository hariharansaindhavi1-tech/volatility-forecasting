"""
har_rv.py
HAR-RV model (Heterogeneous Autoregressive model of Realized Volatility).

Corsi (2009): A Simple Approximate Long-Memory Model of Realized Volatility.
Model: RV_{t+1} = β0 + β1·RV_d + β2·RV_w + β3·RV_m + ε

Estimated via OLS using statsmodels for coefficient significance reporting.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


class HARModel:
    """
    HAR-RV model fitted via OLS.

    Features expected: rv_d, rv_w, rv_m
    """

    def __init__(self):
        self.model = None
        self.result = None
        self.feature_cols = ["rv_d", "rv_w", "rv_m"]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit OLS regression on HAR features."""
        X = sm.add_constant(X_train[self.feature_cols])
        self.result = sm.OLS(y_train, X).fit()
        print("\n--- HAR-RV Model Summary ---")
        print(self.result.summary())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions. Clips negative values to 0 (RV can't be negative)."""
        X_const = sm.add_constant(X[self.feature_cols], has_constant="add")
        preds = self.result.predict(X_const)
        return np.maximum(preds.values, 1e-4)
    def get_coefficients(self) -> pd.Series:
        """Return model coefficients."""
        return self.result.params


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from data_loader import load_data
    from features import prepare_datasets

    df = load_data()
    X_train_har, X_test_har, X_train_full, X_test_full, y_train, y_test, idx = prepare_datasets(df)

    model = HARModel()
    model.fit(X_train_har, y_train)
    preds = model.predict(X_test_har)
    print(f"\nFirst 5 predictions: {preds[:5]}")
    print(f"First 5 actuals:     {y_test.values[:5]}")
