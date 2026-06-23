"""
features.py
Computes realized volatility from SPY returns and builds the HAR feature matrix.

RV proxy: squared daily log return (close-to-close)
HAR lags:
  - RV_d  : daily RV (t)
  - RV_w  : 5-day average RV (weekly)
  - RV_m  : 22-day average RV (monthly)

Target: RV shifted 1 day forward (next-day RV)
"""

import pandas as pd
import numpy as np


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """Compute daily log returns from SPY close prices."""
    log_ret = np.log(df["close"] / df["close"].shift(1))
    log_ret.name = "log_return"
    return log_ret


def compute_rv(log_returns: pd.Series) -> pd.Series:
    """
    Realized volatility proxy: squared daily log return.
    Annualized by multiplying by 252.
    """
    rv = log_returns ** 2 * 252
    rv.name = "rv"
    return rv


def build_har_features(rv: pd.Series) -> pd.DataFrame:
    """
    Build the HAR-RV feature matrix from a realized volatility series.

    Columns:
        rv_d  - daily RV at time t
        rv_w  - 5-day rolling mean of RV (weekly component)
        rv_m  - 22-day rolling mean of RV (monthly component)
        rv_target - RV at t+1 (what we are predicting)
    """
    features = pd.DataFrame(index=rv.index)
    features["rv_d"] = rv
    features["rv_w"] = rv.rolling(window=5).mean()
    features["rv_m"] = rv.rolling(window=22).mean()
    # Target is next day's RV
    features["rv_target"] = rv.shift(-1)
    return features


def build_full_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature matrix including HAR lags and VIX.
    Used for XGBoost models.

    Additional features beyond HAR:
        vix       - VIX close at time t
        vix_chg   - 1-day change in VIX
        ret_abs   - absolute log return (magnitude of move)
    """
    log_ret = compute_log_returns(df)
    rv = compute_rv(log_ret)

    features = build_har_features(rv)

    # Add VIX features
    features["vix"] = df["vix"]
    features["vix_chg"] = df["vix"].diff()

    # Add return magnitude
    features["ret_abs"] = log_ret.abs()

    return features


def prepare_datasets(df: pd.DataFrame, test_size: float = 0.2):
    """
    Build and split feature matrix into train/test sets.
    Preserves time order — no shuffling.

    Returns:
        X_train_har, X_test_har   : HAR features only (rv_d, rv_w, rv_m)
        X_train_full, X_test_full : HAR + VIX features
        y_train, y_test           : target RV series
        test_index                : dates for the test set
    """
    full = build_full_features(df)

    # Drop rows with NaN (from rolling windows and shift)
    full = full.dropna()

    har_cols = ["rv_d", "rv_w", "rv_m"]
    full_cols = ["rv_d", "rv_w", "rv_m", "vix", "vix_chg", "ret_abs"]
    target_col = "rv_target"

    split_idx = int(len(full) * (1 - test_size))

    train = full.iloc[:split_idx]
    test = full.iloc[split_idx:]

    X_train_har = train[har_cols]
    X_test_har = test[har_cols]

    X_train_full = train[full_cols]
    X_test_full = test[full_cols]

    y_train = train[target_col]
    y_test = test[target_col]

    print(f"Train: {len(train)} days | Test: {len(test)} days")
    print(f"Test period: {test.index[0].date()} to {test.index[-1].date()}")

    return (
        X_train_har, X_test_har,
        X_train_full, X_test_full,
        y_train, y_test,
        test.index
    )


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    result = prepare_datasets(df)
    X_train_har, X_test_har, X_train_full, X_test_full, y_train, y_test, idx = result
    print("\nHAR feature sample:")
    print(X_train_har.tail())
    print("\nTarget sample:")
    print(y_train.tail())
