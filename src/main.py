"""
main.py
Full pipeline: load data → build features → train models → evaluate.

Models:
    1. HAR-RV       : OLS on (rv_d, rv_w, rv_m)
    2. XGB_HAR      : XGBoost on (rv_d, rv_w, rv_m)
    3. XGB_FULL     : XGBoost on (rv_d, rv_w, rv_m, vix, vix_chg, ret_abs)

Evaluation modes:
    - Static  : train once on 80%, test on last 20%
    - Rolling : expanding window walk-forward (more realistic)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves without blocking
import matplotlib.pyplot as plt
import seaborn as sns

# Allow running from project root or src/
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_data
from features import prepare_datasets, build_full_features
from models.har_rv import HARModel
from models.xgboost_model import XGBoostVolModel
from evaluation import evaluate_all, run_dm_tests
from rolling_eval import rolling_evaluation, summarize_rolling


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
START_DATE       = "2015-01-01"
END_DATE         = "2025-01-01"
TEST_SIZE        = 0.2     # last 20% for static evaluation
TUNE_XGB         = True    # True = grid search on first window (adds ~5-10 min)
RUN_ROLLING      = True    # True = run walk-forward evaluation
ROLLING_STEP     = 21      # re-fit every 21 trading days (~monthly); use 1 for daily (slow)
MIN_TRAIN_SIZE   = 500     # minimum days before first prediction

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_predictions(test_index, actual, predictions: dict, title: str, save_path: str = None):
    """Plot actual vs predicted RV for all models."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_index, actual, label="Actual RV", color="black", linewidth=1.2, alpha=0.85)
    colors = ["steelblue", "tomato", "seagreen"]
    for (name, preds), color in zip(predictions.items(), colors):
        ax.plot(test_index, preds, label=name, linewidth=1, alpha=0.7, color=color)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized RV")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_feature_importance(xgb_full_model, save_path: str = None):
    """Bar chart of XGB_FULL feature importances."""
    importance = xgb_full_model.feature_importance()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = sns.color_palette("viridis", len(importance))
    ax.barh(importance.index, importance.values, color=colors)
    ax.set_title("XGB_FULL Feature Importances (Static Model)")
    ax.set_xlabel("Importance Score")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_rolling_errors(rolling_df: pd.DataFrame, save_path: str = None):
    """
    Rolling 63-day (quarterly) RMSE over time for each model.
    Shows how forecast accuracy evolves and which model is better in stress periods.
    """
    actual = rolling_df["actual"]
    model_cols = {
        "HAR-RV":   "har_pred",
        "XGB_HAR":  "xgb_har_pred",
        "XGB_FULL": "xgb_full_pred",
    }
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["black", "steelblue", "tomato", "seagreen"]
    for (name, col), color in zip(model_cols.items(), colors[1:]):
        sq_err = (actual - rolling_df[col]) ** 2
        rolling_rmse = sq_err.rolling(63).mean().apply(np.sqrt)
        ax.plot(rolling_df.index, rolling_rmse, label=name, linewidth=1.2, color=color)
    ax.set_title("Rolling 63-Day RMSE Over Time (Walk-Forward Evaluation)")
    ax.set_xlabel("Date")
    ax.set_ylabel("RMSE (63-day rolling)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_scatter(rolling_df: pd.DataFrame, save_path: str = None):
    """Predicted vs actual scatter plots for all three models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    model_cols = {
        "HAR-RV":   "har_pred",
        "XGB_HAR":  "xgb_har_pred",
        "XGB_FULL": "xgb_full_pred",
    }
    actual = rolling_df["actual"].values
    for ax, (name, col) in zip(axes, model_cols.items()):
        preds = rolling_df[col].values
        ax.scatter(actual, preds, alpha=0.3, s=8, color="steelblue")
        # 45-degree perfect prediction line
        lim = max(actual.max(), preds.max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect")
        ax.set_title(name)
        ax.set_xlabel("Actual RV")
        ax.set_ylabel("Predicted RV")
        ax.legend(fontsize=8)
    fig.suptitle("Predicted vs Actual RV — Walk-Forward Evaluation", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    df = load_data(start=START_DATE, end=END_DATE)

    # ------------------------------------------------------------------
    # 2. Static evaluation (train once, test on last 20%)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Static evaluation (single train/test split)")
    print("=" * 60)

    (
        X_train_har, X_test_har,
        X_train_full, X_test_full,
        y_train, y_test,
        test_index
    ) = prepare_datasets(df, test_size=TEST_SIZE)

    actual_static = y_test.values

    har = HARModel()
    har.fit(X_train_har, y_train)

    xgb_har = XGBoostVolModel(name="XGB_HAR")
    xgb_har.fit(X_train_har, y_train, tune=TUNE_XGB)

    xgb_full = XGBoostVolModel(name="XGB_FULL")
    xgb_full.fit(X_train_full, y_train, tune=TUNE_XGB)

    static_preds = {
        "HAR-RV":   har.predict(X_test_har),
        "XGB_HAR":  xgb_har.predict(X_test_har),
        "XGB_FULL": xgb_full.predict(X_test_full),
    }

    static_metrics = evaluate_all(static_preds, actual_static)
    static_dm      = run_dm_tests(static_preds, actual_static)

    print("\n--- Static Forecast Accuracy Metrics ---")
    print(static_metrics.to_string())
    print("\n--- Static Diebold-Mariano Tests ---")
    print(static_dm.to_string(index=False))

    static_metrics.to_csv(os.path.join(RESULTS_DIR, "static_metrics.csv"))
    static_dm.to_csv(os.path.join(RESULTS_DIR, "static_dm_tests.csv"), index=False)

    print("\nPlotting static results...")
    plot_predictions(
        test_index, actual_static, static_preds,
        title="Static Out-of-Sample RV Forecasts (80/20 Split)",
        save_path=os.path.join(RESULTS_DIR, "static_forecasts.png"),
    )
    plot_feature_importance(
        xgb_full,
        save_path=os.path.join(RESULTS_DIR, "feature_importance.png"),
    )

    # ------------------------------------------------------------------
    # 3. Rolling (walk-forward) evaluation
    # ------------------------------------------------------------------
    if RUN_ROLLING:
        print("\n" + "=" * 60)
        print("STEP 3: Rolling walk-forward evaluation")
        print(f"        step={ROLLING_STEP} days | min_train={MIN_TRAIN_SIZE} days")
        print("=" * 60)

        full_features = build_full_features(df).dropna()

        rolling_df = rolling_evaluation(
            features=full_features,
            min_train_size=MIN_TRAIN_SIZE,
            tune_xgb=TUNE_XGB,
            step=ROLLING_STEP,
        )

        rolling_metrics = summarize_rolling(rolling_df)
        rolling_dm = run_dm_tests(
            {
                "HAR-RV":   rolling_df["har_pred"].values,
                "XGB_HAR":  rolling_df["xgb_har_pred"].values,
                "XGB_FULL": rolling_df["xgb_full_pred"].values,
            },
            rolling_df["actual"].values,
        )

        print("\n--- Rolling Forecast Accuracy Metrics ---")
        print(rolling_metrics.to_string())
        print("\n--- Rolling Diebold-Mariano Tests ---")
        print(rolling_dm.to_string(index=False))

        rolling_df.to_csv(os.path.join(RESULTS_DIR, "rolling_predictions.csv"))
        rolling_metrics.to_csv(os.path.join(RESULTS_DIR, "rolling_metrics.csv"))
        rolling_dm.to_csv(os.path.join(RESULTS_DIR, "rolling_dm_tests.csv"), index=False)

        print("\nPlotting rolling results...")
        plot_predictions(
            rolling_df.index, rolling_df["actual"].values,
            {
                "HAR-RV":   rolling_df["har_pred"].values,
                "XGB_HAR":  rolling_df["xgb_har_pred"].values,
                "XGB_FULL": rolling_df["xgb_full_pred"].values,
            },
            title="Walk-Forward RV Forecasts (Expanding Window)",
            save_path=os.path.join(RESULTS_DIR, "rolling_forecasts.png"),
        )
        plot_rolling_errors(
            rolling_df,
            save_path=os.path.join(RESULTS_DIR, "rolling_rmse_over_time.png"),
        )
        plot_scatter(
            rolling_df,
            save_path=os.path.join(RESULTS_DIR, "rolling_scatter.png"),
        )

    print("\n" + "=" * 60)
    print(f"All results saved to: {os.path.abspath(RESULTS_DIR)}")
    print("=" * 60)
    print("\nDone.")


if __name__ == "__main__":
    main()
