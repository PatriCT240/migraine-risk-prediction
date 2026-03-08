# scripts/evaluation.py
# -----------------------------
# Evaluation utilities for Migraine Attack Risk Prediction.
# Includes model comparison with outlier flags, delta computation,
# calibrated model evaluation with bootstrap, and SHAP value computation.

import pandas as pd
import numpy as np
import shap

from config import SEED
from calibration import metrics_from_probs, bootstrap_calibrated


# ---------------------------------------
# 1. Evaluation with outlier flags
# ---------------------------------------
def evaluate_models_with_flags(
    rl_model,
    gb_model,
    xgb_model,
    X_test,
    y_test,
    metrics_fn
) -> pd.DataFrame:
    """
    Evaluate multiple models using a metrics function that incorporates
    outlier flags or additional preprocessing.

    Parameters
    ----------
    rl_model : estimator
        Logistic regression model.
    gb_model : estimator
        Gradient boosting model.
    xgb_model : estimator
        XGBoost model.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True labels.
    metrics_fn : callable
        Function that computes metrics for a given model.

    Returns
    -------
    pd.DataFrame
        Metrics for each model with flags applied.
    """
    results = {
        "logistic_regression_flags": metrics_fn(rl_model, X_test, y_test),
        "gradient_boosting_flags": metrics_fn(gb_model, X_test, y_test),
        "xgboost_flags": metrics_fn(xgb_model, X_test, y_test),
    }

    return pd.DataFrame(results).T


def compare_baseline_vs_flags(
    metrics_base_df: pd.DataFrame,
    metrics_flags_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Concatenate baseline and flagged model metrics.

    Parameters
    ----------
    metrics_base_df : pd.DataFrame
        Baseline model metrics.
    metrics_flags_df : pd.DataFrame
        Metrics after applying outlier flags.

    Returns
    -------
    pd.DataFrame
        Combined metrics table.
    """
    return pd.concat([metrics_base_df, metrics_flags_df], axis=0)


def compute_deltas_for_all_models(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute absolute and relative deltas between baseline and flagged models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Combined metrics table containing baseline and flagged rows.

    Returns
    -------
    pd.DataFrame
        Long-format table with deltas for each model and metric.
    """
    deltas = []

    for model in ["logistic_regression", "gradient_boosting", "xgboost"]:
        base = comparison_df.loc[model]
        flags = comparison_df.loc[f"{model}_flags"]

        delta = pd.DataFrame({
            "Metric": base.index,
            "Baseline": base.values,
            "With Flags": flags.values,
            "Absolute Δ": flags.values - base.values,
            "Relative Δ (%)": ((flags.values - base.values) / base.values) * 100,
            "Model": model,
        })

        deltas.append(delta)

    return pd.concat(deltas, axis=0)


# ---------------------------------------
# 2. Calibrated model evaluation (bootstrap)
# ---------------------------------------
def evaluate_calibrated_model_with_bootstrap(
    base_model,
    calibrator,
    X_test,
    y_test,
    n_boot: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Evaluate a calibrated model using bootstrap resampling.

    Parameters
    ----------
    base_model : estimator
        Trained classifier with predict_proba.
    calibrator : estimator
        Fitted calibration model.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True labels.
    n_boot : int
        Number of bootstrap samples.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Summary table with bootstrap mean, std, CI95, and point estimates.
    """
    X_test = np.asarray(X_test)
    y_true = np.asarray(y_test)

    raw = base_model.predict_proba(X_test)[:, 1]
    y_prob = calibrator.predict(raw.reshape(-1, 1))

    point_metrics = metrics_from_probs(y_true, y_prob)
    boot = bootstrap_calibrated(y_true, y_prob, n_boot=n_boot, seed=seed)

    key_map = {
        "roc": "ROC-AUC",
        "pr": "PR-AUC",
        "brier": "Brier",
        "ece": "ECE",
        "tp": "TP",
        "tn": "TN",
        "fp": "FP",
        "fn": "FN",
        "precision": "Precision",
        "sensitivity": "Sensitivity",
    }

    summary = {}
    for metric, values in boot.items():
        arr = np.array(values)
        mapped_key = key_map.get(metric)
        point_value = point_metrics.get(mapped_key)

        summary[metric] = {
            "Mean": arr.mean(),
            "Std": arr.std(),
            "CI95_low": np.quantile(arr, 0.025),
            "CI95_high": np.quantile(arr, 0.975),
            "Point": point_value,
        }

    return pd.DataFrame(summary).T


# ---------------------------------------
# 3. SHAP value computation
# ---------------------------------------
def compute_shap(
    model_or_path,
    X_or_path,
    n_background: int = 200,
    n_samples: int = 100
):
    """
    Compute SHAP values for a model and dataset.

    Accepts either:
    - loaded objects (Python notebook workflow)
    - file paths (R via reticulate workflow)

    Parameters
    ----------
    model_or_path : estimator or str
        Model object or path to .pkl file.
    X_or_path : DataFrame or str
        Feature matrix or path to .csv file.
    n_background : int
        Number of background samples for KernelExplainer.
    n_samples : int
        Number of SHAP samples.

    Returns
    -------
    tuple
        (shap_values, X_df)
    """
    import shap
    import joblib
    import pandas as pd

    model = joblib.load(model_or_path) if isinstance(model_or_path, str) else model_or_path
    X = pd.read_csv(X_or_path) if isinstance(X_or_path, str) else X_or_path

    if n_samples is not None:
        predict_fn = lambda data: model.predict_proba(data)[:, 1]
        background = shap.sample(X, n_background, random_state=42)
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X, nsamples=n_samples)
    else:
        explainer = shap.LinearExplainer(model, masker=shap.maskers.Independent(X))
        shap_values = explainer.shap_values(X)

    return shap_values, X
