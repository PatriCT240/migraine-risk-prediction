# scripts/calibration.py
# -----------------------------
# Calibration utilities for Migraine Attack Risk Prediction.
# Includes manual calibration (Isotonic / Platt), calibrated prediction,
# probability-based metrics, and bootstrap confidence intervals.

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score
)
from sklearn.model_selection import GroupKFold
from sklearn.calibration import calibration_curve

from modeling import expected_calibration_error


# ---------------------------------------
# 1. Manual calibration (Isotonic / Platt)
# ---------------------------------------
def manual_calibration(
    base_model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    method: str = "isotonic"
):
    """
    Fit a calibration model (Isotonic or Platt scaling) using out-of-fold predictions.

    Parameters
    ----------
    base_model : estimator
        Base classifier with predict_proba.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        True labels.
    groups : np.ndarray
        Group identifiers for GroupKFold.
    method : str
        Calibration method: "isotonic" or "platt".

    Returns
    -------
    Tuple
        (base_model, calibrator)
    """
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in gkf.split(X, y, groups):
        model = clone(base_model)
        model.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
    elif method == "platt":
        calibrator = LogisticRegression(solver="lbfgs")
    else:
        raise ValueError("Method must be 'isotonic' or 'platt'.")

    calibrator.fit(oof_preds.reshape(-1, 1), y)
    return base_model, calibrator


# ---------------------------------------
# 2. Predict calibrated probabilities
# ---------------------------------------
def predict_calibrated(
    base_model,
    calibrator,
    X: np.ndarray
) -> np.ndarray:
    """
    Predict calibrated probabilities using a fitted base model and calibrator.

    Parameters
    ----------
    base_model : estimator
        Trained classifier with predict_proba.
    calibrator : estimator
        Fitted calibration model.
    X : np.ndarray
        Feature matrix.

    Returns
    -------
    np.ndarray
        Calibrated predicted probabilities.
    """
    raw = base_model.predict_proba(X)[:, 1]
    return calibrator.predict(raw.reshape(-1, 1))


# ---------------------------------------
# 3. Metrics from probabilities
# ---------------------------------------
def metrics_from_probs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 10
) -> Dict:
    """
    Compute ROC-AUC, PR-AUC, Brier score, ECE, and confusion-matrix metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.
    threshold : float
        Decision threshold.
    n_bins : int
        Number of bins for ECE.

    Returns
    -------
    Dict
        Dictionary with calibration and classification metrics.
    """
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    y_pred_bin = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "ROC-AUC": roc,
        "PR-AUC": pr,
        "Brier": brier,
        "ECE": ece,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Sensitivity": sensitivity
    }


# ---------------------------------------
# 4. Bootstrap for calibrated model
# ---------------------------------------
def bootstrap_calibrated(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Bootstrap distributions for calibration and classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_boot : int
        Number of bootstrap samples.
    seed : int
        Random seed.

    Returns
    -------
    Dict
        Bootstrap distributions for ROC-AUC, PR-AUC, Brier, ECE,
        confusion-matrix components, precision, and sensitivity.
    """
    rng = np.random.default_rng(seed)

    roc_bs, pr_bs, brier_bs, ece_bs = [], [], [], []
    tp_bs, tn_bs, fp_bs, fn_bs = [], [], [], []
    prec_bs, sens_bs = [], []

    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        yt = y_true[idx]
        yp = y_prob[idx]

        roc_bs.append(roc_auc_score(yt, yp))
        pr_bs.append(average_precision_score(yt, yp))
        brier_bs.append(brier_score_loss(yt, yp))
        ece_bs.append(expected_calibration_error(yt, yp))

        tn, fp, fn, tp = confusion_matrix(yt, (yp > 0.5).astype(int)).ravel()
        tp_bs.append(tp)
        tn_bs.append(tn)
        fp_bs.append(fp)
        fn_bs.append(fn)

        prec_bs.append(precision_score(yt, (yp > 0.5).astype(int), zero_division=0))
        sens_bs.append(recall_score(yt, (yp > 0.5).astype(int), zero_division=0))

    return {
        "roc": roc_bs,
        "pr": pr_bs,
        "brier": brier_bs,
        "ece": ece_bs,
        "tp": tp_bs,
        "tn": tn_bs,
        "fp": fp_bs,
        "fn": fn_bs,
        "precision": prec_bs,
        "sensitivity": sens_bs
    }
