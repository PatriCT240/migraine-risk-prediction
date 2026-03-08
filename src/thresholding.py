# scripts/thresholding.py
# -----------------------------
# Threshold optimization utilities for Migraine Attack Risk Prediction.
# Includes safe confusion matrix computation, metric extraction,
# cost-based threshold selection, and full threshold curve generation.

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# ---------------------------------------
# 1. Safe confusion matrix
# ---------------------------------------
def safe_confmat(y_true, y_pred):
    """
    Compute confusion matrix counts safely.

    Ensures TN, FP, FN, TP are always returned as integers,
    even when sklearn's confusion_matrix fails due to edge cases.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted binary labels.

    Returns
    -------
    tuple
        (TN, FP, FN, TP)
    """
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except Exception:
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tn, fp, fn, tp


# ---------------------------------------
# 2. Metrics from confusion counts
# ---------------------------------------
def metrics_from_counts(tn, fp, fn, tp):
    """
    Compute standard classification metrics from confusion matrix counts.

    Parameters
    ----------
    tn, fp, fn, tp : int
        Confusion matrix components.

    Returns
    -------
    dict
        Sensitivity, specificity, precision, F1, and Youden index.
    """
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    youden = sens + spec - 1

    return {
        "Sens": sens,
        "Spec": spec,
        "Precision": prec,
        "F1": f1,
        "Youden": youden
    }


# ---------------------------------------
# 3. Row computation for a given threshold
# ---------------------------------------
def compute_row(y_true, y_prob, thr, C_FP, C_FN):
    """
    Compute confusion counts, metrics, cost, and alert rate for a given threshold.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_prob : array-like
        Predicted probabilities.
    thr : float
        Decision threshold.
    C_FP : float
        Cost of a false positive.
    C_FN : float
        Cost of a false negative.

    Returns
    -------
    dict
        Metrics, cost, and alert rate for the threshold.
    """
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = safe_confmat(y_true, y_pred)
    mets = metrics_from_counts(tn, fp, fn, tp)

    cost = C_FN * fn + C_FP * fp
    mean_alerts = y_pred.mean()

    return {
        "threshold": thr,
        "C_FP": C_FP,
        "C_FN": C_FN,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Cost": cost,
        "Mean_alerts": mean_alerts,
        **mets
    }


# ---------------------------------------
# 4. Optimize threshold for a cost pair
# ---------------------------------------
def optimize_for_cost_pair(y_true, y_prob, thresholds, C_FP, C_FN, sens_min):
    """
    Select the best threshold for a given (C_FP, C_FN) pair.

    Rules
    -----
    1. Primary objective: minimize Cost.
    2. Constraint: Sensitivity >= sens_min (if possible).
    3. Tie-breaker: higher Youden index.
    4. Fallback: if no threshold meets Sens >= sens_min,
       choose the one with max Sens, then Youden.

    Parameters
    ----------
    y_true : array-like
    y_prob : array-like
    thresholds : array-like
    C_FP : float
    C_FN : float
    sens_min : float

    Returns
    -------
    dict
        Best threshold row for the cost pair.
    """
    best = None

    # First pass: enforce Sens >= sens_min
    for thr in thresholds:
        row = compute_row(y_true, y_prob, thr, C_FP, C_FN)

        if row["Sens"] >= sens_min:
            if best is None:
                best = row
            else:
                if (row["Cost"] < best["Cost"]) or \
                   (row["Cost"] == best["Cost"] and row["Youden"] > best["Youden"]):
                    best = row

    # Fallback: no threshold meets Sens >= sens_min
    if best is None:
        best = max(
            (compute_row(y_true, y_prob, thr, C_FP, C_FN) for thr in thresholds),
            key=lambda r: (r["Sens"], r["Youden"])
        )

    return best


# ---------------------------------------
# 5. Full grid optimization
# ---------------------------------------
def optimize_threshold_grid(
    y_true,
    y_prob,
    thresholds,
    C_FP_range,
    C_FN_range,
    sens_min
):
    """
    Run full grid search over C_FP × C_FN and thresholds.

    Parameters
    ----------
    y_true : array-like
    y_prob : array-like
    thresholds : array-like
    C_FP_range : list
    C_FN_range : list
    sens_min : float

    Returns
    -------
    tuple
        (candidates_df, best_global)
    """
    candidates = []

    for C_FP in C_FP_range:
        for C_FN in C_FN_range:
            best_pair = optimize_for_cost_pair(
                y_true, y_prob, thresholds, C_FP, C_FN, sens_min
            )
            candidates.append(best_pair)

    candidates_df = pd.DataFrame(candidates)
    candidates_df = candidates_df.sort_values(
        ["Cost", "Youden"], ascending=[True, False]
    ).reset_index(drop=True)

    best_global = candidates_df.iloc[0]

    return candidates_df, best_global


# ---------------------------------------
# 6. Detailed threshold curve
# ---------------------------------------
def compute_detailed_curve(y_true, y_prob, thresholds, C_FP, C_FN):
    """
    Compute full threshold curve (metrics + cost + alerts)
    for a selected (C_FP, C_FN) pair.

    Parameters
    ----------
    y_true : array-like
    y_prob : array-like
    thresholds : array-like
    C_FP : float
    C_FN : float

    Returns
    -------
    pd.DataFrame
        Detailed threshold curve.
    """
    rows = [
        compute_row(y_true, y_prob, thr, C_FP, C_FN)
        for thr in thresholds
    ]
    return pd.DataFrame(rows)
