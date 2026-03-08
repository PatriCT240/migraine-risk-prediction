# scripts/modeling.py
# -----------------------------
# Modeling utilities for Migraine Attack Risk Prediction.
# Includes patient-level splitting, preprocessing pipelines,
# calibration metrics, model training, hyperparameter tuning,
# and probability extraction utilities.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib

from config import SEED, param_grids


# ---------------------------------------
# 1. Patient-level split
# ---------------------------------------
def patient_split(
    df: pd.DataFrame,
    patient_col: str = "id",
    test_size: float = 0.4,
    seed: int = SEED
):
    """
    Split dataset into train/validation/test ensuring patient-level separation.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    patient_col : str
        Column identifying patients.
    test_size : float
        Proportion for test+validation.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (train_df, valid_df, test_df)
    """
    unique_ids = df[patient_col].unique()
    train_ids, temp_ids = train_test_split(unique_ids, test_size=test_size, random_state=seed)
    valid_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=seed)

    train_df = df[df[patient_col].isin(train_ids)]
    valid_df = df[df[patient_col].isin(valid_ids)]
    test_df = df[df[patient_col].isin(test_ids)]

    return train_df, valid_df, test_df


# ---------------------------------------
# 2. Preprocessing pipeline
# ---------------------------------------
def build_preprocessor(num_cols, cat_cols):
    """
    Build preprocessing pipeline for numeric and categorical variables.

    Parameters
    ----------
    num_cols : list
        Numeric feature names.
    cat_cols : list
        Categorical feature names.

    Returns
    -------
    ColumnTransformer
        Preprocessing transformer.
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols)
    ])


# ---------------------------------------
# 3. Calibration metric (ECE)
# ---------------------------------------
def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_bins : int
        Number of bins.

    Returns
    -------
    float
        Expected calibration error.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        idx = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if np.any(idx):
            avg_conf = y_prob[idx].mean()
            avg_acc = y_true[idx].mean()
            ece += np.abs(avg_acc - avg_conf) * idx.mean()

    return ece


# ---------------------------------------
# 4. Metrics
# ---------------------------------------
def compute_metrics(
    model,
    X,
    y,
    threshold: float = 0.5,
    n_bins: int = 10
) -> dict:
    """
    Compute ROC-AUC, PR-AUC, Brier score, ECE, and confusion-matrix metrics.

    Parameters
    ----------
    model : estimator
        Trained classifier.
    X : array-like
        Feature matrix.
    y : array-like
        True labels.
    threshold : float
        Decision threshold.
    n_bins : int
        Number of bins for ECE.

    Returns
    -------
    dict
        Dictionary of metrics.
    """
    y_prob = model.predict_proba(X)[:, 1]

    roc = roc_auc_score(y, y_prob)
    pr = average_precision_score(y, y_prob)
    brier = brier_score_loss(y, y_prob)
    ece = expected_calibration_error(y, y_prob, n_bins=n_bins)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "ROC-AUC": roc,
        "PR-AUC": pr,
        "Brier": brier,
        "ECE": ece,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Precision": precision,
        "Sensitivity": sensitivity
    }


# ---------------------------------------
# 5. Train baseline models
# ---------------------------------------
def train_all_models(X_train, y_train, seed: int = SEED):
    """
    Train baseline models: Logistic Regression, Gradient Boosting, XGBoost.

    Parameters
    ----------
    X_train : array-like
    y_train : array-like
    seed : int

    Returns
    -------
    dict
        Dictionary of trained models.
    """
    models = {
        "logistic_regression": LogisticRegression(
            l1_ratio=0.25, solver="saga", max_iter=1000, random_state=seed
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=3, random_state=seed
        ),
        "xgboost": XGBClassifier(
            n_estimators=1000, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="aucpr",
            random_state=seed, tree_method="hist", n_jobs=-1
        )
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


# ---------------------------------------
# 6. Hyperparameter tuning
# ---------------------------------------
def build_base_model(name: str, seed: int = SEED):
    """
    Build base model for hyperparameter tuning.

    Parameters
    ----------
    name : str
        Model name.

    Returns
    -------
    estimator
    """
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    elif name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=seed)
    elif name == "XGBoost":
        return XGBClassifier(
            random_state=seed,
            use_label_encoder=False,
            eval_metric="aucpr",
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def tune_models(X_tr_prep, y_tr, groups, seed: int = SEED):
    """
    Tune models using GroupKFold cross-validation.

    Parameters
    ----------
    X_tr_prep : array-like
    y_tr : array-like
    groups : array-like
    seed : int

    Returns
    -------
    dict
        Best tuned models.
    """
    gkf = GroupKFold(n_splits=5)
    best_tuned_models = {}

    for name, grid in param_grids.items():
        print(f"Tuning {name}...")

        base = build_base_model(name, seed)
        gs = GridSearchCV(base, grid, cv=gkf, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_tr_prep, y_tr, groups=groups)

        best_tuned_models[name] = gs.best_estimator_
        print(f"Best params for {name}: {gs.best_params_}")

    return best_tuned_models


# ---------------------------------------
# 7. Train models with clinical flags
# ---------------------------------------
def train_models_with_flags(X_tr_flags, y_tr, seed: int = SEED):
    """
    Train models using engineered clinical flags.

    Parameters
    ----------
    X_tr_flags : pd.DataFrame
    y_tr : array-like
    seed : int

    Returns
    -------
    tuple
        (rl_model, gb_model, xgb_model)
    """
    rl_model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    rl_model.fit(X_tr_flags, y_tr)

    gb_model = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=3, random_state=seed
    )
    gb_model.fit(X_tr_flags, y_tr)

    xgb_model = XGBClassifier(
        n_estimators=1000, learning_rate=0.03, use_label_encoder=False, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
        objective="binary:logistic", eval_metric="aucpr", random_state=seed,
        tree_method="hist", n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_tr_flags, y_tr)

    return rl_model, gb_model, xgb_model


# ---------------------------------------
# 8. Probability extraction
# ---------------------------------------
def get_y_prob(model, calibrator, X):
    """
    Extract calibrated or uncalibrated probabilities depending on calibrator type.

    Parameters
    ----------
    model : estimator
        Base model.
    calibrator : estimator
        Calibration model.
    X : array-like
        Feature matrix.

    Returns
    -------
    np.ndarray
        Predicted probabilities.
    """
    X = np.asarray(X)
    raw = model.predict_proba(X)[:, 1]

    try:
        return calibrator.predict_proba(X)[:, 1]
    except AttributeError:
        return calibrator.predict(raw)
