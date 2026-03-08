# scripts/config.py
# -----------------------------
# Configuration settings for Migraine Attack Risk Prediction.
# Includes target definition, feature groups, columns to drop,
# hyperparameter grids, and thresholding configuration.

import numpy as np


# ---------------------------------------
# 1. Target variable and seed
# ---------------------------------------
TARGET = "headache"
SEED = 42


# ---------------------------------------
# 2. Feature groups
# ---------------------------------------
num_cols = [
    "prev_attacks",
    "visit_number",
    "days_since_first_visit",
    "airq",
    "airq_prev_mean",
]

cat_cols = [
    "hatype",
    "medication",
    "sex",
    "study_season",
    "phase",
    "age_band",
]

target_col = "target_num"

drop_cols = [
    "rownames",
    "id",
    "sample_weight",
    target_col,
    "time",
    "dos",
    "target",
    "headache",
    "age",
    "month_study",
    "study_date",
    "study_month",
    "rownames_is_outlier",
    "id_is_outlier",
    "time_is_outlier",
    "dos_is_outlier",
    "age_is_outlier",
    "airq_is_outlier",
]


# ---------------------------------------
# 3. Hyperparameter grids
# ---------------------------------------
param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "l1_ratio": [0, 0.5, 1],
        "solver": ["saga"],
        "max_iter": [1000],
    },
    "Gradient Boosting": {
        "n_estimators": [200, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    },
    "XGBoost": {
        "n_estimators": [500, 1000],
        "learning_rate": [0.03, 0.1],
        "max_depth": [3, 4],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [1.0, 2.0],
        "reg_alpha": [0.0, 0.5],
        "tree_method": ["hist"],
        "n_jobs": [-1],
        "verbosity": [0],
        "use_label_encoder": [False],
        "eval_metric": ["aucpr"],
    },
}


# ---------------------------------------
# 4. Thresholding configuration
# ---------------------------------------
SENS_MIN = 0.80

C_FP_RANGE = list(range(1, 11))  # 1..10
C_FN_RANGE = list(range(1, 11))  # 1..10

N_THRESH = 1001
THRESHOLDS = np.linspace(0.0, 1.0, N_THRESH)
