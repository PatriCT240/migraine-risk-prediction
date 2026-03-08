# scripts/features.py
# -----------------------------
# Feature engineering utilities for Migraine Attack Risk Prediction.
# Includes temporal features, demographic groupings, clinical history
# features, and utilities for loading preprocessed datasets.

import pandas as pd
import numpy as np
import joblib

from config import TARGET


# ---------------------------------------
# 1. Treatment phase
# ---------------------------------------
def add_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a categorical variable indicating treatment phase.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'phase' column added.
    """
    df["phase"] = np.where(df["time"] < 0, "No treatment", "Under treatment")
    return df


# ---------------------------------------
# 2. Study date and seasonality features
# ---------------------------------------
def add_study_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add study date, month, and season based on 'dos' (days on study).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with date-based features added.
    """
    df["study_date"] = pd.to_datetime("1997-01-01") + pd.to_timedelta(df["dos"], unit="D")
    df["study_month"] = df["study_date"].dt.month

    season_map = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }

    df["study_season"] = df["study_month"].map(season_map)
    return df


# ---------------------------------------
# 3. Temporal progression features
# ---------------------------------------
def add_days_since_first_visit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute days since first visit for each patient.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df["days_since_first_visit"] = df.groupby("id")["dos"].transform(lambda x: x - x.min())
    return df


def add_visit_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add visit number per patient (1, 2, 3, ...).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df["visit_number"] = df.groupby("id").cumcount() + 1
    return df


# ---------------------------------------
# 4. Demographic features
# ---------------------------------------
def add_age_band(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add age band categories.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df["age_band"] = pd.cut(
        df["age"],
        bins=[17, 30, 40, 50, 66],
        labels=["18-30", "31-40", "41-50", "51-66"],
    )
    return df


# ---------------------------------------
# 5. Target encoding
# ---------------------------------------
def add_target_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert binary target ('yes'/'no') into numeric (1/0).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df["target_num"] = df[TARGET].map({"yes": 1, "no": 0})
    return df


# ---------------------------------------
# 6. Clinical history features
# ---------------------------------------
def add_prev_airq_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add mean air quality from previous visits.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df["airq_prev_mean"] = (
        df.groupby("id")["airq"]
          .transform(lambda x: x.shift(1).expanding().mean())
    )
    return df


def add_prev_attacks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cumulative number of previous migraine attacks.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.sort_values(["id", "time"])
    df["prev_attacks"] = (
        df.groupby("id")["target_num"]
          .cumsum()
          .shift()
          .fillna(0)
    )
    return df


# ---------------------------------------
# 7. Load preprocessed dataset with feature names
# ---------------------------------------
def load_preprocessed_with_names(
    path_csv: str,
    preprocessor_path: str
) -> pd.DataFrame:
    """
    Load preprocessed dataset and assign feature names from a saved preprocessor.

    Parameters
    ----------
    path_csv : str
        Path to transformed dataset.
    preprocessor_path : str
        Path to fitted preprocessor (.pkl).

    Returns
    -------
    pd.DataFrame
        Dataframe with correct feature names.
    """
    pre = joblib.load(preprocessor_path)
    feature_names = pre.get_feature_names_out()

    df = pd.read_csv(path_csv)
    return pd.DataFrame(df.values, columns=feature_names)


# ---------------------------------------
# 8. Clinical flags for model comparison
# ---------------------------------------
def create_clinical_flags(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add simple clinical flags to training and test sets.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test : pd.DataFrame

    Returns
    -------
    tuple
        (X_train_with_flags, X_test_with_flags)
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["high_attacks"] = (X_train["num__prev_attacks"] > 10).astype(int)
    X_train["high_visits"] = (X_train["num__visit_number"] > 5).astype(int)
    X_train["overuse_med"] = (X_train["num__prev_attacks"] > 3).astype(int)

    X_test["high_attacks"] = (X_test["num__prev_attacks"] > 10).astype(int)
    X_test["high_visits"] = (X_test["num__visit_number"] > 5).astype(int)
    X_test["overuse_med"] = (X_test["num__prev_attacks"] > 3).astype(int)

    return X_train, X_test
