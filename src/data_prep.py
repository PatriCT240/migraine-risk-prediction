# scripts/data_prep.py
# -----------------------------
# Data preparation utilities for Migraine Attack Risk Prediction.
# Includes raw data loading, variable typing helpers, frequency tables,
# outlier detection, and basic cleaning routines.

from typing import Tuple, Dict, List
import pandas as pd
import numpy as np


# ---------------------------------------
# 1. Load raw dataset
# ---------------------------------------
def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw migraine dataset from CSV.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    df = pd.read_csv(path, sep=",")
    return df.copy()


# ---------------------------------------
# 2. Variable type helpers
# ---------------------------------------
def get_numeric_vars(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Identify numeric variables (int, float), excluding specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    exclude : list, optional
        Columns to exclude.

    Returns
    -------
    list
        List of numeric variable names.
    """
    exclude = exclude or []
    numeric_types = ["int64", "float64"]

    return [
        col for col in df.columns
        if df[col].dtype.name in numeric_types and col not in exclude
    ]


def get_categorical_vars(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Identify categorical variables (object, category, bool, string), excluding specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    exclude : list, optional
        Columns to exclude.

    Returns
    -------
    list
        List of categorical variable names.
    """
    exclude = exclude or []
    categorical_types = ["object", "category", "bool", "string", "str"]

    return [
        col for col in df.columns
        if df[col].dtype.name in categorical_types and col not in exclude
    ]


# ---------------------------------------
# 3. Categorical frequencies
# ---------------------------------------
def compute_categorical_frequencies(
    df: pd.DataFrame,
    categorical_vars: List[str]
) -> pd.DataFrame:
    """
    Compute absolute and percentage frequencies for categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    categorical_vars : list
        List of categorical variable names.

    Returns
    -------
    pd.DataFrame
        Frequency table with variable, category, count, and percentage.
    """
    tables = []

    for col in categorical_vars:
        abs_count = df[col].value_counts(dropna=False)
        pct_count = df[col].value_counts(normalize=True, dropna=False) * 100

        tmp = pd.DataFrame({
            "variable": col,
            "category": abs_count.index,
            "frequency": abs_count.values,
            "percentage": pct_count.round(2).values
        })

        tables.append(tmp)

    return pd.concat(tables, ignore_index=True)


# ---------------------------------------
# 4. Outlier detection
# ---------------------------------------
def detect_outliers(df: pd.DataFrame, col: str):
    """
    Detect outliers in a numeric column using the IQR rule.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col : str
        Column name.

    Returns
    -------
    tuple
        (outlier_rows, lower_bound, upper_bound)
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return outliers, lower, upper


def report_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> Dict:
    """
    Detect and report outliers for all numeric variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    numeric_cols : list
        List of numeric variable names.

    Returns
    -------
    dict
        Dictionary with outlier information per column.
    """
    results = {}

    for col in numeric_cols:
        outliers, lower, upper = detect_outliers(df, col)
        results[col] = {
            "outliers": outliers,
            "lower": lower,
            "upper": upper
        }

    return results


# ---------------------------------------
# 5. Basic cleaning
# ---------------------------------------
def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform minimal cleaning on the raw migraine dataset.

    Steps
    -----
    - Strip and lowercase column names
    - Remove duplicate rows
    - Drop rows with missing target
    - Print missing value report

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()

    df.columns = df.columns.str.strip().str.lower()
    df = df.drop_duplicates()

    if "target_num" in df.columns:
        df = df.dropna(subset=["target_num"])

    missing_report = df.isna().sum()
    print("Missing values per column:\n", missing_report)

    return df


# ---------------------------------------
# 6. Outlier flagging
# ---------------------------------------
def flag_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Add a boolean flag column indicating outliers in a numeric variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col : str
        Column to evaluate.

    Returns
    -------
    pd.DataFrame
        Dataframe with an added outlier flag column.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    flag_col = f"{col}_is_outlier"
    df[flag_col] = (df[col] < lower) | (df[col] > upper)

    return df
