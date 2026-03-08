# scripts/visualization.py
# -----------------------------
# Visualization utilities for Migraine Attack Risk Prediction.
# Includes exploratory plots, model evaluation curves, calibration plots,
# and SHAP-based interpretability figures.

import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)
from sklearn.calibration import calibration_curve


# ---------------------------------------
# 1. Numeric variable visualizations
# ---------------------------------------
def plot_numeric_distributions(df, num_cols):
    """
    Plot histograms for numeric variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    num_cols : list
        Numeric variable names.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(
        nrows=(len(num_cols) + 1) // 2,
        ncols=2,
        figsize=(12, 6)
    )
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        axes[i].hist(df[col], bins=20, color="skyblue", edgecolor="black")
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig


def plot_numeric_correlations(df, num_cols):
    """
    Plot correlation heatmap for numeric variables.

    Parameters
    ----------
    df : pd.DataFrame
    num_cols : list

    Returns
    -------
    matplotlib.figure.Figure
    """
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation between numeric variables")
    plt.tight_layout()
    return fig


# ---------------------------------------
# 2. Categorical variable visualizations
# ---------------------------------------
def plot_categorical_distributions(df, cat_cols):
    """
    Plot count distributions for categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
    cat_cols : list

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(
        nrows=(len(cat_cols) + 1) // 2,
        ncols=2,
        figsize=(12, 6)
    )
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        sns.countplot(x=col, data=df, ax=axes[i], color="lightcoral", edgecolor="black")
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig


# ---------------------------------------
# 3. Patient-level exploratory plots
# ---------------------------------------
def plot_episodes_per_patient(df):
    """
    Plot distribution of number of episodes per patient.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    matplotlib.figure.Figure
    """
    df_summary = df.groupby("id")["dos"].count().reset_index(name="episodes")
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.histplot(df_summary["episodes"], bins=20, color="skyblue", ax=ax)
    ax.set_title("Distribution of episodes per patient")
    ax.set_xlabel("Number of episodes")
    ax.set_ylabel("Number of patients")
    plt.tight_layout()
    return fig


def plot_median_dos_by_hatype(df):
    """
    Plot median study day per patient by attack type.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    matplotlib.figure.Figure
    """
    df_patient = df.groupby(["id", "hatype"])["dos"].median().reset_index(name="dos_median")
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(x="hatype", y="dos_median", data=df_patient, ax=ax)
    ax.set_title("Median study day per patient and hatype")
    ax.set_xlabel("Attack type (hatype)")
    ax.set_ylabel("Median study day (dos)")
    plt.tight_layout()
    return fig


def plot_dos_density_by_hatype(df):
    """
    Plot density of study day by attack type.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(data=df, x="dos", hue="hatype", fill=True, ax=ax)
    ax.set_title("Temporal density of episodes by hatype")
    ax.set_xlabel("Study day (dos)")
    ax.set_ylabel("Density")
    plt.tight_layout()
    return fig


def plot_boxplots(df, numeric_cols):
    """
    Plot boxplots for numeric variables.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_cols : list

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(
        nrows=(len(numeric_cols) + 1) // 2,
        ncols=2,
        figsize=(12, 6)
    )
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col], ax=axes[i], color="skyblue")
        axes[i].set_title(f"Boxplot of {col}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig


def plot_phase_distribution(df):
    """
    Plot distribution of treatment phase.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    df["phase"].value_counts().plot(kind="bar", ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_title("Distribution of records by phase")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Number of records")
    return fig


def plot_season_distribution(df):
    """
    Plot distribution of episodes by season.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(
        x="study_season",
        data=df,
        order=["winter", "spring", "summer", "autumn"],
        ax=ax
    )
    ax.set_title("Distribution by season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Count")
    return fig


def plot_airq_by_season(df):
    """
    Plot air quality distribution by season.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        x="study_season",
        y="airq",
        data=df,
        order=["winter", "spring", "summer", "autumn"],
        ax=ax
    )
    ax.set_title("Air quality distribution by season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Air quality (airq)")
    return fig


def plot_visits_distribution(df):
    """
    Plot distribution of number of visits per patient.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["visit_number"], bins=20, ax=ax)
    ax.set_title("Distribution of number of visits per patient")
    ax.set_xlabel("Number of visits")
    ax.set_ylabel("Count")
    return fig


# ---------------------------------------
# 4. Model evaluation curves
# ---------------------------------------
def plot_roc_curve(model, X, y, title, y_prob=None):
    """
    Plot ROC curve.

    Parameters
    ----------
    model : estimator
    X : array-like
    y : array-like
    title : str
    y_prob : array-like, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if y_prob is None:
        y_prob = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    return fig


def plot_pr_curve(model, X, y, title, y_prob=None):
    """
    Plot precision-recall curve.

    Parameters
    ----------
    model : estimator
    X : array-like
    y : array-like
    title : str
    y_prob : array-like, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if y_prob is None:
        y_prob = model.predict_proba(X)[:, 1]

    prec, rec, _ = precision_recall_curve(y, y_prob)
    ap = average_precision_score(y, y_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    return fig


# ---------------------------------------
# 5. Calibration curves
# ---------------------------------------
def plot_calibration_curves(models, calibrated_models, X, y, y_prob=None, title=None):
    """
    Plot calibration curves for raw and calibrated models.

    Parameters
    ----------
    models : dict
        Raw models.
    calibrated_models : dict
        Calibrators (isotonic, platt).
    X : array-like
    y : array-like
    y_prob : array-like, optional
    title : str, optional

    Returns
    -------
    dict or matplotlib.figure.Figure
    """
    # Single-model mode
    if y_prob is not None:
        prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(prob_pred, prob_true, "o-", label="Calibrated")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(title or "Calibration curve")
        ax.legend()
        return fig

    # Multi-model mode
    figs = {}

    for name, model in models.items():
        y_prob_raw = model.predict_proba(X)[:, 1]
        prob_true_raw, prob_pred_raw = calibration_curve(y, y_prob_raw, n_bins=10)

        cal_entry = calibrated_models[name]
        iso_cal = cal_entry.get("Isotonic") if isinstance(cal_entry, dict) else cal_entry
        platt_cal = cal_entry.get("Platt") if isinstance(cal_entry, dict) else None

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(prob_pred_raw, prob_true_raw, "s-", label="Raw")

        if iso_cal is not None:
            y_prob_iso = iso_cal.predict(y_prob_raw.reshape(-1, 1))
            prob_true_iso, prob_pred_iso = calibration_curve(y, y_prob_iso, n_bins=10)
            ax.plot(prob_pred_iso, prob_true_iso, "o-", label="Isotonic")

        if platt_cal is not None:
            y_prob_platt = platt_cal.predict(y_prob_raw.reshape(-1, 1))
            prob_true_platt, prob_pred_platt = calibration_curve(y, y_prob_platt, n_bins=10)
            ax.plot(prob_pred_platt, prob_true_platt, "^-", label="Platt")

        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"Calibration curve - {name}")
        ax.legend()

        figs[name] = fig

    return figs


# ---------------------------------------
# 6. Threshold analysis plots
# ---------------------------------------
def plot_probability_distribution(y_true, y_prob, title):
    """
    Plot distribution of predicted probabilities by class.

    Parameters
    ----------
    y_true : array-like
    y_prob : array-like
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="Positive", color="red")
    ax.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="Negative", color="green")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_cost_vs_threshold(df, title):
    """
    Plot cost as a function of threshold.

    Parameters
    ----------
    df : pd.DataFrame
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["Cost"], lw=2)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Cost")
    ax.set_title(title)
    return fig


def plot_sensitivity_vs_threshold(df, title):
    """
    Plot sensitivity as a function of threshold.

    Parameters
    ----------
    df : pd.DataFrame
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["Sens"], lw=2)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title)
    return fig


def plot_alerts_vs_threshold(df, title):
    """
    Plot mean alert rate as a function of threshold.

    Parameters
    ----------
    df : pd.DataFrame
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["Mean_alerts"], lw=2)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean alerts")
    ax.set_title(title)
    return fig


# ---------------------------------------
# 7. SHAP visualizations
# ---------------------------------------
def plot_shap_beeswarm(shap_values, X, output_path):
    """
    Save SHAP beeswarm plot.

    Parameters
    ----------
    shap_values : array-like
    X : pd.DataFrame
    output_path : str
    """
    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_bar(shap_values, X, output_path, top_n=20):
    """
    Save SHAP bar plot for top features.

    Parameters
    ----------
    shap_values : array-like
    X : pd.DataFrame
    output_path : str
    top_n : int
    """
    import numpy as np

    abs_mean = np.abs(shap_values).mean(axis=0)
    idx = abs_mean.argsort()[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(X.columns[idx][::-1], abs_mean[idx][::-1], color="#2E86C1")
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(
        f"Top {top_n} Features by Mean Absolute SHAP Value",
        fontsize=13,
        fontweight="bold"
    )
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_waterfall(shap_values, X, patient_idx, output_path):
    """
    Save SHAP waterfall plot for a specific patient.

    Parameters
    ----------
    shap_values : array-like
    X : pd.DataFrame
    patient_idx : int
    output_path : str
    """
    import numpy as np

    expl = shap.Explanation(
        values=shap_values[patient_idx],
        base_values=np.mean(shap_values),
        data=X.iloc[patient_idx].values,
        feature_names=list(X.columns)
    )

    shap.plots.waterfall(expl, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
