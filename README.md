# Migraine Attack Risk Prediction
### A Clinically Interpretable Machine Learning Pipeline

**Author:** Patricia C. Torrell · Clinical Data Analyst  
**Stack:** Python · R · scikit-learn · tidyverse · SHAP · ggplot2  
**Domain:** Neurology · Longitudinal Clinical Data · Personalised Medicine

---

## Overview

This project develops a fully transparent, end-to-end machine learning pipeline to estimate the probability of a migraine attack from longitudinal clinical, treatment, and environmental data.

The pipeline goes beyond prediction: every modelling decision is clinically justified, every output is interpretable, and the results are validated at both population and patient level. It is designed to reflect the standards expected in a pharmaceutical or clinical research environment — reproducible, auditable, and medically grounded.

Full variable definitions and clinical coding are documented in [`docs/data_dictionary.md`](docs/data_dictionary.md).

> **Core question:** *Can we predict the onset of a migraine attack using available clinical and environmental data — and if so, who is most sensitive to treatment and environment?*

---

## What This Project Does

| Stage | Description |
|---|---|
| **EDA** | Longitudinal structure analysis, target prevalence, distribution of clinical and environmental variables |
| **Feature Engineering** | Treatment phase classification, seasonal decomposition, cumulative attack history, rolling air quality mean, age bands |
| **Modeling** | Logistic Regression, Gradient Boosting, XGBoost — trained with patient-level cross-validation (GroupKFold) |
| **Calibration** | Isotonic and Platt calibration with honest out-of-fold evaluation |
| **Threshold Optimisation** | FP/FN cost-based search with minimum sensitivity constraint (≥0.80) |
| **Subgroup & Fairness Analysis** | OR estimation with clustered robust SEs across sex, age, medication, subtype, air quality quartiles |
| **Interpretability** | Permutation Importance, PDP, ALE (global + stratified), ICE per patient, sensitivity analysis (demonstrate clinical heterogeneity) and SHAP (global + per patient).

---

## Final Model

**Logistic Regression — tuned + isotonic calibration**

Chosen for interpretability, clinical plausibility, and consistent performance across all evaluation criteria.

| Metric | Value | 95% CI (bootstrap) |
|---|---|---|
| ROC-AUC | 0.715 | [0.681 – 0.749] |
| PR-AUC | 0.754 | [0.711 – 0.792] |
| Brier Score | 0.214 | [0.200 – 0.229] |
| ECE | 0.074 	 | [0.051 – 0.099] |
| Precision | 0.672 | [0.638 – 0.708] |
| Sensitivity | 0.867 | [0.836 – 0.894] |
| Optimal threshold | 0.47 | FP/FN cost-optimised |

All metrics validated on a held-out test set of unseen patients (honest patient-level split).

---

## Key Clinical Findings

### What drives migraine risk?
Global interpretability (Permutation Importance + SHAP) consistently identifies three dominant predictors:

- **Prior attack history** → strongest predictor of future risk
- **Medication status** → reduced medication significantly increases attack risk (OR = 2.09 vs continuing, p = 0.023)
- **Clinical follow-up intensity** → more visits act as a protective factor

### Does treatment reduce risk?
The treatment phase alone does not reach statistical significance, but medication type does — strongly. Patients with reduced medication show the highest attack proportion (75.8%), above those on continuous treatment (58.8%) and those on no medication (40.5%), confirming a confounding-by-indication effect consistent with clinical practice.

### Does air quality matter?
No significant association was found between air quality quartile and attack risk. All quartiles show odds ratios very close to 1 (Q2: OR = 0.97, p = 0.806; Q3: OR = 1.00, p = 0.989; Q4: OR = 1.05, p = 0.767), with wide confidence intervals and no consistent directional trend. Air quality does not emerge as a reliable predictor in this model.

### Is clinical response heterogeneous?
Yes — and this is one of the most clinically relevant findings of the project. ICE curves and patient-level sensitivity indices show that:

- The same treatment produces very different predicted risk changes across patients
- Some patients are highly sensitive to environmental exposure; others are not
- Risk is modulated by a combination of treatment response, environmental sensitivity, and attack burden — not by any single factor

This supports the case for **personalised clinical monitoring** rather than population-level protocols.

---

## Interpretability Highlights

The project applies a full suite of model-agnostic interpretability methods:

- **PDP**  confirms that reduced medication predicts highest risk (75.8%) and no-treatment phase shows higher predicted risk than under-treatment, consistent with confounding-by-indication
- **ALE** shows a consistent positive effect of daily air quality across all seasons and headache subtypes, with no clear differential pattern by subgroup
- **ICE per patient** demonstrates substantial heterogeneity in individual responses, particularly for medication
- **SHAP** provides both global importance ranking and per-patient explanation, making the model auditable at individual level

---

## Project Structure

```
.
├── data/
│   ├── raw/                        # Original dataset (Kaggle)
│   └── processed/                  # Engineered features, splits, predictions
│
├── models/                         # Trained models, calibrators, preprocessor
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_calibration.ipynb
│   ├── 06_tuning.ipynb
│   ├── 07_model_experiments.ipynb   # Attempted model improvements — no significant gain over baseline; 
                                     # final model unchanged
│   ├── 08_threshold_optimization.ipynb
│   ├── 09_subgroup_fairness.ipynb  # R kernel
│   └── 10_interpretability.ipynb   # R kernel
│
├── reports/
│   ├── figures/                    # All plots (PNG, 300 dpi)
│   └── tables/                     # All output tables (CSV)
│
├── src/
│   ├── data_prep.py
│   ├── features.py
│   ├── modeling.py
│   ├── calibration.py
│   ├── evaluation.py
│   ├── thresholding.py
│   ├── visualization.py
│   ├── visualization.R
│   ├── subgroup_fairness.R
│   ├── interpretability.R
│   └── config.py
│
├── docs/
│   ├── data_dictionary.md       # Clinical variable definitions and coding
│   ├── CLINICAL_SUMMARY.md      # Key findings for clinical audiences
│   └── PROJECT_STORY.md         # Analytical decisions and lessons learned
│
├── requirements.txt
├── LICENSE.txt
└── README.md
```

---

## Technical Notes

**Why logistic regression over tree-based models?**  
Gradient Boosting and XGBoost showed lower ROC-AUC and worse calibration in validation. More importantly, logistic regression produces directly interpretable coefficients and probability outputs that are clinically meaningful — a key requirement for any decision-support tool in a medical context.

**Why isotonic calibration?**  
Platt scaling consistently worsened Brier and ECE across all models. Isotonic calibration improved reliability of predicted probabilities without degrading discrimination.

**Why a mixed Python/R pipeline?**  
Python handles preprocessing, modelling, and SHAP (via scikit-learn and KernelExplainer). R handles the statistical modelling layer — robust clustered standard errors, OR estimation, forest plots, and advanced interpretability (PDP, ALE, ICE) — where the R ecosystem is more mature and audit-ready for a clinical context.

---

## Data Source

Hebaqueen. *Medicine and Environment Impact on Migraine*. Kaggle.  
[kaggle.com/datasets/hebaqueen/medicine-and-environment-impact-on-migraine](https://www.kaggle.com/datasets/hebaqueen/medicine-and-environment-impact-on-migraine)  

133 patients · 4,152 longitudinal records · 10 original variables · study period 1997–2000

---

## Limitations

- Small sample (n = 133 patients): some subgroup analyses lack statistical power
- Calibration slope (≈0.80) indicates residual underestimation of high probabilities
- Fairness analysis limited by near-absence of negative cases in some subgroups
- Dataset does not include hormonal, genetic, or lifestyle covariates relevant to migraine
- Causal inference is not possible from observational data

---

## Contact

Patricia C. Torrell  
Clinical Data Analyst — open to opportunities in pharmaceutical data analytics, clinical research, and medical affairs

---

*MIT License*
