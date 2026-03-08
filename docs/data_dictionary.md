# Data Dictionary

This document describes the raw variables and derived features used in the migraine study dataset. It includes variable types, definitions, ranges, units, and relevant notes for interpretation and modelling.

---

## 1. Raw Variables

| Variable | Type | Description | Range | Units | Notes |
|---------|------|-------------|--------|--------|--------|
| **rownames** | Integer | Record index | 1–4152 | — | Unique per row; not a clinical identifier |
| **id** | Integer | Unique patient identifier | 1–133 | — | Each patient has multiple longitudinal records |
| **time** | Integer | Temporal index relative to migraine treatment | −? to 23 | days | Negative values: days before treatment start; 0 = treatment start; positive = days after |
| **dos** | Integer | Study day count | 98–1239 | days | Days elapsed since study start (Jan 1 of first year) |
| **hatype** | Categorical | Migraine subtype | Aura, No Aura, Mixed | — | Nominal variable with 3 categories |
| **age** | Integer | Patient age | 18–66 | years | Adult population |
| **airq** | Numeric | Air quality index | 3–73 | AQI | Likely PM2.5 or general pollution index |
| **medication** | Categorical | Medication regimen | continuing, reduced, none | — | Treatment status |
| **headache** | Categorical | Headache occurrence | yes, no | — | Binary target variable |
| **sex** | Categorical | Patient sex | female, male | — | Binary |

---

## 2. Derived Variables

| Variable | Type | Description | Range | Units | Notes |
|---------|------|-------------|--------|--------|--------|
| **phase** | Categorical | Treatment phase indicator | no treatment / treatment | — | Derived from `time` |
| **month_study** | Numeric | Calendar month | 1–12 | month | Computed from `dos` using origin date 1997‑01‑01 |
| **season_study** | Categorical | Season of the year | winter, spring, summer, autumn | — | Month-to-season mapping |
| **days_since_first_visit** | Numeric | Days since patient’s first visit | 0–128 | days | Longitudinal difference per patient |
| **age_band** | Categorical | Age groups | 18–30, 31–40, 41–50, 51–66 | years | Binned
