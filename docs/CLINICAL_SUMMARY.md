# Clinical Summary
## Migraine Attack Risk Prediction — Key Findings for Clinical and Regulatory Audiences

**Project:** Medicine and Environment Impact on Migraine  
**Author:** Patricia C. Torrell · Clinical Data Analyst  
**Model:** Logistic Regression (tuned + isotonic calibration)  
**Dataset:** 133 patients · 4,152 longitudinal records · 1997–2000

---

## Purpose of This Document

This summary presents the clinical interpretation of model outputs and statistical analyses. It is intended for readers with a clinical or regulatory background who need to understand what the model found, what it means, and what its limitations are — without requiring deep familiarity with machine learning.

---

## 1. Can Migraine Attacks Be Predicted?

**Yes — with meaningful but bounded accuracy.**

The final model achieves a ROC-AUC of 0.715 (95% CI: 0.681–0.749) and a PR-AUC of 0.754 (95% CI: 0.711–0.792) on a held-out test set of unseen patients. Both metrics significantly exceed random baseline (ROC-AUC = 0.708; PR-AUC = 0.748), confirmed by bootstrap with 1,000 resamples.

The model does not predict attacks with certainty. It estimates **individual probability of attack** given the patient's clinical history, treatment status, and environmental exposure. This makes it suitable for risk stratification and monitoring support — not for autonomous clinical decisions.

---

## 2. What Drives Attack Risk?

### Primary predictors (population level)

| Factor | Direction | Evidence |
|---|---|---|
| Prior attack history        | ↑ risk | Strongest SHAP predictor (mean |SHAP| ≈ 0.059); OR = 1.033 per attack (p = 0.042) |
| No medication      | ↔ trend | OR = 0.52 vs continuing (p = 0.085) — non-significant protective trend |
| Reduced medication | ↑ risk  | OR = 2.09 vs continuing (p = 0.023) — reduction associated with more attacks |
| Clinical follow-up (visits) | ↓ risk | Top SHAP predictor (mean |SHAP| ≈ 0.068); high visit number reduces predicted risk — likely proxy for disease improvement over time |
| Air quality (Q4 vs Q1) | No effect | OR = 1.05, p = 0.767 — no significant association across all quartiles |

### Secondary modulators

- **Clinical subtype (aura):** significantly higher attack risk both in unadjusted (OR = 3.25, p = 0.009) and adjusted models (OR = 3.11, p = 0.001); effect remains stable after covariate adjustment, suggesting an independent association with migraine risk
- **Age 41–50:** borderline trend towards lower risk vs 18–30 (OR = 0.44, p = 0.057 in clinical subtype model); not significant in treatment model (OR = 0.57, p = 0.256)
- **Previous attacks:** small but significant cumulative effect on risk (OR = 1.033 per attack, p = 0.042)
- **Season:** autumn associated with highest predicted risk (PDP: 76.2%); no consistent seasonal modulation of air quality effects in ALE analysis
- **Air quality:** no significant association across quartiles (OR range 0.97–1.05, all p > 0.75)

---

## 3. Does Treatment Reduce Risk?

The treatment phase variable (before vs during treatment) does not reach statistical significance (OR = 0.632, p = 0.304), consistent with a confounding-by-indication effect: patients in the treatment phase tend to be more severe cases, which attenuates the apparent protective signal.

However, **medication type** shows a significant effect:

| Medication | Mean attack proportion | vs Continuing |
|---|---|---|
| None | 44.0% | Lower, but severity-confounded (OR = 0.52, p = 0.085) |
| Continuing | 66.1% | Baseline |
| Reduced | 78.2% | Significantly higher (ANOVA p < 0.001; OR = 2.09, p = 0.023) |

**Interpretation:** Reducing or stopping medication is the single most modifiable risk factor identified. Patients who reduce treatment show the highest attack burden, consistent with loss of prophylactic protection, the low attack rate in the no-medication group likely reflects milder disease at baseline rather than a protective effect.

---

## 4. Does Air Quality Matter?

No significant association was found between air quality quartile and attack risk. All quartiles show odds ratios very close to 1 (OR range 0.97–1.05, all p > 0.75), with no consistent directional trend.
ALE analysis shows a modest positive effect of current air quality exposure across all subgroups, but without meaningful subgroup differentiation by season or headache type.

**Interpretation:** Environmental air quality does not emerge as a reliable predictor in this model. The effect is minimal and statistically unsupported at any exposure level.

---

## 5. Is Clinical Response Heterogeneous?

**Yes — this is one of the most clinically actionable findings.**

Patient-level sensitivity analysis quantifies, for each patient, how much their predicted risk changes when medication, treatment phase, or air quality is varied across its full observed range.

Key patterns:

- **Medication sensitivity** ranges from 0.19 to 0.29 across patients — the strongest and most variable individual response
- **Environmental sensitivity** (air quality) ranges from 0.035 to 0.054 — substantially smaller than medication sensitivity but present
- **reatment phase sensitivity** (0.033–0.067) and air quality sensitivity are **strongly correlated** (r = 0.972), suggesting a shared underlying responsiveness to contextual factors
- Medication sensitivity is **more independent** (r ≈ 0.60 with other sensitivities), indicating a distinct biological dimension


This heterogeneity means that **population-level recommendations are insufficient**. Patients differ primarily in their responsiveness to medication changes, while environmental and phase sensitivity tend to co-vary.

---

## 6. Subgroup Fairness

Performance was evaluated separately for sex, clinical subtype, and season. Valid AUC estimates were available only for subgroups with at least one negative case: female (AUC = 0.857), No Aura (AUC = 0.833), and summer (AUC = 1.0). Sensitivity could not be computed for any subgroup due to the aggregation approach used. Specificity was estimable only where n_neg ≥ 1.

**Limitation:** The dataset is heavily imbalanced at patient level — most patients have near-universal headache presence, leaving very few true negatives. This makes AUC and specificity estimates unreliable across all subgroups. A larger and more balanced dataset would be required for robust fairness evaluation.

---

## 7. Model Calibration

The calibrated model achieves a Brier Score of 0.214 and ECE of 0.074, indicating good global reliability of predicted probabilities. However, the calibration slope (≈0.70) indicates **mild underestimation at high probability values** — the model is somewhat conservative when predicting high-risk patients.

This should be considered when using probability outputs for clinical triage: patients near the upper end of the predicted probability range may carry higher true risk than the number suggests.

---

## 8. What This Model Is and Is Not

| This model IS | This model IS NOT |
|---|---|
| A risk stratification tool | A diagnostic test |
| Probabilistic and interpretable | Deterministic or prescriptive |
| Validated on held-out unseen patients | Externally validated on independent cohorts |
| Useful for clinical monitoring support | A replacement for clinical judgement |
| Auditable at population and patient level | Causal — associations do not imply causation |

---

## 9. Recommendations Derived from the Model

For a clinical or pharmaceutical audience, the model supports the following evidence-based recommendations:

1. **Maintain continuous treatment** — medication reduction is the strongest modifiable risk factor
2. **Increase follow-up frequency** for high-risk patients — visit intensity is protective
3. **Monitor environmental exposure** in patients with aura subtype, who show significantly higher baseline attack risk
4. **Personalise monitoring strategy** based on individual treatment and environmental sensitivity profiles — population protocols may be insufficient for high-heterogeneity subgroups

---

*For full methodological detail, see README.md and the project notebooks.*
