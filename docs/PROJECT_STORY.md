# Project Story
## Why I Built This — and What I Learned

---

### The Starting Point

I wanted to build a project that felt real — not a clean academic exercise with a tidy dataset and a clear answer, but something closer to what clinical data analysis actually looks like in practice: messy longitudinal data, clinical context that shapes every modelling decision, and a question that matters.

Migraine is one of the most common and disabling neurological conditions in the world. It affects mostly working-age adults, disproportionately women, and its triggers are poorly understood at the individual level. That made it an ideal target for a predictive modelling project: the data is inherently longitudinal, the clinical heterogeneity is real, and interpretability matters because any output needs to make sense to a clinician.

---

### What I Was Trying to Answer

The main question was straightforward: *can we predict the onset of a migraine attack?*

But as I worked through the data, secondary questions emerged that turned out to be more interesting:

- Does treatment actually reduce risk — or is that confounded by who gets treated?
- Does air quality matter, and if so, for which patients?
- Are all patients equally responsive to treatment and environment?

That last question became the analytical centrepiece of the project.

---

### The Analytical Decisions That Shaped the Project

**Choosing logistic regression over tree-based models**  
Gradient Boosting and XGBoost both showed lower ROC-AUC and worse calibration in validation. But even if they had performed similarly, I would have chosen logistic regression for a clinical context. The outputs need to be interpretable — not just to me, but to anyone reviewing the model in a regulatory or clinical setting. Logistic regression produces coefficients with clinical meaning and probability estimates that can be explained to a physician.

**The confounding-by-indication problem**  
When I first ran the subgroup analysis on treatment phase, the results looked wrong — being in the treatment phase barely reduced predicted risk which attenuates the apparent protective effect. What emerges instead is a clear medication type effect: patients with reduced medication show significantly higher attack burden (OR = 2.09, p = 0.023), while absence of medication is associated with lower attack rates — likely reflecting milder disease at baseline.. I had to think carefully about why. The answer was clinical, not statistical: the patients who receive continuous treatment tend to be the most severe cases. So the treatment group is systematically sicker than the untreated group, which attenuates the apparent protective effect. This is textbook confounding-by-indication, and handling it required robust standard errors clustered at patient level and careful interpretation of the medication variable rather than the phase variable.

**Building a mixed Python/R pipeline**  
The project uses Python for data preparation, modelling, and SHAP, and R for statistical modelling, interpretability (PDP, ALE, ICE), and fairness analysis. This was not an arbitrary choice. The R ecosystem — particularly `geepack`, `clubSandwich`, `iml`, and `DALEX` — provides a more mature and audit-ready toolkit for the kind of robust clinical statistics that a pharmaceutical environment would expect. Running R from within a Jupyter notebook via `reticulate` adds technical complexity, but it reflects a realistic hybrid workflow.

**The sensitivity analysis**  
The most clinically interesting output of the project was not the model itself, but the patient-level sensitivity indices. By systematically perturbing each patient's feature values across their full observed range and measuring the change in predicted probability, I was able to quantify — for each patient — how much their risk responds to treatment changes versus environmental changes. This revealed genuine heterogeneity: some patients are treatment-sensitive, some are environment-sensitive, some are both, some are neither. That is exactly the kind of insight that could support personalised clinical monitoring in practice.

---

### What Went Wrong and What I Learned From It

**Calibration slope**  
After tuning and isotonic calibration, the model's global calibration metrics (Brier, ECE) looked good. But the calibration slope was 0.80 — at the lower boundary of the acceptable range (0.8–1.2).. A secondary Platt recalibration did not fix it. In the end, I documented this as a limitation rather than trying to engineer around it. The lesson: calibration is harder than discrimination, and a slope of 0.80 in a small dataset is not a failure — it is an honest finding that needs to be reported transparently.

**Model experiments**
I dedicated a full notebook to exploring whether the final model could be improved — alternative architectures, feature interactions, ensemble approaches. None produced a meaningful gain over the tuned logistic regression. The decision to keep the simpler model was deliberate: a marginal improvement in AUC does not justify added complexity in a clinical context where interpretability is non-negotiable.

**Air quality**
Air quality, initially hypothesised as a significant environmental predictor, showed no statistically significant association at any quartile (OR range 0.97–1.05, all p > 0.75). The ALE analysis confirmed a modest directional effect but insufficient to support clinical recommendations.

**Fairness analysis**  
The fairness evaluation revealed something uncomfortable: most subgroups (male patients, aura subtype, most seasons) had almost no negative cases in the test set. AUC and specificity were either undefined or unreliable for those groups. Rather than hiding this, I reported it clearly and noted what would be needed — a larger, more balanced dataset — to do this analysis properly. That transparency is, I think, more valuable to a clinical audience than inflated results.

**The `reticulate` path problem**  
A more practical lesson: running Python modules from R via `reticulate` requires careful path management on Windows, especially when modules import from a `src/` package structure. The solution was to add the absolute path to `sys.path` and remove all `from src.` import statements from the module files. It took time to debug, but it deepened my understanding of how Python environments and import resolution work in cross-language pipelines.

---

### What This Project Demonstrates

Looking back, this project demonstrates several things I care about as a clinical data analyst:

- **Clinical reasoning drives analytical decisions** — every modelling choice connects back to a clinical question
- **Interpretability is not optional** — in a medical context, a model that cannot be explained is a model that cannot be trusted
- **Honesty about limitations** — acknowledging what the data cannot answer is as important as reporting what it can
- **End-to-end ownership** — from raw data to reproducible outputs, documented and version-controlled

---

### What I Would Do Differently

With a larger dataset or more time, I would:

- Explore mixed-effects models to handle the longitudinal structure more rigorously
- Include hormonal cycle data for female patients — likely a strong confounder in migraine research
- Conduct external validation on an independent cohort
- Build a simple risk dashboard to make the model outputs accessible to non-technical clinical users

---

### Who This Project Is For

This project is for anyone evaluating my profile for a role in clinical data analysis, pharmacoepidemiology, or healthcare analytics. It is designed to show that I can:

- Work with real clinical data and reason about it medically
- Build rigorous, reproducible analytical pipelines
- Communicate findings to both technical and clinical audiences
- Be honest about uncertainty and limitations

If you have questions about any methodological decision, I am always happy to discuss it.

---

*Patricia C. Torrell · Clinical Data Analyst*
