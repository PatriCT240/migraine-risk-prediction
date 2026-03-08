# scripts/subgroup_fairnes.R
# -----------------------------
# Subgroup fairness utilities for Migraine Attack Risk Prediction.
# Includes treatment‑phase effects, medication comparison, clinical subtype
# effects, environmental effects, and fairness metrics across subgroups.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(brglm2)
  library(clubSandwich)
  library(lmtest)
  library(sandwich)
  library(stringi)
  library(rlang)
  library(pROC)
})


# ---------------------------------------
# 1. Load processed data and threshold
# ---------------------------------------
load_data <- function() {
  # Load processed dataset, test IDs, predicted probabilities and threshold.
  #
  # Returns
  # -------
  # list
  #   df : data.frame with y_prob merged
  #   threshold : numeric best threshold

  df       <- readr::read_csv("../data/processed/final_migraine.csv", show_col_types = FALSE)
  test_ids <- readr::read_csv("../data/processed/test_ids.csv", show_col_types = FALSE)$id
  y_prob   <- readr::read_csv("../reports/tables/y_prob_full.csv", show_col_types = FALSE)$y_prob
  
  df$y_prob <- NA_real_
  df$y_prob[df$id %in% test_ids] <- as.numeric(y_prob)
  
  thr       <- readr::read_csv("../reports/tables/best_threshold.csv", show_col_types = FALSE)
  threshold <- thr$threshold[1]
  list(df = df, threshold = threshold)
}


# ---------------------------------------
# 2. Treatment effect model
# ---------------------------------------
treatment_effect <- function(df, threshold) {
  # Fit logistic regression for treatment‑phase effect with clustered SEs.
  #
  # Parameters
  # ----------
  # df : data.frame
  # threshold : numeric
  #
  # Returns
  # -------
  # list
  #   coef_table : coefficients with clustered SEs
  #   or_table   : odds ratios with CI

  df <- df %>%
    mutate(
      y_prob = as.numeric(y_prob),
      pred = as.integer(y_prob >= threshold),
      id = factor(id),
      treatment_phase = factor(phase, levels = c("No treatment", "Under treatment")),
      sex = factor(sex, levels = c("female", "male")),
      age_band = factor(age_band),
      medication = factor(medication),
      prev_attacks = as.numeric(prev_attacks)
    ) %>%
    filter(is.finite(y_prob)) %>%
    drop_na(target_num, id, treatment_phase, sex, age_band, medication, prev_attacks)

  fit <- glm(
    target_num ~ treatment_phase + sex + age_band + medication + prev_attacks,
    data = df,
    family = binomial("logit"),
    method = "brglmFit"
  )

  V <- vcovCR(fit, cluster = df$id, type = "CR2")
  coefs <- coef(fit)
  se <- sqrt(diag(V))
  z <- coefs / se
  pval <- 2 * pnorm(-abs(z))

  OR <- exp(coefs)
  CIlo <- exp(coefs - 1.96 * se)
  CIhi <- exp(coefs + 1.96 * se)

  coef_table <- data.frame(
    term = names(coefs),
    estimate = coefs,
    se = se,
    z = z,
    p = pval,
    row.names = NULL
  )

  or_table <- data.frame(
    term = names(coefs),
    OR = OR,
    CI_low = CIlo,
    CI_high = CIhi,
    p_value = pval,
    row.names = NULL
  ) %>%
    filter(term != "(Intercept)")

  list(
    coef_table = coef_table,
    or_table = or_table
  )
}


# ---------------------------------------
# 3. Medication comparison
# ---------------------------------------
medication_comparison <- function(df) {
  # Compare headache frequency across medication groups using weighted ANOVA.
  #
  # Parameters
  # ----------
  # df : data.frame
  #
  # Returns
  # -------
  # list
  #   summary_attacks : summary stats per medication group
  #   df_id           : patient‑level dataset
  #   anova           : ANOVA results

  df <- df %>%
    mutate(
      medication = factor(medication, levels = c("none", "reduced", "continuing")),
      headache = ifelse(headache == "yes", 1, 0)
    ) %>%
    drop_na(medication, headache, id)
  df_id <- df %>%
    group_by(id, medication) %>%
    summarise(mean_target = mean(headache, na.rm = TRUE), .groups = "drop")
  summary_attacks <- df_id %>%
    group_by(medication) %>%
    summarise(
      mean_attacks = mean(mean_target, na.rm = TRUE),
      sd_attacks   = sd(mean_target, na.rm = TRUE),
      n            = n()
    )
  anova_model <- aov(mean_target ~ medication, data = df_id)
  anova_out   <- summary(anova_model)
  list(
    summary_attacks = summary_attacks,
    df_id = df_id,
    anova = anova_out
  )
}


# ---------------------------------------
# 4. Clinical subtype effect
# ---------------------------------------
clinical_subtype_effect <- function(df, threshold) {
  # Fit logistic regression for clinical subtype effect with clustered SEs.
  #
  # Parameters
  # ----------
  # df : data.frame
  # threshold : numeric
  #
  # Returns
  # -------
  # list
  #   df       : processed dataset
  #   or_table : odds ratios with CI

  df <- df %>%
    mutate(
      y_prob = as.numeric(y_prob),
      pred = as.integer(y_prob >= threshold),
      id = factor(id),
      hatype = factor(hatype, levels = c("No Aura", "Aura", "Mixed")),
      sex = factor(sex, levels = c("female", "male")),
      age_band = factor(age_band, levels = c("18-30", "31-40", "41-50", "51-66")),
      prev_attacks = as.numeric(prev_attacks)
    ) %>%
    drop_na(pred, id, hatype, sex, age_band, prev_attacks)

  if (nlevels(df$hatype) < 2 ||
      nlevels(df$sex) < 2 ||
      nlevels(df$age_band) < 2) {
    stop("Model cannot run: one or more factors have fewer than 2 levels.")
  }

  fit <- glm(target_num ~ hatype + sex + age_band + prev_attacks,
             data = df,
             family = binomial("logit"))

  Vcl <- vcovCL(fit, cluster = df$id, type = "HC0")
  est <- coef(fit)
  se <- sqrt(diag(Vcl))

  out <- data.frame(
    term = names(est),
    OR = exp(est),
    CI_low = exp(est - 1.96 * se),
    CI_high = exp(est + 1.96 * se),
    p_value = coeftest(fit, vcov. = Vcl)[, 4],
    row.names = NULL
  ) %>%
    filter(term != "(Intercept)")

  list(
    df = df,
    or_table = out
  )
}


# ---------------------------------------
# 5. Clinical subtype model comparison
# ---------------------------------------
clinical_subtype_models_comparison <- function(df) {
  # Compare two logistic models: hatype‑only vs hatype + covariates.
  #
  # Parameters
  # ----------
  # df : data.frame
  #
  # Returns
  # -------
  # list
  #   model1     : OR table for model 1
  #   model2     : OR table for model 2
  #   comparison : subset of comparable hatype terms

  fit1 <- glm(target_num ~ hatype,
              data = df,
              family = binomial("logit"))

  Vcl1 <- vcovCL(fit1, cluster = df$id, type = "HC0")
  ct1  <- coeftest(fit1, vcov. = Vcl1)

  est1 <- coef(fit1)
  se1  <- sqrt(diag(Vcl1))

  out1 <- data.frame(
    Model   = "hatype only",
    term    = names(est1),
    OR      = exp(est1),
    CI_low  = exp(est1 - 1.96 * se1),
    CI_high = exp(est1 + 1.96 * se1),
    p_value = ct1[, 4],
    row.names = NULL
  ) %>%
    filter(term != "(Intercept)")

  fit2 <- glm(target_num ~ hatype + sex + age_band + prev_attacks,
              data = df,
              family = binomial("logit"))

  Vcl2 <- vcovCL(fit2, cluster = df$id, type = "HC0")
  ct2  <- coeftest(fit2, vcov. = Vcl2)

  est2 <- coef(fit2)
  se2  <- sqrt(diag(Vcl2))

  out2 <- data.frame(
    Model   = "hatype + covariates",
    term    = names(est2),
    OR      = exp(est2),
    CI_low  = exp(est2 - 1.96 * se2),
    CI_high = exp(est2 + 1.96 * se2),
    p_value = ct2[, 4],
    row.names = NULL
  ) %>%
    filter(term != "(Intercept)")

  comparison <- bind_rows(out1, out2) %>%
    filter(term %in% c("hatypeAura", "hatypeMixed"))

  list(
    model1     = out1,
    model2     = out2,
    comparison = comparison
  )
}


# ---------------------------------------
# 6. Environmental effect
# ---------------------------------------
environment_effect <- function(df, threshold) {
  # Fit logistic regression for environmental (air quality) effect.
  #
  # Parameters
  # ----------
  # df : data.frame
  # threshold : numeric
  #
  # Returns
  # -------
  # list
  #   df       : processed dataset
  #   or_table : odds ratios with CI

  df <- df %>%
    mutate(
      y_prob = as.numeric(y_prob),
      pred = as.integer(y_prob >= threshold),
      id = factor(id),
      sex = factor(sex, levels = c("female", "male")),
      age_band = factor(age_band, levels = c("18-30", "31-40", "41-50", "51-66")),
      prev_attacks = as.numeric(prev_attacks),
      airq_quartile = cut(
        airq,
        breaks = quantile(airq, probs = seq(0, 1, 0.25), na.rm = TRUE),
        include.lowest = TRUE,
        labels = c("Q1_low", "Q2", "Q3", "Q4_high")
      )
    ) %>%
    drop_na(target_num, id, sex, age_band, prev_attacks, airq_quartile)

  fit <- glm(target_num ~ airq_quartile + sex + age_band + prev_attacks,
             data = df,
             family = binomial("logit"))

  Vcl <- vcovCL(fit, cluster = df$id, type = "HC0")
  est <- coef(fit)
  se <- sqrt(diag(Vcl))

  out <- data.frame(
    term = names(est),
    OR = exp(est),
    CI_low = exp(est - 1.96 * se),
    CI_high = exp(est + 1.96 * se),
    p_value = coeftest(fit, vcov. = Vcl)[, 4],
    row.names = NULL
  ) %>%
    filter(term != "(Intercept)")

  list(
    df = df,
    or_table = out
  )
}


# ---------------------------------------
# 7. Fairness metrics for one subgroup
# ---------------------------------------
fairness_metrics <- function(group_var, df_fairness, threshold_fairness) {
  # Compute AUC, sensitivity and specificity per subgroup.
  #
  # Parameters
  # ----------
  # group_var : character
  # df_fairness : data.frame
  # threshold_fairness : numeric
  #
  # Returns
  # -------
  # data.frame
  #     Fairness metrics per subgroup level.

  df_patient <- df_fairness %>%
    group_by(id, !!sym(group_var)) %>%
    summarise(
      y_true = max(target_num),
      y_pred = mean(y_prob),
      .groups = "drop"
    )

  res <- df_patient %>%
    group_by(!!sym(group_var)) %>%
    summarise(
      n     = n(),
      n_pos = sum(y_true == 1),
      n_neg = sum(y_true == 0),

      AUC = if (n_pos > 0 & n_neg > 0)
        tryCatch(
          as.numeric(pROC::roc(y_true, y_pred, quiet = TRUE)$auc),
          error = function(e) NA_real_
        )
        else NA_real_,

      Sensitivity = if (n_pos > 0)
        sum(y_pred >= threshold_fairness & y_true == 1) / n_pos
        else NA_real_,

      Specificity = if (n_neg > 0)
        sum(y_pred < threshold_fairness & y_true == 0) / n_neg
        else NA_real_,

      valid_auc  = (n_pos > 0 & n_neg > 0),
      valid_sens = (n_pos > 0),
      valid_spec = (n_neg > 0),

      .groups = "drop"
    ) %>%
    mutate(
      variable = group_var,
      level = .data[[group_var]]
    ) %>%
    select(
      variable, level,
      AUC, Sensitivity, Specificity,
      n, n_pos, n_neg,
      valid_auc, valid_sens, valid_spec
    )

  res
}


# ---------------------------------------
# 8. Fairness metrics for all subgroups
# ---------------------------------------
compute_fairness_all <- function(df_fairness, threshold_fairness) {
  # Compute fairness metrics for sex, hatype and season.
  #
  # Returns
  # -------
  # list
  #   fairness_results : all subgroup metrics
  #   fairness_valid   : only valid subgroup metrics

  metrics_sex    <- fairness_metrics("sex", df_fairness, threshold_fairness)
  metrics_hatype <- fairness_metrics("hatype", df_fairness, threshold_fairness)
  metrics_season <- fairness_metrics("study_season", df_fairness, threshold_fairness)

  fairness_results <- bind_rows(
    metrics_sex,
    metrics_hatype,
    metrics_season
  )

  fairness_valid <- fairness_results %>%
    filter(valid_auc & valid_sens & valid_spec)

  list(
    fairness_results = fairness_results,
    fairness_valid = fairness_valid
  )
}


# ---------------------------------------
# 9. Table of fairness OR-like metrics
# ---------------------------------------
create_fairness_or_table <- function(fairness_valid) {
  # Convert fairness metrics into OR‑style table for reporting.
  #
  # Parameters
  # ----------
  # fairness_valid : data.frame
  #
  # Returns
  # -------
  # data.frame
  #     Long-format table with AUC, Sensitivity, Specificity.

  fairness_valid %>%
    pivot_longer(
      cols = c(AUC, Sensitivity, Specificity),
      names_to = "term",
      values_to = "OR"
    ) %>%
    mutate(
      CI_low  = OR,
      CI_high = OR,
      label = paste0(variable, ": ", level, " — ", term)
    ) %>%
    select(variable, level, term, OR, CI_low, CI_high, label)
}
