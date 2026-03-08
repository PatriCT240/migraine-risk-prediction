# scripts/interpretability.R
# -----------------------------
# Interpretability utilities for Migraine Attack Risk Prediction.
# Includes permutation importance (patient‑grouped), PDP for one‑hot
# categorical variables, ALE curves (numeric and stratified), ICE curves,
# and sensitivity analysis.

suppressPackageStartupMessages({
  library(yardstick)
  library(dplyr)
  library(data.table)
})


# ---------------------------------------
# 1. Permutation Importance (PR‑AUC grouped by patient)
# ---------------------------------------
compute_permutation_importance <- function(X, y, ids, model, pred_fun) {
  # Compute permutation importance grouped by patient using PR-AUC.
  #
  # Parameters
  # ----------
  # X : data.frame
  #     Feature matrix.
  # y : numeric
  #     Binary labels (0/1).
  # ids : vector
  #     Patient IDs for grouped permutation.
  # model : model object
  #     Trained model.
  # pred_fun : function
  #     Prediction function(model, X) → probabilities.
  #
  # Returns
  # -------
  # data.frame
  #     Variable importance scores.

  features   <- colnames(X)
  y_factor   <- factor(y, levels = c(0, 1))
  baseline_p <- pred_fun(model, X)
  baseline_prauc <- yardstick::pr_auc_vec(truth = y_factor, estimate = baseline_p)

  block_permute_feature <- function(X, groups, feature_name) {
    X_perm <- X
    uniq   <- unique(groups)
    perm   <- sample(uniq, length(uniq), replace = FALSE)
    map    <- setNames(perm, uniq)

    for (g_dst in uniq) {
      g_src    <- map[[as.character(g_dst)]]
      src_rows <- which(groups == g_src)
      dst_rows <- which(groups == g_dst)
      sampled  <- sample(src_rows, length(dst_rows), replace = TRUE)
      X_perm[dst_rows, feature_name] <- X[sampled, feature_name]
    }
    X_perm
  }

  importances <- numeric(length(features))

  for (j in seq_along(features)) {
    fj      <- features[j]
    X_perm  <- block_permute_feature(X, ids, fj)
    p_perm  <- pred_fun(model, X_perm)
    pr_perm <- yardstick::pr_auc_vec(truth = y_factor, estimate = p_perm)
    importances[j] <- baseline_prauc - pr_perm
  }

  data.frame(Variable = features, Importance = importances)
}


# ---------------------------------------
# 2. PDP for one‑hot encoded categorical variables
# ---------------------------------------
compute_pdp_onehot <- function(model, X, prefix_levels, pred_fun) {
  # Compute PDP values for one-hot encoded categorical variables.
  #
  # Parameters
  # ----------
  # model : model object
  #     Trained model.
  # X : data.frame
  #     Feature matrix.
  # prefix_levels : character vector
  #     One-hot encoded columns for a categorical variable.
  # pred_fun : function
  #     Prediction function(model, X) → probabilities.
  #
  # Returns
  # -------
  # data.frame
  #     PDP values per category level.

  base <- X

  values <- sapply(prefix_levels, function(lvl) {
    X_tmp <- base
    X_tmp[, prefix_levels] <- 0
    X_tmp[[lvl]] <- 1
    preds <- pred_fun(model, X_tmp)
    mean(preds, na.rm = TRUE)
  })

  data.frame(
    level = prefix_levels,
    pdp   = values,
    stringsAsFactors = FALSE
  )
}


run_all_pdp <- function(model, X, pred_fun) {
  # Compute PDPs for all categorical groups defined by one-hot patterns.
  #
  # Parameters
  # ----------
  # model : model object
  # X : data.frame
  # pred_fun : function
  #
  # Returns
  # -------
  # list
  #     List of PDP results with names, filenames, and data frames.

  get_levels <- function(pattern) grep(pattern, colnames(X), value = TRUE)

  features <- list(
    list(name = "PDP Medication", filename = "pdp_medication.png", pattern = "^cat__medication_"),
    list(name = "PDP Phase",      filename = "pdp_phase.png",      pattern = "^cat__phase_"),
    list(name = "PDP Hatype",     filename = "pdp_hatype.png",     pattern = "^cat__hatype_"),
    list(name = "PDP Season",     filename = "pdp_season.png",     pattern = "^cat__study_season_")
  )

  results <- lapply(features, function(f) {

    levels <- get_levels(f$pattern)
    if (length(levels) == 0)
      stop(paste("No columns found for:", f$name))

    df <- compute_pdp_onehot(model, X, levels, pred_fun)

    list(
      name     = f$name,
      filename = f$filename,
      df       = df
    )
  })

  results
}


# ---------------------------------------
# 3. ALE for numeric variables (with optional stratification)
# ---------------------------------------
compute_ale <- function(X, feature, model, pred_fun,
                        bins = 20,
                        group_levels = NULL,
                        group_name   = NULL) {
  # Compute ALE curves for numeric variables, optionally stratified by one-hot groups.
  #
  # Parameters
  # ----------
  # X : data.frame
  # feature : character
  #     Numeric feature name.
  # model : model object
  # pred_fun : function
  # bins : integer
  #     Number of ALE bins.
  # group_levels : character vector or NULL
  #     One-hot encoded group columns.
  # group_name : character or NULL
  #     Name of the grouping variable.
  #
  # Returns
  # -------
  # data.frame
  #     ALE curve (possibly stratified).

  # Standard ALE (no groups)
  if (is.null(group_levels) || is.null(group_name)) {

    x   <- X[[feature]]
    qs  <- quantile(x, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE)
    mids <- (qs[-1] + qs[-length(qs)]) / 2
    deltas <- numeric(bins)

    for (b in seq_len(bins)) {
      idx <- which(x >= qs[b] & x < qs[b + 1])
      if (length(idx) == 0) { deltas[b] <- 0; next }
      X_low  <- X; X_low[idx, feature]  <- qs[b]
      X_high <- X; X_high[idx, feature] <- qs[b + 1]
      deltas[b] <- mean(pred_fun(model, X_high) - pred_fun(model, X_low))
    }

    ale <- cumsum(deltas)
    ale <- ale - mean(ale, na.rm = TRUE)

    return(data.frame(mid = mids, ale = ale))
  }

  # Stratified ALE (one‑hot groups)
  curves <- list()

  for (lvl in group_levels) {

    idx <- which(X[[lvl]] == 1)
    if (length(idx) < 10) next

    x   <- X[idx, feature]
    qs  <- quantile(x, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE)
    mids <- (qs[-1] + qs[-length(qs)]) / 2
    deltas <- numeric(bins)

    for (b in seq_len(bins)) {
      bin_idx <- idx[which(x >= qs[b] & x < qs[b + 1])]
      if (length(bin_idx) == 0) { deltas[b] <- 0; next }
      X_low  <- X; X_low[bin_idx, feature]  <- qs[b]
      X_high <- X; X_high[bin_idx, feature] <- qs[b + 1]
      deltas[b] <- mean(pred_fun(model, X_high) - pred_fun(model, X_low))
    }

    ale <- cumsum(deltas)
    ale <- ale - mean(ale, na.rm = TRUE)

    df_lvl <- data.frame(mid = mids, ale = ale)
    df_lvl[[group_name]] <- lvl

    curves[[lvl]] <- df_lvl
  }

  do.call(rbind, curves)
}


run_all_ale <- function(model, X, pred_fun, bins = 20) {
  # Compute ALE curves for all configured numeric features and stratifications.
  #
  # Parameters
  # ----------
  # model : model object
  # X : data.frame
  # pred_fun : function
  # bins : integer
  #
  # Returns
  # -------
  # list
  #     ALE results with names, filenames, and data frames.

  numeric_features <- c("num__airq", "num__airq_prev_mean")

  configs <- list(
    list(suffix = "",          group_levels = NULL,                                      group_name = NULL),
    list(suffix = "_season",   group_levels = grep("^cat__study_season_", colnames(X), value = TRUE), group_name = "season"),
    list(suffix = "_hatype",   group_levels = grep("^cat__hatype_",       colnames(X), value = TRUE), group_name = "hatype"),
    list(suffix = "_phase",    group_levels = grep("^cat__phase_",        colnames(X), value = TRUE), group_name = "phase")
  )

  results <- list()

  for (feat in numeric_features) {
    for (cfg in configs) {

      df <- compute_ale(
        X            = X,
        feature      = feat,
        model        = model,
        pred_fun     = pred_fun,
        bins         = bins,
        group_levels = cfg$group_levels,
        group_name   = cfg$group_name
      )

      results[[length(results) + 1]] <- list(
        name     = paste("ALE", feat,
                         if (!is.null(cfg$group_name)) paste("by", cfg$group_name) else ""),
        filename = paste0("ale_", feat, cfg$suffix, ".png"),
        df       = df
      )
    }
  }

  results
}


# ---------------------------------------
# 4. ICE curves by patient
# ---------------------------------------
compute_ice_by_id <- function(X, ids, feature, model, pred_fun, bins = 25) {
  # Compute ICE curves for each patient, for numeric or one-hot categorical features.
  #
  # Parameters
  # ----------
  # X : data.frame
  # ids : vector
  #     Patient IDs.
  # feature : character vector
  #     One-hot encoded levels (categorical) or single numeric feature.
  # model : model object
  # pred_fun : function
  # bins : integer
  #
  # Returns
  # -------
  # data.frame
  #     ICE curves stacked by patient.

  # Categorical (one‑hot)
  if (length(feature) > 1) {
    levels <- feature

    return(do.call(rbind, lapply(unique(ids), function(pid) {

      X_id <- X[pid, , drop = FALSE]

      do.call(rbind, lapply(levels, function(lv) {
        X_mod <- X_id
        for (c in levels) X_mod[[c]] <- 0
        X_mod[[lv]] <- 1

        data.frame(
          id  = pid,
          x   = lv,
          ice = pred_fun(model, X_mod)
        )
      }))
    })))
  }

  # Numeric
  grid <- seq(
    min(X[[feature]], na.rm = TRUE),
    max(X[[feature]], na.rm = TRUE),
    length.out = bins
  )

  do.call(rbind, lapply(unique(ids), function(pid) {

    X_id <- X[pid, , drop = FALSE]

    ice_vals <- sapply(grid, function(v) {
      X_mod <- X_id
      X_mod[[feature]] <- v
      pred_fun(model, X_mod)
    })

    data.frame(
      id  = pid,
      x   = grid,
      ice = ice_vals
    )
  }))
}


# ---------------------------------------
# 5. Sensitivity analysis
# ---------------------------------------
compute_sensitivity <- function(X, ids, features, model, pred_fun) {
  # Compute sensitivity (range of predicted probabilities) per patient
  # when varying a feature or one-hot encoded group.
  #
  # Parameters
  # ----------
  # X : data.frame
  # ids : vector
  # features : character vector
  #     One-hot encoded levels or single numeric feature.
  # model : model object
  # pred_fun : function
  #
  # Returns
  # -------
  # data.frame
  #     Sensitivity per patient.

  unique_ids <- unique(ids)

  sens_list <- lapply(unique_ids, function(pid) {
    idx  <- which(ids == pid)
    X_id <- X[idx, , drop = FALSE]

    is_cat <- length(features) > 1

    if (is_cat) {
      curves <- sapply(seq_along(features), function(i) {
        X_mod <- X_id
        for (f in features) X_mod[[f]] <- 0
        X_mod[[features[i]]] <- 1
        mean(pred_fun(model, X_mod), na.rm = TRUE)
      })
    } else {
      col  <- features[1]
      vals <- X[[col]]
      grid <- seq(min(vals, na.rm = TRUE), max(vals, na.rm = TRUE), length.out = 25)

      curves <- sapply(grid, function(v) {
        X_mod <- X_id
        X_mod[[col]] <- v
        mean(pred_fun(model, X_mod), na.rm = TRUE)
      })
    }

    data.frame(id = pid, sensitivity = max(curves) - min(curves))
  })

  do.call(rbind, sens_list)
}
