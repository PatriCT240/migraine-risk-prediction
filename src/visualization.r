# scripts/visualization.R
# -----------------------------
# Visualization utilities for Migraine Attack Risk Prediction.
# Includes forest plots, medication boxplots, PDP, ALE, ICE and sensitivity plots.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(scales)
  library(tidyverse)
})


# ---------------------------------------
# 1. General forest plot
# ---------------------------------------
plot_forest <- function(or_table, term_filter = NULL, label_map = NULL,
                        title = "Forest Plot", output_path = NULL) {
  # Create a forest plot for odds ratios with optional filtering and relabeling.
  #
  # Parameters
  # ----------
  # or_table : data.frame
  #     Table with OR, CI_low, CI_high, term.
  # term_filter : character or formula or NULL
  #     Filter terms to include.
  # label_map : named list or NULL
  #     Mapping from term → label.
  # title : character
  # output_path : character or NULL
  #
  # Returns
  # -------
  # ggplot object

  df <- or_table

  if (!is.null(term_filter)) {
    if (is.character(term_filter)) {
      df <- df %>% filter(term %in% term_filter)
    }
    if (inherits(term_filter, "formula")) {
      df <- df %>% filter(grepl(as.character(term_filter)[2], term))
    }
  }

  if (!is.null(label_map)) {
    df <- df %>% mutate(label = dplyr::recode(term, !!!label_map))
  } else {
    df <- df %>% mutate(label = term)
  }

  df <- df %>% arrange(OR) %>% mutate(label = factor(label, levels = unique(label)))

  p <- ggplot(df, aes(x = OR, y = label)) +
    geom_point(size = 3, color = "blue", na.rm = TRUE) +
    geom_errorbarh(aes(xmin = CI_low, xmax = CI_high),
                   height = 0.2, color = "gray40", na.rm = TRUE) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "red") +
    scale_x_log10() +
    labs(title = title, x = "Odds Ratio (log scale)", y = "") +
    theme_minimal(base_size = 14)

  if (!is.null(output_path)) {
    ggsave(output_path, p, width = 10, height = 7, dpi = 300)
  }

  p
}


# ---------------------------------------
# 2. Medication boxplot
# ---------------------------------------
plot_medication_attacks <- function(df_id, summary_tbl, output_path = NULL) {
  # Boxplot of attack proportion by medication type.
  #
  # Parameters
  # ----------
  # df_id : data.frame
  # summary_tbl : data.frame
  # output_path : character or NULL
  #
  # Returns
  # -------
  # ggplot object

  summary_tbl <- summary_tbl %>%
    mutate(label = paste0(round(mean_attacks * 100, 1), "%"))

  p <- ggplot(df_id, aes(x = medication, y = mean_target, fill = medication)) +
    geom_boxplot(alpha = 0.6, outlier.color = "red") +
    geom_jitter(width = 0.2, alpha = 0.4) +
    geom_text(data = summary_tbl,
              aes(x = medication, y = mean_attacks, label = label),
              vjust = -0.8, size = 5, fontface = "bold", inherit.aes = FALSE) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(title = "Attack Proportion by Medication Type",
         x = "Medication Type",
         y = "Attack Proportion (mean per id)") +
    theme_minimal(base_size = 14) +
    scale_fill_manual(values = c("gray70", "skyblue", "orange"))

  if (!is.null(output_path)) {
    ggsave(output_path, p, width = 10, height = 7, dpi = 300)
  }

  p
}


# ---------------------------------------
# 3. Clinical subtype model comparison
# ---------------------------------------
plot_clinical_subtype_comparison <- function(comparison_table, output_path = NULL) {
  # Forest plot comparing clinical subtype models.
  #
  # Parameters
  # ----------
  # comparison_table : data.frame
  # output_path : character or NULL
  #
  # Returns
  # -------
  # ggplot object

  df_plot <- comparison_table %>%
    mutate(term = recode(term, "hatypeAura" = "Aura", "hatypeMixed" = "Mixed"))

  p <- ggplot(df_plot, aes(x = OR, y = term)) +
    geom_point(size = 3, color = "blue") +
    geom_errorbarh(aes(xmin = CI_low, xmax = CI_high),
                   height = 0.2, color = "gray40") +
    geom_vline(xintercept = 1, linetype = "dashed", color = "red") +
    scale_x_log10() +
    facet_wrap(~Model) +
    labs(title = "Clinical Subtype Comparison — Odds Ratios",
         x = "Odds Ratio (log scale)", y = "") +
    theme_minimal(base_size = 14)

  if (!is.null(output_path)) {
    ggsave(output_path, p, width = 10, height = 7, dpi = 300)
  }

  p
}


# ---------------------------------------
# 4. Permutation importance plot
# ---------------------------------------
plot_permutation_importance <- function(imp_df) {
  # Barplot of permutation importance (Δ PR-AUC).
  #
  # Parameters
  # ----------
  # imp_df : data.frame
  #
  # Returns
  # -------
  # ggplot object

  ggplot(imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
    geom_col(fill = "#2E86C1") +
    coord_flip() +
    labs(
      x = "Variables",
      y = "Importance (Delta PR-AUC)",
      title = "Permutation Importance (PR-AUC grouped by patient)"
    ) +
    theme_minimal() +
    theme(
      panel.grid = element_blank(),
      axis.text.y = element_text(size = 9),
      axis.title.x = element_text(size = 11),
      plot.title = element_text(size = 13, face = "bold")
    )
}


# ---------------------------------------
# 5. PDP plot
# ---------------------------------------
plot_pdp_onehot <- function(df, title) {
  # Plot PDP for one-hot encoded categorical variable.
  #
  # Parameters
  # ----------
  # df : data.frame
  # title : character
  #
  # Returns
  # -------
  # ggplot object

  ggplot(df, aes(x = reorder(level, pdp), y = pdp, fill = level)) +
    geom_col(show.legend = FALSE) +
    geom_text(aes(label = percent(pdp, accuracy = 0.1)),
              hjust = 1.0, color = "black", fontface = "bold", size = 6) +
    coord_flip() +
    labs(title = title, x = "Category", y = "Predicted risk") +
    theme_minimal() +
    theme(
      panel.grid = element_blank(),
      axis.text.y = element_text(size = 9),
      axis.title.x = element_text(size = 11),
      plot.title = element_text(size = 13, face = "bold")
    )
}

build_all_pdp_plots <- function(results) {
  # Build all PDP plots from PDP results list.
  lapply(results, function(res) plot_pdp_onehot(res$df, res$name))
}


# ---------------------------------------
# 6. ALE plot
# ---------------------------------------
plot_ale_unified <- function(df, title) {
  # Plot ALE curve, optionally stratified by season/hatype/phase.
  #
  # Parameters
  # ----------
  # df : data.frame
  # title : character
  #
  # Returns
  # -------
  # ggplot object

  group_cols <- intersect(c("season", "hatype", "phase"), colnames(df))

  if (length(group_cols) == 0) {
    ggplot(df, aes(mid, ale)) +
      geom_line(color = "#2E86C1", linewidth = 1.2) +
      labs(title = title, x = "Feature value", y = "ALE effect") +
      theme_minimal()
  } else {
    group_col <- group_cols[1]

    ggplot(df, aes(mid, ale, color = .data[[group_col]])) +
      geom_line(linewidth = 1.2) +
      labs(title = title, x = "Feature value", y = "ALE effect", color = group_col) +
      theme_minimal()
  }
}

build_all_ale_plots <- function(results) {
  # Build all ALE plots from ALE results list.
  lapply(results, function(res) plot_ale_unified(res$df, res$name))
}


# ---------------------------------------
# 7. ICE plot
# ---------------------------------------
plot_ice <- function(df, feature, highlight_id = NULL) {
  # Plot ICE curves for each patient, with optional highlight.
  #
  # Parameters
  # ----------
  # df : data.frame
  # feature : character
  # highlight_id : numeric or NULL
  #
  # Returns
  # -------
  # ggplot object

  df$highlight <- if (is.null(highlight_id)) FALSE else df$id == highlight_id

  ggplot(df, aes(x, ice, group = id, color = highlight)) +
    geom_line(alpha = 0.3) +
    stat_summary(aes(group = 1), fun = mean, geom = "line", color = "red") +
    scale_color_manual(values = c("gray70", "red"), guide = "none") +
    labs(title = paste("ICE for", feature, "by patient"),
         x = feature, y = "Predicted risk") +
    theme_minimal()
}


# ---------------------------------------
# 8. Sensitivity scatter plot
# ---------------------------------------
plot_sensitivity_facets <- function(df_sens) {
  # Faceted scatter plot of sensitivity vs prior attacks.
  #
  # Parameters
  # ----------
  # df_sens : data.frame
  #
  # Returns
  # -------
  # ggplot object

  df_plot <- df_sens %>%
    select(id, prev_attacks_real, sens_medication, sens_fase, sens_airq, sens_airq_prev_mean) %>%
    pivot_longer(
      cols      = starts_with("sens_"),
      names_to  = "feature",
      values_to = "sensitivity"
    ) %>%
    mutate(feature = factor(recode(feature,
      sens_medication     = "Treatment Response",
      sens_fase           = "Treatment Phase",
      sens_airq           = "Air Quality (daily)",
      sens_airq_prev_mean = "Air Quality (mean)"
    ), levels = c("Treatment Response", "Treatment Phase",
                  "Air Quality (daily)", "Air Quality (mean)")))

  ggplot(df_plot, aes(x = prev_attacks_real, y = sensitivity)) +
    geom_point(aes(color = feature), alpha = 0.55, size = 2) +
    geom_smooth(method = "loess", se = TRUE, color = "grey25",
                linewidth = 0.9, fill = "grey85") +
    facet_wrap(~feature, scales = "free_y", ncol = 2) +
    scale_color_manual(values = c(
      "Treatment Response"  = "#C0392B",
      "Treatment Phase"     = "#2471A3",
      "Air Quality (daily)" = "#1E8449",
      "Air Quality (mean)"  = "#D4AC0D"
    )) +
    scale_y_continuous(labels = number_format(accuracy = 0.01)) +
    labs(
      title    = "Individual Sensitivity to Clinical and Environmental Factors",
      subtitle = "Sensitivity = range of predicted headache probability · n = 133 patients",
      x        = "Number of Prior Migraine Attacks",
      y        = "Sensitivity (Δ predicted probability)",
      caption  = "Logistic regression · LOESS 95% CI · Migraine dataset (1997–2000)"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title       = element_text(face = "bold", size = 13),
      plot.subtitle    = element_text(color = "grey40", size = 9.5),
      plot.caption     = element_text(color = "grey55", size = 8, hjust = 0),
      strip.text       = element_text(face = "bold", size = 10.5),
      strip.background = element_rect(fill = "grey95", color = NA),
      legend.position  = "none",
      panel.grid.minor = element_blank(),
      panel.spacing    = unit(1.2, "lines")
    )
}


# ---------------------------------------
# 9. Sensitivity bubble plot
# ---------------------------------------
plot_sensitivity_bubble <- function(df_sens) {
  # Bubble plot of sensitivity to treatment vs prior attacks.
  #
  # Parameters
  # ----------
  # df_sens : data.frame
  #
  # Returns
  # -------
  # ggplot object

  ggplot(df_sens, aes(
      x     = prev_attacks_real,
      y     = sens_medication,
      color = sens_airq,
      size  = sens_fase
    )) +
    geom_point(alpha = 0.80) +
    geom_smooth(
      aes(x = prev_attacks_real, y = sens_medication),
      method = "loess", se = TRUE,
      color = "grey25", linewidth = 0.9, fill = "grey85",
      inherit.aes = FALSE
    ) +
    geom_text(
      data = df_sens %>% filter(sens_medication > quantile(sens_medication, 0.90, na.rm = TRUE)),
      aes(label = paste0("ID ", id)),
      size = 3, hjust = -0.2, color = "grey30", fontface = "italic"
    ) +
    scale_color_gradientn(
      colours = c("#2471A3", "#A9CCE3", "#F1948A", "#C0392B"),
      name    = "Sensitivity\nto air quality",
      labels  = number_format(accuracy = 0.01)
    ) +
    scale_size_continuous(
      range  = c(2, 9),
      name   = "Sensitivity\nto phase",
      labels = number_format(accuracy = 0.01)
    ) +
    labs(
      title    = "Clinical Heterogeneity in Treatment Response",
      subtitle = "Each point = one patient · Color = air quality sensitivity · Size = phase sensitivity",
      x        = "Number of Prior Migraine Attacks",
      y        = "Sensitivity to Treatment (Δ predicted probability)",
      caption  = "Logistic regression · LOESS 95% CI · Migraine dataset (1997–2000)"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title       = element_text(face = "bold", size = 13),
      plot.subtitle    = element_text(color = "grey40", size = 9.5),
      plot.caption     = element_text(color = "grey55", size = 8, hjust = 0),
      legend.title     = element_text(size = 9, face = "bold"),
      panel.grid.minor = element_blank()
    )
}
