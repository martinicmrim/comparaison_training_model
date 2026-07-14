#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
  library(emmeans)
  library(broom.mixed)
})

args <- commandArgs(trailingOnly = TRUE)
project_root <- if (length(args) >= 1) normalizePath(args[[1]], mustWork = TRUE) else getwd()
out_dir <- file.path(project_root, "paper_analysis_diagnosis_5fold")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

all_budgets <- c("10", "15", "25", "35", "50", "75", "100", "150", "250", "500", "full")
figure_budgets <- c("10", "25", "50", "100", "250", "full")
reference_method <- "DINOv2"
n_boot <- 2000
bootstrap_seed <- 2026
method_order <- c("ResNet50-Scratch", "ResNet50-ImageNet", "DINOv2", "MAE-SSL-10k", "MAE-SSL-50k", "MAE-SSL-100k")

macro_f1 <- function(y_true, y_pred) {
  classes <- sort(unique(c(y_true, y_pred)))
  scores <- vapply(classes, function(cls) {
    tp <- sum(y_true == cls & y_pred == cls)
    fp <- sum(y_true != cls & y_pred == cls)
    fn <- sum(y_true == cls & y_pred != cls)
    precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    recall <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    if ((precision + recall) == 0) 0 else 2 * precision * recall / (precision + recall)
  }, numeric(1))
  mean(scores)
}

qwk <- function(y_true, y_pred) {
  labels <- sort(unique(c(y_true, y_pred)))
  k <- length(labels)
  if (k < 2) return(NA_real_)
  true_idx <- match(y_true, labels)
  pred_idx <- match(y_pred, labels)
  observed <- table(factor(true_idx, levels = seq_len(k)), factor(pred_idx, levels = seq_len(k)))
  observed <- observed / sum(observed)
  expected <- outer(rowSums(observed), colSums(observed))
  weights <- outer(seq_len(k), seq_len(k), function(i, j) ((i - j)^2) / ((k - 1)^2))
  observed_disagreement <- sum(weights * observed)
  expected_disagreement <- sum(weights * expected)
  if (expected_disagreement == 0) return(NA_real_)
  1 - observed_disagreement / expected_disagreement
}

read_required_csv <- function(path) {
  if (!file.exists(path)) stop("Missing file: ", path)
  read_csv(path, show_col_types = FALSE)
}

standardize_predictions <- function(df, method, default_budget = NULL) {
  required <- c("participant_id", "fold", "y_true", "y_pred")
  missing_cols <- setdiff(required, names(df))
  if (length(missing_cols) > 0) stop(method, " is missing required columns: ", paste(missing_cols, collapse = ", "))
  if (!"budget" %in% names(df)) {
    if (is.null(default_budget)) stop(method, " has no budget column and no default budget was supplied.")
    df$budget <- default_budget
  }
  image_key <- if ("image" %in% names(df)) as.character(df$image) else if ("image_name" %in% names(df)) as.character(df$image_name) else if ("file_id" %in% names(df)) as.character(df$file_id) else as.character(seq_len(nrow(df)))
  tibble(
    method = method,
    fold = as.integer(df$fold),
    budget = as.character(df$budget),
    participant_id = as.character(df$participant_id),
    image_key = image_key,
    y_true = as.integer(df$y_true),
    y_pred = as.integer(df$y_pred)
  ) %>%
    mutate(
      sample_id = paste(fold, participant_id, image_key, sep = "::"),
      correct = as.integer(y_true == y_pred),
      ordinal_abs_error = abs(y_true - y_pred)
    ) %>%
    filter(budget %in% all_budgets)
}

dino_dir <- file.path(project_root, "outputs_dino_diagnosis_multiclass_5fold", "predictions")
dino_files <- sort(list.files(dino_dir, pattern = "^fold[0-9]+_budget.+_predictions\\.csv$", full.names = TRUE))
if (length(dino_files) == 0) stop("No DINO prediction files found in: ", dino_dir)
if (length(dino_files) != 55) warning("Expected 55 DINO files, found ", length(dino_files))
dino <- map_dfr(dino_files, read_required_csv) %>% standardize_predictions("DINOv2")

mae10 <- read_required_csv(file.path(project_root, "outputs_mae10k_diagnosis_multiclass_5fold", "predictions_all_folds.csv")) %>% standardize_predictions("MAE-SSL-10k")
mae50 <- read_required_csv(file.path(project_root, "outputs_mae50k_diagnosis_multiclass_5fold", "predictions_all_folds.csv")) %>% standardize_predictions("MAE-SSL-50k")
mae100 <- read_required_csv(file.path(project_root, "outputs_mae100k_diagnosis_multiclass_5fold", "predictions_all_folds.csv")) %>% standardize_predictions("MAE-SSL-100k")

resnet_root <- file.path(project_root, "outputs_resnet50_scratch_diagnosis_multiclass_5fold")
resnet_scratch <- read_required_csv(file.path(resnet_root, "predictions_all_folds_none.csv")) %>% standardize_predictions("ResNet50-Scratch", default_budget = "full")
resnet_imagenet <- read_required_csv(file.path(resnet_root, "predictions_all_folds_imagenet.csv")) %>% standardize_predictions("ResNet50-ImageNet", default_budget = "full")

predictions <- bind_rows(resnet_scratch, resnet_imagenet, dino, mae10, mae50, mae100) %>%
  distinct(method, fold, budget, sample_id, .keep_all = TRUE) %>%
  mutate(method = factor(method, levels = method_order), budget = factor(budget, levels = all_budgets))
write_csv(predictions, file.path(out_dir, "predictions_harmonized.csv"))

sanity_counts <- predictions %>% count(method, budget, fold, name = "n_predictions")
write_csv(sanity_counts, file.path(out_dir, "sanity_prediction_counts.csv"))

truth_check <- predictions %>% group_by(fold, budget, sample_id) %>% summarise(n_truth = n_distinct(y_true), .groups = "drop") %>% filter(n_truth > 1)
if (nrow(truth_check) > 0) {
  write_csv(truth_check, file.path(out_dir, "warning_inconsistent_y_true.csv"))
  warning("Some matched samples have inconsistent y_true values.")
}

metrics_by_fold <- predictions %>%
  group_by(method, budget, fold) %>%
  summarise(n = n(), macro_f1 = macro_f1(y_true, y_pred), qwk = qwk(y_true, y_pred), .groups = "drop")
write_csv(metrics_by_fold, file.path(out_dir, "metrics_by_fold.csv"))

metrics_summary <- metrics_by_fold %>%
  group_by(method, budget) %>%
  summarise(
    n_folds = n(),
    macro_f1_mean = mean(macro_f1, na.rm = TRUE),
    macro_f1_sd = sd(macro_f1, na.rm = TRUE),
    qwk_mean = mean(qwk, na.rm = TRUE),
    qwk_sd = sd(qwk, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    macro_f1_se = macro_f1_sd / sqrt(n_folds),
    qwk_se = qwk_sd / sqrt(n_folds),
    t_crit = if_else(n_folds > 1, qt(0.975, df = n_folds - 1), NA_real_),
    macro_f1_ci_low = macro_f1_mean - t_crit * macro_f1_se,
    macro_f1_ci_high = macro_f1_mean + t_crit * macro_f1_se,
    qwk_ci_low = qwk_mean - t_crit * qwk_se,
    qwk_ci_high = qwk_mean + t_crit * qwk_se
  ) %>% select(-t_crit)
write_csv(metrics_summary, file.path(out_dir, "metrics_summary.csv"))

metrics_oof <- predictions %>% group_by(method, budget) %>% summarise(n = n(), macro_f1 = macro_f1(y_true, y_pred), qwk = qwk(y_true, y_pred), .groups = "drop")
write_csv(metrics_oof, file.path(out_dir, "metrics_pooled_oof.csv"))

paired_bootstrap <- function(reference_df, comparison_df, metric_fun, n_boot, seed) {
  paired <- inner_join(
    reference_df %>% select(sample_id, y_true_ref = y_true, y_pred_ref = y_pred),
    comparison_df %>% select(sample_id, y_true_cmp = y_true, y_pred_cmp = y_pred),
    by = "sample_id"
  ) %>% filter(y_true_ref == y_true_cmp)
  if (nrow(paired) == 0) return(NULL)
  observed_delta <- metric_fun(paired$y_true_ref, paired$y_pred_cmp) - metric_fun(paired$y_true_ref, paired$y_pred_ref)
  set.seed(seed)
  deltas <- replicate(n_boot, {
    idx <- sample.int(nrow(paired), nrow(paired), replace = TRUE)
    b <- paired[idx, ]
    metric_fun(b$y_true_ref, b$y_pred_cmp) - metric_fun(b$y_true_ref, b$y_pred_ref)
  })
  tibble(
    n_paired = nrow(paired),
    delta = observed_delta,
    ci_low = unname(quantile(deltas, 0.025, na.rm = TRUE)),
    ci_high = unname(quantile(deltas, 0.975, na.rm = TRUE)),
    p_bootstrap = 2 * min(mean(deltas <= 0, na.rm = TRUE), mean(deltas >= 0, na.rm = TRUE))
  )
}

comparison_grid <- predictions %>%
  distinct(method, budget) %>%
  filter(as.character(method) != reference_method) %>%
  semi_join(predictions %>% filter(as.character(method) == reference_method) %>% distinct(budget), by = "budget")

bootstrap_results <- pmap_dfr(comparison_grid, function(method, budget) {
  method_chr <- as.character(method)
  budget_chr <- as.character(budget)
  reference_df <- predictions %>% filter(as.character(.data$method) == reference_method, as.character(.data$budget) == budget_chr)
  comparison_df <- predictions %>% filter(as.character(.data$method) == method_chr, as.character(.data$budget) == budget_chr)
  bind_rows(
    paired_bootstrap(reference_df, comparison_df, macro_f1, n_boot, bootstrap_seed) %>% mutate(metric = "Macro-F1"),
    paired_bootstrap(reference_df, comparison_df, qwk, n_boot, bootstrap_seed + 1) %>% mutate(metric = "QWK")
  ) %>% mutate(reference = reference_method, comparison = method_chr, budget = budget_chr)
}) %>%
  select(reference, comparison, budget, metric, everything()) %>%
  mutate(significant_95ci = !(ci_low <= 0 & ci_high >= 0))
write_csv(bootstrap_results, file.path(out_dir, "paired_bootstrap_vs_dinov2.csv"))

mixed_data <- predictions %>%
  filter(method %in% c("DINOv2", "MAE-SSL-10k", "MAE-SSL-50k", "MAE-SSL-100k")) %>%
  mutate(method = droplevels(method), budget = droplevels(budget), participant_id = factor(participant_id), fold = factor(fold))

correctness_model <- glmer(
  correct ~ method * budget + (1 | participant_id) + (1 | fold),
  data = mixed_data,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
)
saveRDS(correctness_model, file.path(out_dir, "glmm_correctness.rds"))
write_csv(tidy(correctness_model, effects = "fixed", conf.int = TRUE, exponentiate = TRUE), file.path(out_dir, "glmm_correctness_fixed_effects.csv"))
correctness_emmeans <- emmeans(correctness_model, pairwise ~ method | budget, type = "response", adjust = "tukey")
write_csv(as.data.frame(correctness_emmeans$contrasts), file.path(out_dir, "glmm_correctness_pairwise.csv"))

ordinal_error_model <- lmer(ordinal_abs_error ~ method * budget + (1 | participant_id) + (1 | fold), data = mixed_data)
saveRDS(ordinal_error_model, file.path(out_dir, "lmer_ordinal_error.rds"))
write_csv(tidy(ordinal_error_model, effects = "fixed", conf.int = TRUE), file.path(out_dir, "lmer_ordinal_error_fixed_effects.csv"))

curve_data <- metrics_summary %>%
  filter(method %in% c("DINOv2", "MAE-SSL-10k", "MAE-SSL-50k", "MAE-SSL-100k"), as.character(budget) %in% figure_budgets) %>%
  mutate(budget_plot = factor(as.character(budget), levels = figure_budgets))

p_f1 <- ggplot(curve_data, aes(x = budget_plot, y = macro_f1_mean, group = method, linetype = method, shape = method)) +
  geom_line() + geom_point(size = 2.2) +
  geom_errorbar(aes(ymin = macro_f1_ci_low, ymax = macro_f1_ci_high), width = 0.12) +
  labs(x = "Number of labeled training images", y = "Macro-F1", linetype = NULL, shape = NULL) +
  theme_classic(base_size = 11) + theme(legend.position = "bottom")
ggsave(file.path(out_dir, "figure_macro_f1_learning_curve.pdf"), p_f1, width = 6.8, height = 4.2)
ggsave(file.path(out_dir, "figure_macro_f1_learning_curve.png"), p_f1, width = 6.8, height = 4.2, dpi = 300)

p_qwk <- ggplot(curve_data, aes(x = budget_plot, y = qwk_mean, group = method, linetype = method, shape = method)) +
  geom_line() + geom_point(size = 2.2) +
  geom_errorbar(aes(ymin = qwk_ci_low, ymax = qwk_ci_high), width = 0.12) +
  labs(x = "Number of labeled training images", y = "Quadratic weighted kappa", linetype = NULL, shape = NULL) +
  theme_classic(base_size = 11) + theme(legend.position = "bottom")
ggsave(file.path(out_dir, "figure_qwk_learning_curve.pdf"), p_qwk, width = 6.8, height = 4.2)
ggsave(file.path(out_dir, "figure_qwk_learning_curve.png"), p_qwk, width = 6.8, height = 4.2, dpi = 300)

full_data <- metrics_summary %>% filter(as.character(budget) == "full") %>% mutate(method = fct_reorder(method, macro_f1_mean))
p_full <- ggplot(full_data, aes(x = method, y = macro_f1_mean)) +
  geom_point(size = 2.6) +
  geom_errorbar(aes(ymin = macro_f1_ci_low, ymax = macro_f1_ci_high), width = 0.12) +
  coord_flip() + labs(x = NULL, y = "Macro-F1") + theme_classic(base_size = 11)
ggsave(file.path(out_dir, "figure_full_data_comparison.pdf"), p_full, width = 6.4, height = 3.8)
ggsave(file.path(out_dir, "figure_full_data_comparison.png"), p_full, width = 6.4, height = 3.8, dpi = 300)

table_main <- metrics_summary %>%
  filter(as.character(budget) %in% figure_budgets) %>%
  transmute(Method = as.character(method), Budget = as.character(budget), `Macro-F1` = sprintf("%.3f ± %.3f", macro_f1_mean, macro_f1_sd), QWK = sprintf("%.3f ± %.3f", qwk_mean, qwk_sd)) %>%
  arrange(factor(Budget, levels = figure_budgets), Method)
write_csv(table_main, file.path(out_dir, "table_main_mean_sd.csv"))

table_full <- metrics_summary %>%
  filter(as.character(budget) == "full") %>%
  transmute(Method = as.character(method), `Macro-F1` = sprintf("%.3f ± %.3f", macro_f1_mean, macro_f1_sd), QWK = sprintf("%.3f ± %.3f", qwk_mean, qwk_sd)) %>%
  arrange(desc(`Macro-F1`))
write_csv(table_full, file.path(out_dir, "table_full_data.csv"))
write_csv(bootstrap_results %>% arrange(metric, budget, comparison), file.path(out_dir, "table_statistical_comparisons.csv"))

message("Analysis complete. Outputs: ", out_dir)
