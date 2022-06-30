library(tidymodels)
tidymodels_prefer()

data("ad_data", package = "modeldata")
alz <- ad_data

# eda ---------------------------------------------------------------------

alz %>% skimr::skim()
alz %>% count(Class)
alz %>% count(Genotype)
#alz %>% select(-Genotype, -Class) %>% correlate()

# initial split -----------------------------------------------------------

set.seed(123)
alz_split <- initial_split(alz, strata = Class, prop = .9)
alz_train <- training(alz_split)
alz_test <- testing(alz_split)


# GLM ---------------------------------------------------------------------

lr_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

lr_fit <- lr_spec %>%
  fit(Class ~ tau + VEGF,
      data = alz)

alz_new <-
  tibble(tau = c(5, 6, 7),
         VEGF = c(15, 15, 15),
         Class = c("Control", "Control", "Impaired")) %>%
  mutate(Class = factor(Class, levels = c("Impaired", "Control")))

lr_fit %>%
  predict(new_data = alz_new)

lr_fit %>%
  predict(new_data = alz_new) %>%
  bind_cols(alz_new) %>%
  accuracy(truth = Class, estimate = .pred_class)

# YT: decision tree -------------------------------------------------------

tree_spec <- decision_tree() %>%
  set_engine("rpart") %>%
  set_mode("classification")

tree_fit <- tree_spec %>%
  fit(Class ~ tau + VEGF, data = alz)

tree_fit %>%
  predict(new_data = alz_test) %>%
  bind_cols(alz_test) %>%
  accuracy(truth = Class, estimate = .pred_class)

## other prediction types -> augment

lr_fit %>%
  predict(new_data = alz_new, type = "prob")

lr_fit %>%
  augment(new_data = alz_new)

# lr_fit %>%
#   augment(new_data = alz_new) %>%
#   roc_auc(.pred_Impaired, truth = Class)
#
# lr_fit %>%
#   augment(new_data = alz_new) %>%
#   roc_curve(.pred_Impaired, truth = Class) %>%
#   autoplot()

# cross-validation --------------------------------------------------------

set.seed(100)
alz_folds <-
  vfold_cv(alz_train, v = 5, strata = Class)
alz_folds

lr_spec %>%
  fit_resamples(
    Class ~ tau + VEGF,
    resamples = alz_folds
  ) %>%
  collect_metrics()

# cv with tree

tree_spec %>%
  fit_resamples(
    Class ~ tau + VEGF,
    resamples = alz_folds
  ) %>%
  collect_metrics()



# feature engineering -----------------------------------------------------

lr_pen_spec <- logistic_reg(penalty = 0.1) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

alz_rec <-
  recipe(Class ~ ., data = alz_train) %>%
  step_other(Genotype, threshold = .03) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric())

alz_wflow <- workflow() %>%
  add_model(lr_pen_spec) %>%
  add_recipe(alz_rec)

alz_wflow %>%
  fit_resamples(alz_folds) %>%
  collect_metrics()


# E: knn

# same recipe

knn_spec <- nearest_neighbor() %>%
  set_engine("kknn") %>%
  set_mode("classification")

alz_knn_wflow <- workflow() %>%
  add_model(knn_spec) %>%
  add_recipe(alz_rec)

alz_knn_wflow %>%
  fit_resamples(alz_folds) %>%
  collect_metrics()

# tuning  -----------------------------------------------------------------

lr_pen_spec <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")
alz_rec <-
  recipe(Class ~ ., data = alz_train) %>%
  step_other(Genotype, threshold = tune("genotype_threshold")) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric())
alz_wfl <-
  workflow() %>%
  add_model(lr_pen_spec) %>%
  add_recipe(alz_rec)

set.seed(2)
grid <-
  alz_wfl %>%
  extract_parameter_set_dials() %>%
  grid_latin_hypercube(size = 25)

set.seed(9)
lr_pen_res <-
  alz_wfl %>%
  tune_grid(resamples = alz_folds, grid = grid)

show_best(lr_pen_res, metric = "roc_auc")

# now with knn

knn_spec <-
  nearest_neighbor(neighbors = tune(), weight_func = tune()) %>%
  set_engine("kknn") %>%
  set_mode("classification")

alz_knn_wflow <- workflow() %>%
  add_model(knn_spec) %>%
  add_recipe(alz_rec)

set.seed(9)
knn_res <-
  alz_knn_wflow %>%
  tune_grid(resamples = alz_folds, grid = 25)

show_best(knn_res, metric = "roc_auc")


# then compare
show_best(lr_pen_res, metric = "roc_auc")
show_best(knn_res, metric = "roc_auc")



# finalize ----------------------------------------------------------------

best_auc <- select_best(lr_pen_res, metric = "roc_auc")
best_auc

alz_wflow <- alz_wflow %>%
  finalize_workflow(best_auc)

test_res <- alz_wflow %>%
  last_fit(split = alz_split)
test_res

# compare test to resampling results
collect_metrics(test_res)
show_best(lr_pen_res, metric = "roc_auc", n = 1)
