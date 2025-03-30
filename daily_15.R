library(tidymodels)
library(tidyverse)
library(workflowsets)


data(penguins)

set.seed(123)

penguins_split <- initial_split(penguins, prop = 0.7)

penguins_train <- training(penguins_split)
penguins_test <- testing(penguins_split)

penguins_folds <- vfold_cv(penguins_train, v = 10)

log_reg_model <-
  multinom_reg(mode = "classification") %>%
  set_engine("nnet")

rand_forest_model <-
  rand_forest(mode = 'classification') %>%
  set_engine ("ranger")

penguins_recipe <-
  recipe(species ~ ., data = penguins_train) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

penguins_workflows <-
  workflow_set (
    preproc = list(penguins_recipe),
    models = list(logistic = log_reg_model,
                  random_forest = rand_forest_model)
  )

penguins_results <-
  penguins_workflows %>%
  workflow_map("fit_resamples",
               resamples = penguins_folds,
               metrics = metric_set(accuracy),
               verbose = TRUE)

penguins_results %>%
  rank_results(rank_metric = "accuracy")

#I think the random forest model will likely preform better overall because of its ability to handle nonlinearities and interaction, but logistic regression is simpler and more interpretable
