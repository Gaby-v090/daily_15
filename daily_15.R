library(tidymodels)

data(penguins)

set.seed(123)

penguins_split <- initial_split(penguins, prop = 0.7)

penguins_train <- training(penguins_split)
penguins_test <- testing(penguins_split)

penguins_folds <- vfold_cv(penguins_train, v = 10)

