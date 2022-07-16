#rm(list = ls())
#To clean the data
library(tidyverse)
#To make the data where we use mtry to tune the data 
library(tidymodels)
#date and time
library(lubridate)
#vector data
library(skimr)
#to plot the data in a graph
library(ggplot2)
#missmap
library(Amelia)

aus <- read.csv("weatherAUS.csv")

#### Summary of the data

glimpse(aus)

summary(aus)

missmap(aus, main = "Missing Data")

sapply(aus,function(x) sum(is.na(x)))

aus <- aus %>% select(-Sunshine, -Evaporation)

#we dropped the column Sunshine and Evaporation as both the values contained a lot of NA values.




#we dropped all the NA values in RainToday and RainTomorrow
aus %>% 
  count(RainToday, RainTomorrow) %>% 
  drop_na()

# gave us the X-Square value of RainToday and Tomorrow
chisq.test(table(aus$RainToday, aus$RainTomorrow))

# only 16 columns are selected according to the relevance of data
#Date,Sunshine, Evaporation, Cloud3pm, Cloud9am, Wind speed 9am/3pm
aus_df <- aus %>% 
  select(Location,
         RainTomorrow,
         WindGustDir,
         WindDir9am,
         WindDir3pm,
         RainToday,
         WindGustSpeed,
         MinTemp,
         MaxTemp,
         Rainfall,
         Humidity9am,
         Humidity3pm,
         Pressure9am,
         Pressure3pm,
         Temp3pm
  ) %>% 
  mutate(Month = factor(lubridate::month(aus$Date, label = TRUE), ordered = FALSE)) %>% # Extracted only months from the Date Variable
  drop_na() %>% 
  mutate_if(is.character, as.factor) %>% 
  mutate(RainTomorrow = relevel(RainTomorrow, ref = "Yes")) 

sapply(aus_df,function(x) sum(is.na(x)))

# kept the RainTomorrow data in leveled form where levels are Yes and No
levels(aus_df$RainTomorrow)

# set seed so next time too the group of data chosen remains same
set.seed(123)
#Data reduced to 1000 since Random Forest has more computation if the data is big
aus_df <- aus_df[0:1000,]

aus_split <- initial_split(aus_df, strata = RainTomorrow)
aus_train <- training(aus_split)
aus_test <- testing(aus_split)
aus_split # Shows Training/Testing/Total Observations
#749/251/1000
#75% and 25%

#parameter tuning for corss validation
aus_cv <- vfold_cv(aus_train, strata = RainTomorrow)
aus_cv
#randomly split the data into 10 folds

####Random Forest

#####model type is Rand_forest
rf_spec <- rand_forest(
  mtry = tune(), ### We tune this hyperparameter via Cross Validation
  trees = 500,   ### We grow 500 random trees, That's going to be used for prediction
  min_n = tune() ### We tune this hyperparameter via Cross Validation
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

#the number of randomly selected variables to be considered at each split in the trees - mtry

#setting the classification mode and ranger as our engine since Random Forest takes that
rf_spec

aus_wf <- workflow() %>%
  add_formula(RainTomorrow ~ .)

aus_wf

tune_wf <- aus_wf %>% 
  add_model(rf_spec)

tune_wf

doParallel::registerDoParallel()

rf_rs_tune <- tune_grid(
  object = tune_wf,
  resamples = aus_cv,
  grid = 25,
  control = control_resamples(save_pred = TRUE)
)

### Results of Tuning
rf_rs_tune
### Check Accuracy and AUC after tuning
rf_rs_tune %>% 
  collect_metrics()

rf_rs_tune %>% 
  collect_predictions() %>% 
  group_by(id) %>% 
  roc_curve(RainTomorrow, .pred_Yes) %>%
  ggplot(aes(1 - specificity, sensitivity, color = id)) +
  geom_abline(lty = 2, color = "gray80", size = 1.5) +
  geom_path(show.legend = TRUE, alpha = 0.6, size = 1.2) +
  coord_equal()

best_acu <- select_best(rf_rs_tune, "accuracy")
best_acu

best_auc <- select_best(rf_rs_tune, "roc_auc")
best_auc

rf_final <- finalize_model(
  rf_spec,
  best_acu
)

rf_final #### This is the Best Model from our CV

final_res <- aus_wf %>%
  add_model(spec = rf_final) %>%
  last_fit(aus_split)

final_res %>%
  collect_metrics()

Confusion_matrix_rf <- final_res %>%
  collect_predictions() %>% 
  conf_mat(RainTomorrow, .pred_class)
Confusion_matrix_rf
final_res %>%
  collect_predictions() %>% 
  sensitivity(RainTomorrow, .pred_class)
final_res %>%
  collect_predictions() %>% 
  specificity(RainTomorrow, .pred_class)
final_res %>%
  collect_predictions() %>% 
  accuracy(RainTomorrow, .pred_class)
final_res %>%
  collect_predictions() %>% 
  recall(RainTomorrow, .pred_class)

#___________________________________________________________________________________________________________

set.seed(123)

#### XGB Model
xgb_spec <- boost_tree(
  mtry = tune(), ### We tune this hyperparameter via Cross Validation
  trees = 500,   ### We grow 500 random trees, That's going to be used for prediction
  min_n = tune() ### We tune this hyperparameter via Cross Validation
) %>%
  set_mode("classification") %>%
  set_engine("xgboost")

tune_wf_xgb <- aus_wf %>% 
  add_model(xgb_spec)

tune_wf_xgb

xgb_rs_tune <- tune_grid(
  object = tune_wf_xgb,
  resamples = aus_cv,
  grid = 25,
  control = control_resamples(save_pred = TRUE)
)

xgb_rs_tune
### Check Accuracy and AUC after tuning
xgb_rs_tune %>% 
  collect_metrics()

best_acu_xgb <- select_best(xgb_rs_tune, "accuracy")
best_acu_xgb

best_auc_xgb <- select_best(xgb_rs_tune, "roc_auc")
best_auc_xgb

xgb_final <- finalize_model(
  xgb_spec,
  best_acu_xgb
)

xgb_final #### This is the Best Model from our CV

final_res_xgb <- aus_wf %>%
  add_model(spec = xgb_final) %>%
  last_fit(aus_split)

final_res_xgb %>%
  collect_metrics()

Confusion_matrix_xgb <- final_res_xgb %>%
  collect_predictions() %>% 
  conf_mat(RainTomorrow, .pred_class)

Confusion_matrix_xgb

final_res_xgb %>%
  collect_predictions() %>% 
  specificity(RainTomorrow, .pred_class)

final_res_xgb %>%
  collect_predictions() %>% 
  sensitivity(RainTomorrow, .pred_class)

final_res_xgb %>%
  collect_predictions() %>% 
  recall(RainTomorrow, .pred_class)

