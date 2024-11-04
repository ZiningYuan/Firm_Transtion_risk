#### What we hope to do
#### On CTVaR coveraged global firms (3000+, North America,Europe, Asia, etc.)
#### Illustrate on American firms

library(sqldf)
library(stringr)
library(readr)
library(forecast)
library(dplyr)
library(lubridate)
library(ggplot2)
library(stargazer)
library(gridExtra)
library(reshape2)
library(tseries)
library(urca)
library(zoo)
library(tidyverse)
library(leaps)
library(lmtest)
library(readxl)


#### Step 1 Pure Replication
#### Meaning: for period 2005 - 2017
#### Check summary statistics of the variables and panel regression on selected variables
#### Could show results separately for American firms (to compare) and global 

data <- read.csv("Data/full_range_data/BK_data_p_SPX.csv")
colnames(data)
selected_columns_emi <- data[, c("GHG_INT_DIRECT",  "GHG_INT_INDIRECT","GHG_INT_1",
                                 "GHG_INT_2_loc",   "GHG_INT_3_up",    "LOG_GHG_ABS_1", "LOG_GHG_ABS_2_loc", "LOG_GHG_ABS_2_mkt" ,  "LOG_GHG_ABS_3_up_tot" ,   
                                 "GHG_DIRECT_IR",   "GHG_INDIRECT_IR", "LOG_GHG_ABS_3_down_tot","GHG_INT_3_down_tot", "LOG_GHG_ABS_AR_1",   
                                 "LOG_GHG_ABS_AR_2_loc","GHG_ABS_2_mkt",   "GHG_INT_2_mkt",   "GHG_ABS_AR_2_mkt",  "GR_GHG_ABS_AR_1", "GR_GHG_ABS_1",   "GR_GHG_ABS_AR_2_loc"   ,"GR_GHG_ABS_AR_2_mkt", "GR_GHG_ABS_2_loc","GR_GHG_ABS_2_mkt","GR_GHG_ABS_3_up_tot","GR_GHG_ABS_3_down_tot" )]

selected_columns_controls <- data[, c("trt1m","LOGSIZE", "BM", "LEVERAGE", "LEVERAGE_c",
                                      "MOM_CUM",  "MOM_MA", "INVESTA","ROE","HHI", "LOGPPE",  
                                      "Beta",  "VOLAT_abs", "VOLAT",   "SALESGR", "EPSGR")]

stargazer(as.data.frame(selected_columns_controls),
          type="text", 
          title  = "Descriptive statsitics", 
          summary.stat = c( "mean","median", "sd"),
          digits = 2)
### can make plots if needed


#### Panel Regression
library(plm)
data$year_month <- as.factor(data$`year.month`)
pdata <- pdata.frame(data, index = c("gvkey", "year_month"))

control_vars <- c("LOGSIZE_lag", "BM_lag", "LEVERAGE_lag","MOM_CUM_lag", "INVESTA_lag","ROE_lag","HHI", "LOGPPE_lag",  
                  "Beta_lag",  "VOLAT_abs_lag", "SALESGR", "EPSGR")
explanatory_vars <- c("LOG_GHG_ABS_1", "LOG_GHG_ABS_2_loc", "LOG_GHG_ABS_3_up_tot","GR_GHG_ABS_1",   "GR_GHG_ABS_2_loc", "GR_GHG_ABS_3_up_tot","GHG_INT_1 " ,"GHG_INT_2_loc ", "GHG_INT_3_up ")






cs_models <- list()
for (expl_var in explanatory_vars) {
  formula <- as.formula(
    paste("trt1m", "~", paste(c(expl_var,control_vars,"factor(year_month)"), collapse = " + "))
  )
  
  cs_pooled_model <- plm(formula,data = pdata, model ="pooling")
  # Store model in the list
  cs_models[[expl_var]] <-cs_pooled_model
  print(summary(cs_pooled_model))
}




cs_models_ind <- list()
for (expl_var in explanatory_vars) {
  formula <- as.formula(
    paste("trt1m", "~", paste(c(expl_var,control_vars,"factor(year_month) + factor(gind)"), collapse = " + "))
  )
  
  cs_pooled_model_ind <- plm(formula,data = pdata, model ="pooling")
  # Store model in the list
  cs_models_ind[[expl_var]] <-cs_pooled_model_ind
  print(summary(cs_pooled_model_ind))
}




#### Step 2: Replace Emission with CTVar
#### Meaning: we are working on another dataset, for which the variables are defined just like before
#### The only thing changed is the time period
JP_data <- read.csv("Data/full_range_data/JP_data_p_SPX.csv")
selected_columns_emi <-JP_data[, c("GHG_INT_DIRECT",  "GHG_INT_INDIRECT","GHG_INT_1",
                                 "GHG_INT_2_loc",   "GHG_INT_3_up",    "LOG_GHG_ABS_1", "LOG_GHG_ABS_2_loc", "LOG_GHG_ABS_2_mkt" ,  "LOG_GHG_ABS_3_up_tot" ,   
                                 "GHG_DIRECT_IR",   "GHG_INDIRECT_IR", "LOG_GHG_ABS_3_down_tot","GHG_INT_3_down_tot", "LOG_GHG_ABS_AR_1",   
                                 "LOG_GHG_ABS_AR_2_loc","GHG_ABS_2_mkt",   "GHG_INT_2_mkt",   "GHG_ABS_AR_2_mkt",  "GR_GHG_ABS_AR_1", "GR_GHG_ABS_1",   "GR_GHG_ABS_AR_2_loc"   ,"GR_GHG_ABS_AR_2_mkt", "GR_GHG_ABS_2_loc","GR_GHG_ABS_2_mkt","GR_GHG_ABS_3_up_tot","GR_GHG_ABS_3_down_tot" )]

selected_columns_controls <- JP_data[, c("trt1m","LOGSIZE", "BM", "LEVERAGE", "LEVERAGE_c",
                                      "MOM_CUM",  "MOM_MA", "INVESTA","ROE","HHI", "LOGPPE",  
                                      "Beta",  "VOLAT_abs", "VOLAT",   "SALESGR", "EPSGR")]

stargazer(as.data.frame(selected_columns_controls),
          type="text", 
          title  = "Descriptive statsitics", 
          summary.stat = c( "mean","median", "sd"),
          digits = 2)

#### bring in CTVaR data
#### 
CTVaR <- read.csv("Data/CTVaR_FIRM_CHA.csv")
CTVaR <- CTVaR %>%
  select(-starts_with("..."))
CTVaR  <-CTVaR   %>% select(-starts_with("X."))
CTVaR <- CTVaR   %>% select(-"G")

JP_data$month <- substr(JP_data$`year.month`, 6, 7)
JP_merged <- merge(JP_data, CTVaR[, c("Ticker", "ct_var", "ev_var")], 
                        by.x = "tic", by.y = "Ticker", all.x = TRUE)

##### HERE, chose the relevant time that you want to run regression on
### option 1: panel with month fixed effect --- code is similar to the presvious one


### option 2: for each month, run a cross-sectional regression 
JP_merged_X <-  JP_merged %>% filter(`year.month` == "2023-01") ## can create many other datasets for different months too
formula <- as.formula(
  paste("trt1m", "~", paste(c("ct_var",control_vars,"factor(gind)"), collapse = " + "))
)

cs_ctvar <- lm(formula,data = JP_merged_X)
summary(cs_ctvar)
##### can try: put into a for loop to check for every month
### option 3: make an average for monthly returns / calculate annual return (ignore if NA)

### put emission back for comparison
cs_emission <- list()
for (expl_var in explanatory_vars) {
  formula <- as.formula(
    paste("trt1m", "~", paste(c(expl_var,control_vars,"factor(year_month) "), collapse = " + "))
  )
  #### if using global data, add factor(country) or factor(region)
  cs_emission_model <- lm(formula,data = JP_merged_X)
  # Store model in the list
  cs_emission [[expl_var]] <-cs_emission_model
  print(summary(cs_emission_model))
}








#### STEP 3: PUT CTVaR and Emission together 
### again the three options above
cs_emission_ctvar <- list()
for (expl_var in explanatory_vars) {
  formula <- as.formula(
    paste("trt1m", "~", paste(c("ct_var", expl_var,control_vars,"factor(year_month) "), collapse = " + "))
  )
  #### if using global data, add factor(country) or factor(region)
  cs_emission_ctvar_model <- lm(formula,data = JP_merged_X)
  # Store model in the list
  cs_emission_ctvar[[expl_var]] <-cs_emission_ctvar_model
  print(summary(cs_emission_ctvar_model))
}