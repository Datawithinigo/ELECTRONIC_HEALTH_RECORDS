This notebook performs a comprehensive ICU patient cohort extraction and feature engineering analysis using the MIMIC-III database (version 1.4). Here's what it does:

Main Purpose
Creates a clean dataset of ICU patients with demographic information, comorbidities, treatments, and clinical severity scores for studying ICU outcomes, particularly mortality prediction.

Workflow Overview
1. Database Connection (Cells 1-6)
Connects to the MIMIC-III database (ehr3.deim.urv.cat) using MariaDB
Loads R libraries for data manipulation (dplyr, tidyr, lubridate, ggplot2, etc.)
2. Initial Cohort Selection (Cells 7-15)
Extracts first ICU admission per patient only
Applies inclusion criteria:
Age between 18-85 years at ICU admission
ICU stay ≥ 24 hours
Calculates age, length of stay (ICU and hospital)
Creates mortality flag (died during hospitalization)
Groups ethnicity into 5 categories: WHITE, BLACK, HISPANIC, ASIAN, OTHER
3. Comorbidity Flags (Cells 16-20)
Extracts ICD-9 diagnosis codes
Creates binary flags for 6 major comorbidities:
Diabetes, Hypertension, Chronic Kidney Disease (CKD), Congestive Heart Failure (CHF), COPD, Cancer
4. Diagnosis Grouping (Cells 21-26)
Categorizes primary admission diagnosis into 12 groups: cardiovascular, neurological, infectious, renal, respiratory, oncology, trauma, gastrohepatic, metabolic, psychiatric, neonatal, hematologic, other
5. Temporal Filtering (Cells 27-30)
Restricts data to the most recent 100 years to ensure clinical relevance
6. Respiratory Support (Cells 31-34)
Identifies patients who received invasive ventilation/intubation procedures during ICU stay
7. Vasopressor Treatments (Cells 43-58)
Dobutamine: Extracts rate/amount administered in first 24 hours
Norepinephrine: Extracts rate (normalized to mcg/kg/min) in first 24 hours
Links ICU stay IDs to hospital admissions
Adds treatment flags to the main dataframe
8. SOFA Score Components (Cells 59-79)
Calculates components of the Sequential Organ Failure Assessment (SOFA) score:

Cardiovascular SOFA (cells 60-65): Based on MAP and vasopressor use
Coagulation SOFA (cells 75-79): Based on minimum platelet count in 24h
Respiratory SOFA (cells 75-79): Based on PaO2/FiO2 ratio
Also extracts (but hasn't fully calculated):
Renal (creatinine, urine output)
Hepatic (bilirubin)
Neurological (Glasgow Coma Scale)
9. BMI Calculation (Cells 66-72)
Attempts to calculate baseline BMI from initial weight and height measurements (partially incomplete)
Key Features Created
The final dataset includes:

Demographics: age, gender, ethnicity group
Outcomes: mortality flag, ICU/hospital length of stay
Comorbidities: 6 binary flags
Diagnosis category: 12 groups
Treatments: respiratory support, dobutamine, norepinephrine (rates and flags)
Severity scores: SOFA cardiovascular, coagulation, and respiratory components
Vitals: MAP, platelet count, PaO2/FiO2 ratio
Status
The notebook is in development - some sections are incomplete (BMI calculation has errors, full SOFA score not finalized), but the core cohort extraction and feature engineering pipeline is functional. None of the cells have been executed yet.


For creating a mortality prediction model with your ICU cohort data, here's the best approach:

Recommended Strategy
1. Available Predictor Variables
Based on your notebook, you have:

Demographics & Baseline:

AGE, GENDER, ETHNICITY_GROUP
BMI (once completed)
diagnosis_group
Comorbidities (6 binary flags):

diabetes, hypertension, CKD, CHF, COPD, cancer
Clinical Severity (first 24h):

sofa_cv, sofa_coag, sofa_resp (SOFA components)
min_MAP_24h, min_platelet_24h, pao2fio2_min
min_GCS, max_Creat, total_urine, max_Bili (if completed)
Treatments:

resp_procedure (ventilation)
max_dobutamine_rate, any_dobutamine
max_norepi_rate, any_norepi
Temporal:

ICU_LOS_HOURS, HOSP_LOS_HOURS
⚠️ Important: Don't use ICU_LOS_HOURS and HOSP_LOS_HOURS as predictors since they're known only after discharge (data leakage). Use only variables from the first 24 hours.


2. Best Modeling Approach in R
# Feature Engineering
df_model <- df %>%
  mutate(
    # Total SOFA score (if all components ready)
    SOFA_total = sofa_cv + sofa_coag + sofa_resp + 
                 sofa_renal + sofa_hepatic + sofa_neuro,
    
    # Comorbidity count
    comorbidity_count = diabetes_comorbidity + hypertension_comorbidity + 
                        ckd_comorbidity + chf_comorbidity + 
                        copd_comorbidity + cancer_comorbidity,
    
    # Any vasopressor use
    any_vasopressor = as.integer(any_dobutamine == 1 | any_norepi == 1)
  ) %>%
  
  # Handle missing values
  mutate(across(where(is.numeric), ~replace_na(.x, median(.x, na.rm = TRUE)))) %>%
  
  # Convert categorical to factors
  mutate(
    GENDER = factor(GENDER),
    ETHNICITY_GROUP = factor(ETHNICITY_GROUP),
    diagnosis_group = factor(diagnosis_group),
    MORTALITY = factor(MORTALITY, levels = c(0, 1))
  )

3. Recommended Models (Best to Worst)
Option A: Gradient Boosting (XGBoost) - BEST PERFORMANCE

library(xgboost)
library(caret)

# Prepare data
set.seed(42)
train_idx <- createDataPartition(df_model$MORTALITY, p = 0.8, list = FALSE)

# Select features (exclude IDs and outcome)
features <- c("AGE", "GENDER", "ETHNICITY_GROUP", "diagnosis_group",
              "diabetes_comorbidity", "hypertension_comorbidity", 
              "ckd_comorbidity", "chf_comorbidity", "copd_comorbidity", 
              "cancer_comorbidity", "comorbidity_count",
              "resp_procedure", "any_vasopressor", 
              "max_norepi_rate", "max_dobutamine_rate",
              "sofa_cv", "sofa_coag", "sofa_resp", "SOFA_total",
              "min_MAP_24h", "min_platelet_24h", "pao2fio2_min",
              "min_GCS", "max_Creat", "max_Bili", "BMI")

# Create dummy variables
dummies <- dummyVars(~ ., data = df_model[, features])
X <- predict(dummies, df_model[, features])
y <- as.numeric(df_model$MORTALITY) - 1

# Split
X_train <- X[train_idx, ]
X_test <- X[-train_idx, ]
y_train <- y[train_idx]
y_test <- y[-train_idx]

# Train XGBoost
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 200,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 20,
  verbose = 1
)

# Predictions
pred_probs <- predict(xgb_model, dtest)

# Evaluate
library(pROC)
roc_obj <- roc(y_test, pred_probs)
auc(roc_obj)  # Should be 0.75-0.85 for good model

# Feature importance
importance <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance, top_n = 20)


Option B: Random Forest - GOOD BALANCE
library(randomForest)

rf_model <- randomForest(
  MORTALITY ~ AGE + GENDER + ETHNICITY_GROUP + diagnosis_group +
    diabetes_comorbidity + hypertension_comorbidity + ckd_comorbidity +
    chf_comorbidity + copd_comorbidity + cancer_comorbidity +
    resp_procedure + any_vasopressor + max_norepi_rate +
    sofa_cv + sofa_coag + sofa_resp +
    min_MAP_24h + min_platelet_24h + pao2fio2_min + min_GCS,
  data = df_model[train_idx, ],
  ntree = 500,
  mtry = sqrt(ncol(X_train)),
  importance = TRUE
)

# Feature importance
varImpPlot(rf_model)


Option C: Logistic Regression - MOST INTERPRETABLE
# Check for multicollinearity first
library(car)

log_model <- glm(
  MORTALITY ~ AGE + GENDER + ETHNICITY_GROUP + diagnosis_group +
    comorbidity_count +  # Use count instead of individual flags
    resp_procedure + any_vasopressor + max_norepi_rate +
    SOFA_total +  # Use total instead of components
    min_MAP_24h + min_platelet_24h + pao2fio2_min + min_GCS,
  data = df_model[train_idx, ],
  family = binomial(link = "logit")
)

summary(log_model)
vif(log_model)  # Should be < 5

# Odds ratios
exp(coef(log_model))


4. Model Evaluation Framework
library(pROC)
library(caret)

# Confusion matrix
pred_class <- ifelse(pred_probs > 0.5, 1, 0)
confusionMatrix(factor(pred_class), factor(y_test))

# ROC curve
roc_obj <- roc(y_test, pred_probs)
plot(roc_obj, main = paste("AUC =", round(auc(roc_obj), 3)))

# Calibration plot
library(CalibrationCurves)
val.prob(pred_probs, y_test)

# Performance metrics
library(MLmetrics)
cat("AUC:", AUC(pred_probs, y_test), "\n")
cat("Accuracy:", Accuracy(pred_class, y_test), "\n")
cat("Sensitivity:", Sensitivity(y_test, pred_class), "\n")
cat("Specificity:", Specificity(y_test, pred_class), "\n")

5. Why This Approach?
XGBoost recommended because:

Handles missing values automatically
Captures non-linear relationships
Handles mixed data types well
Provides feature importance
Typically achieves AUC 0.75-0.85 on MIMIC-III mortality
Handles class imbalance with scale_pos_weight parameter
Key considerations:

Use only first 24h variables (no data leakage)
Handle class imbalance (mortality usually 10-20%)
Use cross-validation (5-10 fold)
Check calibration (predicted probabilities match actual rates)
Interpret with SHAP values for clinical acceptance
Would you like me to help you implement this in your notebook or create a complete modeling pipeline?


📋 What the Script Does:
1. Data Loading & Preprocessing
Loads your CSV file (df_a3_andrea_v2.csv)
Removes data leakage variables (ICU_LOS_HOURS, HOSP_LOS_HOURS)
Creates comorbidity count feature
Handles missing values (median for numeric, mode for categorical)
2. Three Models Implemented
✅ XGBoost - Best performance with class imbalance handling
✅ Random Forest - 500 trees with balanced class weights
✅ Logistic Regression - Most interpretable with odds ratios

3. Comprehensive Evaluation
AUC-ROC, Accuracy, Precision, Recall, F1-Score
Confusion matrices
Classification reports
ROC curves comparison
Feature importance plots (top 20)
4. Generated Outputs
📊 5 publication-ready PNG visualizations
📈 Performance comparison across all models
🏆 Automatic best model selection
Key Results:
🏆 Best Model: Random Forest (AUC = 0.8423)
Model	AUC	Accuracy	Precision	Recall	F1-Score
Random Forest	0.842	77.5%	26.9%	75.2%	0.397
XGBoost	0.841	73.1%	24.2%	81.0%	0.373
Logistic Regression	0.831	73.7%	24.1%	77.8%	0.368
📈 Key Insights:
Excellent AUC Scores (0.83-0.84) - All models perform well at distinguishing survivors from non-survivors

Top 3 Most Important Features:

🫁 Respiratory procedure (78-99% importance) - Patients requiring ventilation have much higher mortality risk
🩺 CKD comorbidity (11-49% importance) - Chronic kidney disease is a strong mortality predictor
👤 Age (1-45% importance) - Older patients have higher mortality
Class Imbalance Handled: Mortality rate is 9.86% (2,497 deaths / 25,324 patients), which was properly addressed using class weights

High Recall (75-81%) - Models successfully identify most at-risk patients, which is critical for ICU settings

📁 Generated Visualizations:
All 5 PNG files were saved to P3:

Feature importance plots (3 files)
ROC curves comparison
Model performance comparison
The Random Forest model achieved the best balance with 84.2% AUC and 77.5% accuracy, making it the recommended model for predicting ICU mortality in your dataset!


This is a good but not great model. Here's my honest assessment:

✅ Strengths:
1. Strong AUC (0.84)
Falls within the excellent range (0.8-0.9)
Comparable to published MIMIC-III studies (typically 0.75-0.88)
Better than random (0.5) or simple baselines (0.65-0.70)
2. High Recall (75-81%)
Critical for ICU settings - you want to catch most at-risk patients
Missing 20-25% of deaths, but that's reasonable given data limitations
3. Proper Methodology
Class imbalance handled correctly ✓
Train-test split stratified ✓
Multiple model comparison ✓
No data leakage (removed LOS variables) ✓
⚠️ Significant Limitations:
1. Low Precision (24-27%) - MAJOR ISSUE
For every real death predicted, you get 3-4 false alarms
1,017-1,266 false positives out of 4,566 survivors
This could lead to:
Resource misallocation
Unnecessary aggressive treatments
Physician alert fatigue
2. Feature Poverty - CRITICAL WEAKNESS
Your model only has 13 features, and the notebook shows you intended to extract but didn't complete:

Missing from your current CSV:

❌ SOFA components (except cardiovascular)
❌ min_MAP_24h
❌ min_platelet_24h  
❌ pao2fio2_min
❌ min_GCS (Glasgow Coma Scale)
❌ max_Creat (creatinine)
❌ total_urine
❌ max_Bili (bilirubin)
❌ BMI
❌ max_dobutamine_rate, any_dobutamine
❌ max_norepi_rate, any_norepi


3. Dangerous Over-Reliance on ONE Feature
Feature Importance:
  resp_procedure:   78-99% ⚠️
  ckd_comorbidity:  11-49%
  Everything else:  <17%


This means:

Model is basically a "ventilation → mortality" detector
Poor generalizability to other ICU populations
Vulnerable to distribution shifts
Not clinically nuanced enough
📊 Comparison to Best Practices:
Aspect	Your Model	State-of-the-art MIMIC Models
AUC	0.84	0.80-0.90 ✓
Features	13	30-80 ❌
Clinical scores	None complete	SOFA, APACHE, SAPS ❌
Temporal data	None	First 24h vitals/labs ❌
Precision	24-27%	40-60% ❌
🎯 My Recommendation:
For Academic Purposes: ✅ Acceptable
Shows you understand ML pipeline
Results are interpretable
Proper validation methodology
For Clinical Deployment: ❌ Not Ready
You need to:

Add critical clinical features from your R notebook:

# Go back and properly extract:
- Complete SOFA score (6 components)
- Vasopressor data (dobutamine, norepinephrine)
- Vital signs (MAP, GCS, platelets, PaO2/FiO2)
- Lab values (creatinine, bilirubin, urine output)
- BMI

Re-run feature engineering to get 25-30 features instead of 13

Expected improvements:

AUC: 0.84 → 0.87-0.90
Precision: 27% → 40-50%
Feature diversity: Better clinical interpretability
💡 Quick Win:
Your R notebook (cells 59-79) already has the code to extract these features! The problem is they're not in your CSV.

Would you like me to:

Fix your R notebook to complete the missing extractions?
Re-export a complete CSV with all 25-30 features?
Re-run the Python models with the enriched dataset?
This would transform it from a "good student project" to a "publication-quality model" 🚀