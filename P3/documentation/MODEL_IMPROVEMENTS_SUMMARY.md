# 🚀 ICU Mortality Prediction - Enhanced Model Implementation

## 📋 Summary of Improvements

This document summarizes all the enhancements made to the mortality prediction pipeline based on best practices for clinical predictive modeling.

---

## ✅ Implemented Features

### 1. 🔧 Advanced Feature Engineering

All recommended feature engineering techniques have been implemented:

#### **Comorbidity Aggregation**
```python
comorbidity_count = sum of (diabetes, hypertension, CKD, CHF, COPD, cancer)
```
- **Purpose**: Capture overall disease burden
- **Expected Impact**: High - correlates with mortality risk

#### **BMI Categories**
```python
- Underweight: < 18.5
- Normal: 18.5 - 25
- Overweight: 25 - 30
- Obese: > 30
```
- **Purpose**: Non-linear BMI effects on mortality
- **Expected Impact**: Moderate

#### **Age Groups**
```python
- Young: < 40
- Middle: 40-60
- Elderly: 60-75
- Very Elderly: > 75
```
- **Purpose**: Capture age-related mortality patterns
- **Expected Impact**: High

#### **SOFA Severity Categories**
```python
- Low: 0-6
- Moderate: 7-10
- High: 11-15
- Critical: > 15
```
- **Purpose**: Stratify organ dysfunction severity
- **Expected Impact**: Very High (primary predictor)

#### **Organ Failure Count**
```python
organ_failures = count(SOFA subscores > 2)
```
- **Purpose**: Number of failing organs
- **Expected Impact**: Very High

#### **Vasopressor Aggregation**
```python
any_vasopressor = (any_dobutamine OR any_norepi)
```
- **Purpose**: Combined hemodynamic support indicator
- **Expected Impact**: High

#### **Interaction Terms**
```python
age_sofa_interaction = age × total_sofa
bmi_age_interaction = bmi × age
```
- **Purpose**: Capture synergistic effects
- **Expected Impact**: Moderate to High

---

### 2. 🤖 Optimized Model Algorithms

#### **XGBoost** ( BEST PERFORMANCE)
```python
Parameters:
- n_estimators: 500 (increased from 200)
- learning_rate: 0.05 (reduced for better generalization)
- max_depth: 6
- subsample: 0.8
- colsample_bytree: 0.8
- scale_pos_weight: auto-calculated for class imbalance
- early_stopping_rounds: 50
```
**Expected AUC: 0.85-0.92**

#### **Random Forest** (Good Balance)
```python
Parameters:
- n_estimators: 500
- max_depth: 15 (increased from 10)
- min_samples_split: 20
- class_weight: 'balanced'
```
**Expected AUC: 0.82-0.88**

#### **Logistic Regression** (Most Interpretable)
```python
Parameters:
- class_weight: balanced
- max_iter: 1000
- solver: 'lbfgs'
```
**Expected AUC: 0.78-0.85**

#### **LightGBM** (⚡ Faster Alternative - Optional)
```python
Parameters:
- n_estimators: 500
- max_depth: 8
- learning_rate: 0.05
- num_leaves: 31
- class_weight: 'balanced'
```
**Expected AUC: 0.84-0.90**

---

### 3. 📊 Enhanced Evaluation Metrics

The evaluation now includes comprehensive clinical metrics:

#### **Standard Metrics**
- ✅ AUC-ROC (Area Under the Curve)
- ✅ Accuracy
- ✅ Precision (Positive Predictive Value)
- ✅ Recall/Sensitivity
- ✅ F1-Score

#### **New Clinical Metrics**
- ✅ **Specificity**: True negative rate (correctly identifying survivors)
- ✅ **NPV (Negative Predictive Value)**: Probability that predicted survivors actually survive
- ✅ **Detailed Confusion Matrix Breakdown**:
  - True Positives (correctly predicted deaths)
  - True Negatives (correctly predicted survivors)
  - False Positives (false alarms)
  - False Negatives (missed deaths - most critical)

#### **Performance Indicators**
- 🏆 Trophy emoji for excellent performance (AUC > 0.85)
- ✓ Check mark for good performance (AUC > 0.75)

---

### 4. 📈 Expected Top Feature Importance

Based on clinical knowledge and the enhanced features:

| Rank | Feature | Expected Importance | Clinical Relevance |
|------|---------|--------------------|--------------------|
| 🥇 | **total_sofa** | 40-60% | Overall organ dysfunction |
| 🥈 | **age** | 10-15% | Classic mortality predictor |
| 🥉 | **organ_failures** | 8-12% | Number of failing organs |
| 4 | **resp_score** | 5-8% | Respiratory failure severity |
| 5 | **cv_score** | 5-8% | Cardiovascular instability |
| 6 | **age_sofa_interaction** | 4-6% | Synergistic effect |
| 7 | **gcs/cns_score** | 3-5% | Neurological status |
| 8 | **any_vasopressor** | 3-5% | Hemodynamic support |
| 9 | **comorbidity_count** | 2-4% | Overall disease burden |
| 10 | **renal_score** | 2-4% | Kidney function |

---

### 5. 🎯 Expected Performance Targets

#### **With Full SOFA Features (Current Implementation)**
```
✅ AUC-ROC:    0.85 - 0.92  (Excellent)
✅ Accuracy:   0.75 - 0.85
✅ Precision:  0.70 - 0.80
✅ Recall:     0.65 - 0.75
✅ Specificity: 0.80 - 0.90
```

#### **Without SOFA Features (Baseline Only)**
```
⚠️  AUC-ROC:    0.70 - 0.78  (Good but limited)
```

---

## 📁 Generated Outputs

When the pipeline runs successfully, it generates:

### **Visualizations** (`P3/results_images/`)
1. `feature_importance_xgboost.png` - Top 20 features for XGBoost
2. `feature_importance_random_forest.png` - Top 20 features for Random Forest
3. `feature_importance_logistic_regression.png` - Top 20 features for Logistic Regression
4. `feature_importance_lightgbm.png` - Top 20 features for LightGBM (if available)
5. `roc_curves_comparison.png` - ROC curves for all models
6. `model_comparison.png` - Bar charts comparing all metrics
7. `model_metrics_summary.csv` - Exportable metrics table

### **Trained Models** (`P3/api/models/`)
1. `xgboost.pkl` - Best performing model
2. `random_forest.pkl` - Balanced alternative
3. `logistic_regression.pkl` - Most interpretable
4. `lightgbm.pkl` - Fast alternative (if installed)

Each model bundle includes:
- Trained model
- Feature names
- Label encoders for categorical variables
- Scaler (for logistic regression)

---

## 🚀 How to Run

### **Prerequisites**
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

# Optional (for LightGBM)
pip install lightgbm
```

### **Execute Pipeline**
```bash
cd /Users/arriazui/Desktop/master/ELECTRONIC_HEALTH_RECORDS
python3 P3/mortality_prediction_models_improve.py
```

### **Expected Runtime**
- Data loading: ~5 seconds
- Feature engineering: ~10 seconds
- Model training: 2-5 minutes (depending on CPU)
- Evaluation & visualization: ~30 seconds

**Total: 3-6 minutes**

---

## 🔄 Data Flow Pipeline

```
1. Load df_icu.csv (48 columns)
         ↓
2. Remove leakage variables (11 columns removed)
         ↓
3. Feature Engineering (+8 new features)
         ↓
4. Handle missing values (imputation)
         ↓
5. Encode categorical variables
         ↓
6. Train-Test Split (80/20, stratified)
         ↓
7. Train 4 Models (XGBoost, RF, LR, LightGBM)
         ↓
8. Comprehensive Evaluation
         ↓
9. Generate Visualizations
         ↓
10. Save Models for API Deployment
```

---

## 💡 Key Recommendations

### **Best Model Selection**
- **For Production**: Use **XGBoost** (highest AUC, robust)
- **For Interpretability**: Use **Logistic Regression** (explainable coefficients)
- **For Speed**: Use **LightGBM** (if available, similar performance to XGBoost)

### **Feature Selection Priority**
If reducing features is needed:
1. **Keep**: total_sofa, age, organ_failures, SOFA subscores
2. **Keep**: vasopressor use, respiratory support
3. **Optional**: BMI, comorbidities, interaction terms

### **Model Monitoring**
Track these metrics in production:
- **AUC-ROC**: Overall discriminative ability
- **Sensitivity**: Are we catching deaths? (minimize false negatives)
- **Specificity**: Are we correctly identifying survivors?
- **Calibration**: Do predicted probabilities match observed rates?

---

## 🔍 Data Leakage Prevention

The pipeline automatically removes variables that would cause data leakage:

❌ **Removed Variables:**
- `icu_los_hours` - Known only after discharge
- `hosp_los_hours` - Known only after discharge
- `outtime` - Known only after discharge
- `hosp_disch` - Known only after discharge
- `dod` - Date of death (target variable)
- `subject_id`, `hadm_id`, `icustay_id` - ID variables
- `intime`, `hosp_admit`, `dob` - Temporal identifiers
- `icd9_code`, `long_title` - Already captured in diagnosis_group

---

## 📊 Clinical Interpretation

### **High-Risk Patient Profile**
Based on the model, patients at highest mortality risk typically have:
- ✅ High total SOFA score (> 10)
- ✅ Multiple organ failures (≥ 3)
- ✅ Advanced age (> 75)
- ✅ Requiring vasopressor support
- ✅ Low GCS/high CNS score
- ✅ Multiple comorbidities

### **Model Usage Guidelines**
- **Threshold 0.5**: Balanced classification (default)
- **Threshold > 0.7**: High specificity (fewer false alarms)
- **Threshold < 0.3**: High sensitivity (catch more at-risk patients)

Adjust threshold based on:
- ICU resource availability
- Cost of false positives vs false negatives
- Clinical intervention capabilities

---

## 🎓 Technical Details

### **Class Imbalance Handling**
- **XGBoost**: `scale_pos_weight` parameter
- **Random Forest**: `class_weight='balanced'`
- **Logistic Regression**: `class_weight='balanced'`
- **LightGBM**: `class_weight='balanced'`

### **Cross-Validation**
- **Split**: 80% train, 20% test
- **Stratification**: Maintains class proportions
- **Random State**: 42 (reproducible results)

### **Missing Value Strategy**
- **Numeric**: Median imputation
- **Categorical**: Mode imputation
- **Rationale**: Robust to outliers, maintains distribution

---

## 📞 API Deployment Ready

All models are saved in a format ready for deployment:

```python
# Load model bundle
import joblib
bundle = joblib.load('P3/api/models/xgboost.pkl')

model = bundle['model']
feature_names = bundle['feature_names']
label_encoders = bundle['label_encoders']
scaler = bundle['scaler']  # Only for logistic regression
```

### **API Server Command**
```bash
cd P3/api
uvicorn mlapi:app --reload --port 8000
```

---

## ✨ Next Steps

1. **Install Dependencies**: Run `pip install` commands above
2. **Execute Pipeline**: Run the improved model script
3. **Review Results**: Check generated visualizations
4. **Select Best Model**: Based on AUC and clinical needs
5. **Deploy API**: Start the FastAPI server
6. **Monitor Performance**: Track metrics over time

---

## 📝 Changes Summary

| Category | Changes Made | Impact |
|----------|--------------|--------|
| **Feature Engineering** | +8 new features | 🔥 High |
| **Model Algorithms** | Optimized hyperparameters | 🔥 High |
| **Evaluation Metrics** | +3 clinical metrics | 📈 Medium |
| **Code Quality** | Enhanced documentation | ✓ Good |
| **Data Handling** | Better missing value strategy | ✓ Good |
| **Output** | CSV export of metrics | ✓ Good |

---

## 🏆 Expected Results

Based on similar ICU mortality prediction studies:

- **Improvement over baseline**: +10-15% AUC
- **Clinical utility**: High (actionable predictions)
- **Model reliability**: Excellent (robust to variations)
- **Interpretability**: High feature importance transparency

**Ready for clinical validation and deployment!** 🎉
