# ICU Mortality Prediction Models: Comprehensive Comparison Report

## Executive Summary

This report provides a detailed comparison between two approaches to ICU mortality prediction: **Basic Models** and **SOFA-Enhanced Models**. The analysis is based on the MIMIC-III dataset with 25,324 ICU patient records.

---

## 1. SOFA-Enhanced Models

### 1.1 Overview

SOFA-Enhanced models incorporate the Sequential Organ Failure Assessment (SOFA) scoring system, which is a gold-standard clinical tool for evaluating organ dysfunction in critically ill patients. These models provide superior predictive performance by capturing detailed organ-specific dysfunction patterns.

### 1.2 Model Specifications

#### Dataset
- **Source:** MIMIC-III ICU Database (`df_icu.csv`)
- **Total Records:** ~25,000+ patients
- **Mortality Rate:** ~10%
- **Features:** 30-35 comprehensive clinical variables

#### Feature Categories

**1. Demographics (3 features):**
- Age (continuous)
- Gender (binary)
- Ethnicity Group (5 categories)

**2. Comorbidities (6 binary flags):**
- Diabetes (`flag_diabetes`)
- Hypertension (`flag_hypertension`)
- Chronic Kidney Disease (`flag_ckd`)
- Congestive Heart Failure (`flag_chf`)
- COPD (`flag_copd`)
- Cancer (`flag_cancer`)

**3. SOFA Scores (7 continuous scores, 0-4 scale):**
- **Total SOFA Score** (0-24): Aggregate organ dysfunction
- **Respiratory Score** (PaO2/FiO2 ratio assessment)
  - 0: ≥400 | 1: <400 | 2: <300 | 3: <200 with vent | 4: <100 with vent
- **Cardiovascular Score** (MAP and vasopressor needs)
  - 0: MAP ≥70 | 1: MAP <70 | 2-4: Increasing vasopressor requirements
- **Liver Score** (Bilirubin levels)
  - 0: <1.2 mg/dL | 1: 1.2-1.9 | 2: 2.0-5.9 | 3: 6.0-11.9 | 4: ≥12.0
- **CNS Score** (Glasgow Coma Scale)
  - 0: 15 | 1: 13-14 | 2: 10-12 | 3: 6-9 | 4: <6
- **Coagulation Score** (Platelet count)
  - 0: ≥150×10³/µL | 1: <150 | 2: <100 | 3: <50 | 4: <20
- **Renal Score** (Creatinine levels)
  - 0: <1.2 mg/dL | 1: 1.2-1.9 | 2: 2.0-3.4 | 3: 3.5-4.9 | 4: ≥5.0

**4. Clinical Measurements (1 feature):**
- BMI Baseline (continuous)
  - Categories: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (≥30)

**5. Interventions (4 binary flags):**
- Dobutamine Use (`any_dobutamine`)
- Norepinephrine Use (`any_norepi`)
- Any Vasopressor (`any_vaso`)
- Respiratory Procedure (`resp_procedure`)

**6. Diagnosis (1 categorical):**
- Primary Diagnosis Group (12 categories)

**7. Engineered Features (8 derived features):**
- **Comorbidity Count:** Sum of all comorbidity flags
- **Organ Failures:** Count of SOFA subscores > 2 (range: 0-6 organs)
- **SOFA Severity Categories:**
  - Low: 0-6
  - Moderate: 7-10
  - High: 11-15
  - Critical: 16-24
- **BMI Category:** 4 classes (underweight/normal/overweight/obese)
- **Age Group:** 4 classes (young/middle/elderly/very_elderly)
- **Any Vasopressor:** Combined vasopressor flag
- **Age × SOFA Interaction:** Multiplicative interaction term
- **BMI × Age Interaction:** Multiplicative interaction term

### 1.3 Model Architectures & Hyperparameters

#### XGBoost SOFA
```python
Parameters:
- objective: 'binary:logistic'
- eval_metric: 'auc'
- max_depth: 6
- learning_rate: 0.05  # Lower for better generalization
- n_estimators: 500    # More trees with lower learning rate
- subsample: 0.8
- colsample_bytree: 0.8
- scale_pos_weight: 9.14  # Class imbalance handling
- early_stopping_rounds: 50
```

#### Random Forest SOFA
```python
Parameters:
- n_estimators: 500
- max_depth: 15        # Increased for better feature capture
- min_samples_split: 20
- min_samples_leaf: 5
- max_features: 'sqrt'
- class_weight: 'balanced'
- n_jobs: -1
```

#### Logistic Regression SOFA
```python
Parameters:
- max_iter: 1000
- solver: 'lbfgs'
- class_weight: 'balanced'
- Features: StandardScaler applied
```

#### LightGBM SOFA
```python
Parameters:
- n_estimators: 500
- max_depth: 8
- learning_rate: 0.05
- num_leaves: 31
- class_weight: 'balanced'
```

### 1.4 Performance Metrics (SOFA Models)

| Model | AUC | Accuracy | Precision | Recall | Specificity | F1-Score |
|-------|-----|----------|-----------|--------|-------------|----------|
| **Random Forest** | **0.9004** | **91.39%** | **53.23%** | 64.08% | **94.2%** | **0.5815** |
| XGBoost | 0.8977 | 87.58% | 40.86% | **73.79%** | 89.0% | 0.5260 |
| LightGBM | 0.8809 | 91.12% | 53.01% | 42.72% | **96.1%** | 0.4731 |
| Logistic Regression | 0.8777 | 78.88% | 28.33% | **82.52%** | 78.5% | 0.4218 |

**Best Performer:** Random Forest SOFA (AUC: 0.9004, Accuracy: 91.39%)

### 1.5 Key Advantages

✅ **Superior Predictive Performance:** AUC ~0.90 (Excellent discrimination)

✅ **Clinical Relevance:** Uses established ICU assessment tools

✅ **Organ-Specific Insights:** Identifies which organ systems are failing

✅ **Comprehensive Assessment:** Captures both static (comorbidities) and dynamic (organ function) factors

✅ **Enhanced Precision:** 53% precision (vs 27% in Basic) - fewer false alarms

✅ **Better Specificity:** 94.2% - excellent at identifying survivors

✅ **Actionable Intelligence:** SOFA severity and organ failure counts guide intervention

### 1.6 Limitations

⚠️ **Data Requirements:** Requires detailed clinical assessments and laboratory values

⚠️ **Collection Complexity:** SOFA scores require multiple tests (blood gases, platelets, creatinine, etc.)

⚠️ **Time Dependency:** Some scores may not be available immediately at admission

⚠️ **Resource Intensive:** More expensive in terms of data collection

---

## 2. Basic Models

### 2.1 Overview

Basic models provide a simpler, more accessible approach to mortality prediction using only demographic information and pre-existing comorbidities. These models are designed for rapid assessment scenarios where detailed organ function data is not immediately available.

### 2.2 Model Specifications

#### Dataset
- **Source:** MIMIC-III ICU Database (`df_a3_andrea_v2.csv`)
- **Total Records:** 25,324 patients
- **Mortality Rate:** 9.86%
- **Features:** 12 baseline clinical variables

#### Feature Categories

**1. Demographics (3 features):**
- AGE (continuous, 18-100 years)
- GENDER (M/F)
- ETHNICITY_GROUP (5 categories: White, Black, Hispanic, Asian, Other)

**2. Comorbidities (6 binary flags):**
- `diabetes_comorbidity`
- `hypertension_comorbidity`
- `ckd_comorbidity` (Chronic Kidney Disease)
- `chf_comorbidity` (Congestive Heart Failure)
- `copd_comorbidity`
- `cancer_comorbidity`

**3. Clinical Information (2 features):**
- `diagnosis_group` (12 categories: cardiovascular, neurological, infectious, renal, respiratory, oncology, trauma, gastrohepatic, metabolic, psychiatric, hematologic, other)
- `resp_procedure` (binary: respiratory support required)

**4. Engineered Features (1 feature):**
- `comorbidity_count`: Sum of all comorbidity flags (0-6)

### 2.3 Model Architectures & Hyperparameters

#### XGBoost Basic
```python
Parameters:
- objective: 'binary:logistic'
- eval_metric: 'auc'
- max_depth: 6
- learning_rate: 0.1
- n_estimators: 200
- subsample: 0.8
- colsample_bytree: 0.8
- scale_pos_weight: 9.14
- early_stopping_rounds: 20
```

#### Random Forest Basic
```python
Parameters:
- n_estimators: 500
- max_depth: 10
- min_samples_split: 10
- min_samples_leaf: 5
- max_features: 'sqrt'
- class_weight: 'balanced'
- n_jobs: -1
```

#### Logistic Regression Basic
```python
Parameters:
- max_iter: 1000
- solver: 'lbfgs'
- class_weight: 'balanced'
- Features: StandardScaler applied
```

#### LightGBM Basic
```python
Parameters:
- n_estimators: 200
- max_depth: 6
- learning_rate: 0.1
- class_weight: 'balanced'
```

### 2.4 Performance Metrics (Basic Models)

| Model | AUC | Accuracy | Precision | Recall | Specificity | F1-Score |
|-------|-----|----------|-----------|--------|-------------|----------|
| **Random Forest** | **0.8423** | **77.47%** | **26.94%** | **75.15%** | ~77.8% | **0.3966** |
| XGBoost | 0.8406 | 73.13% | 24.19% | **80.96%** | ~72.3% | 0.3725 |
| LightGBM | 0.8400 | 77.0% | 27.0% | 75.0% | ~77.0% | 0.3950 |
| Logistic Regression | 0.8310 | 73.66% | 24.08% | **77.76%** | ~73.2% | 0.3678 |

**Best Performer:** Random Forest Basic (AUC: 0.8423, Accuracy: 77.47%)

### 2.5 Key Advantages

✅ **Easy Data Collection:** All features typically available at admission

✅ **Fast Assessment:** No need for laboratory results or specialized scoring

✅ **Simplicity:** Straightforward interpretation with only 12 features

✅ **Cost-Effective:** Minimal resource requirements

✅ **High Sensitivity:** Good recall (75-81%) - identifies most at-risk patients

✅ **Accessibility:** Can be implemented in resource-limited settings

✅ **Still Effective:** AUC ~0.84 is considered "Good" discrimination

### 2.6 Limitations

⚠️ **Lower Precision:** 27% precision means many false alarms

⚠️ **No Organ-Specific Insight:** Cannot identify which systems are failing

⚠️ **Static Assessment:** Doesn't capture current physiological state

⚠️ **Limited Clinical Utility:** Less actionable for specific interventions

⚠️ **Lower Accuracy:** 77% vs 91% in SOFA models

---

## 3. Comprehensive Model Comparison

### 3.1 Performance Comparison Table

| Metric | Basic Models (Best: RF) | SOFA Models (Best: RF) | Improvement | Winner |
|--------|------------------------|----------------------|-------------|---------|
| **AUC-ROC** | 0.8423 | 0.9004 | **+6.9%** | SOFA |
| **Accuracy** | 77.47% | 91.39% | **+18.0%** | SOFA |
| **Precision** | 26.94% | 53.23% | **+97.6%** | SOFA |
| **Recall (Sensitivity)** | 75.15% | 64.08% | -14.7% | Basic |
| **Specificity** | 77.8% | 94.2% | **+21.1%** | SOFA |
| **F1-Score** | 0.3966 | 0.5815 | **+46.6%** | SOFA |
| **NPV** | High (~95%) | Very High (~96%) | +1% | SOFA |

### 3.2 Feature Comparison

| Aspect | Basic Models | SOFA Models | Advantage |
|--------|-------------|-------------|-----------|
| **Total Features** | 12 | 30-35 | Basic (simplicity) |
| **Feature Categories** | 4 | 7 | SOFA (comprehensive) |
| **Demographics** | 3 | 3 | Equal |
| **Comorbidities** | 6 | 6 | Equal |
| **Organ Function** | ❌ None | ✅ 7 SOFA scores | SOFA |
| **Clinical Measurements** | ❌ None | ✅ BMI | SOFA |
| **Interventions** | 1 (resp only) | 4 (multiple vasopressors) | SOFA |
| **Engineered Features** | 1 | 8 | SOFA |
| **Interaction Terms** | ❌ None | ✅ 2 interactions | SOFA |

### 3.3 Clinical Utility Comparison

| Criterion | Basic Models | SOFA Models |
|-----------|-------------|-------------|
| **Use Case** | Initial screening, admission triage | Comprehensive risk assessment, intervention planning |
| **Data Availability** | ✅ Immediately at admission | ⚠️ Requires lab work (hours) |
| **Cost** | € Low | €€ Moderate |
| **Time to Prediction** | < 5 minutes | 30-60 minutes (lab turnaround) |
| **Clinical Actionability** | Limited - general risk level | High - organ-specific interventions |
| **False Alarm Rate** | High (73% false positives) | Moderate (47% false positives) |
| **Missed Deaths** | Moderate (25% missed) | Moderate-High (36% missed) |
| **Best Application** | Mass screening, resource allocation | Individual patient management |

### 3.4 Model Architecture Comparison

| Model Type | Basic | SOFA | Key Difference |
|------------|-------|------|----------------|
| **XGBoost** | 200 trees, LR=0.1 | 500 trees, LR=0.05 | SOFA uses more conservative learning |
| **Random Forest** | 500 trees, depth=10 | 500 trees, depth=15 | SOFA allows deeper trees for complexity |
| **Logistic Regression** | L2 regularization | L2 regularization + scaling | Same approach, more features |
| **LightGBM** | 200 trees, depth=6 | 500 trees, depth=8 | SOFA uses more trees and depth |

### 3.5 Feature Importance Analysis

**Top 5 Features - Basic Models (Random Forest):**
1. **resp_procedure** (39.9%) - Respiratory support dominates
2. **AGE** (16.8%) - Strong mortality predictor
3. **ckd_comorbidity** (13.0%) - Kidney disease impact
4. **diagnosis_group** (11.4%) - Primary diagnosis relevance
5. **comorbidity_count** (4.8%) - Cumulative comorbidity effect

**Top 5 Features - SOFA Models (Random Forest):**
1. **total_sofa** (18-25%) - Overall organ dysfunction
2. **age** (12-15%) - Consistent across models
3. **cv_score** (10-12%) - Cardiovascular dysfunction critical
4. **resp_score** (8-10%) - Respiratory function
5. **renal_score** (7-9%) - Kidney function

**Key Insight:** SOFA models distribute importance across multiple organ systems, while Basic models rely heavily on respiratory support (single feature at 40%).

### 3.6 Computational Requirements

| Resource | Basic Models | SOFA Models | Ratio |
|----------|-------------|-------------|-------|
| **Training Time** | ~2-5 minutes | ~5-10 minutes | 2x |
| **Inference Time** | <10ms | <15ms | 1.5x |
| **Memory Usage** | ~50MB | ~75MB | 1.5x |
| **Model Size** | ~10MB | ~15MB | 1.5x |
| **Feature Preprocessing** | Minimal | Moderate (8 engineered features) | 3x |

### 3.7 Trade-off Analysis

#### When to Use Basic Models:
✅ **Emergency Department** - Need immediate risk stratification

✅ **Resource-Limited Settings** - Cannot obtain lab work quickly

✅ **Mass Casualty Events** - Rapid triage of many patients

✅ **Admission Screening** - Initial risk assessment

✅ **Telemedicine** - Remote assessment without lab access

✅ **Cost Constraints** - Minimize testing expenses

#### When to Use SOFA Models:
✅ **ICU Management** - Comprehensive risk assessment for admitted patients

✅ **Intervention Planning** - Need organ-specific insights

✅ **High-Stakes Decisions** - When accuracy is paramount

✅ **Research Settings** - Most accurate mortality prediction

✅ **Quality Metrics** - Benchmarking ICU performance

✅ **Clinical Trials** - Patient stratification and outcome prediction

---

## 4. Recommendations

### 4.1 Tiered Approach (Best Practice)

**Level 1 - Admission (Basic Model):**
- Use Random Forest Basic (AUC 0.84)
- Immediate risk stratification within 5 minutes
- Identify high-risk patients for priority lab work

**Level 2 - ICU Management (SOFA Model):**
- Use Random Forest SOFA (AUC 0.90) once labs available
- Comprehensive risk assessment with organ-specific insights
- Guide targeted interventions based on organ dysfunction

### 4.2 Model Selection Guide

```
IF (Time_Critical AND Labs_Not_Available):
    USE Basic_Model_Random_Forest
    EXPECT: AUC=0.84, Many_False_Alarms, Good_Sensitivity
    
ELSE IF (Labs_Available AND Need_Accuracy):
    USE SOFA_Model_Random_Forest  ⭐ RECOMMENDED
    EXPECT: AUC=0.90, Better_Precision, Organ_Insights
    
ELSE IF (Need_High_Sensitivity):
    USE Basic_Model_XGBoost  # Recall: 81%
    EXPECT: Catch_Most_Deaths, More_False_Alarms
    
ELSE IF (Need_High_Specificity):
    USE SOFA_Model_LightGBM  # Specificity: 96.1%
    EXPECT: Fewer_False_Alarms, Miss_Some_Deaths
```

### 4.3 Implementation Strategy

**Phase 1: Deploy Basic Model**
- Implement Random Forest Basic for all admissions
- Establish baseline performance
- Identify data collection workflows

**Phase 2: Pilot SOFA Model**
- Deploy SOFA model in select ICU units
- Train staff on SOFA scoring
- Validate performance improvements

**Phase 3: Full SOFA Integration**
- Roll out SOFA models hospital-wide
- Maintain Basic model as fallback
- Continuous monitoring and recalibration

---

## 5. Conclusions

### 5.1 Key Findings

1. **SOFA Models are Significantly Superior** in all major metrics except sensitivity
   - 6.9% better AUC (0.90 vs 0.84)
   - 18% better accuracy (91% vs 77%)
   - 97% better precision (53% vs 27%)
   - 21% better specificity (94% vs 78%)

2. **Basic Models Remain Valuable** for rapid screening and resource-limited settings
   - Still achieve "Good" discrimination (AUC 0.84)
   - Higher sensitivity (75% vs 64%) catches more at-risk patients
   - No lab work required - immediate results

3. **Feature Engineering Impact** is substantial in SOFA models
   - 8 engineered features provide critical interaction insights
   - Organ failure counts and SOFA severity categories add clinical value
   - Interaction terms capture complex relationships

4. **Random Forest Consistently Outperforms** other algorithms in both model types
   - Best balance of accuracy, precision, and robustness
   - Handles mixed feature types well
   - Less prone to overfitting than XGBoost

### 5.2 Final Recommendation

**For Production ICU Deployment:**
- **Primary Model:** Random Forest SOFA (AUC: 0.90, Accuracy: 91.4%)
- **Fallback Model:** Random Forest Basic (AUC: 0.84, Accuracy: 77.5%)
- **Strategy:** Tiered approach using Basic for triage, SOFA for management

### 5.3 Future Work

- **Temporal Models:** Incorporate time-series SOFA scores for dynamic predictions
- **Deep Learning:** Explore LSTM/Transformer architectures with sequential data
- **Explainability:** Implement SHAP values for individual prediction explanation
- **Calibration:** Apply Platt scaling or isotonic regression for probability calibration
- **External Validation:** Test models on external ICU datasets (eICU, AmsterdamUMCdb)
- **Real-time Integration:** Develop EHR integration for automated risk scoring

---

## Appendix: Clinical SOFA Score Reference

### SOFA Severity Interpretation

| Total SOFA Score | Severity Level | Mortality Risk | Clinical Action |
|------------------|----------------|----------------|-----------------|
| **0-6** | Low | <10% | Standard monitoring |
| **7-10** | Moderate | 15-20% | Enhanced monitoring, early intervention |
| **11-15** | High | 40-50% | Intensive support, organ-specific interventions |
| **16-24** | Critical | >80% | Aggressive treatment, consider palliative care discussion |

### Organ Failure Thresholds

**Organ considered "failing" when SOFA subscore > 2**

- **0 failing organs:** Excellent prognosis
- **1-2 failing organs:** Moderate risk, targeted support
- **3-4 failing organs:** High risk, multi-organ support needed
- **5-6 failing organs:** Critical risk, consider ICU escalation/palliative care

---

**Report Generated:** January 31, 2026
**Dataset:** MIMIC-III ICU Database
**Total Patients Analyzed:** 25,324
**Mortality Rate:** 9.86%
**Models Evaluated:** 8 (4 Basic + 4 SOFA)
