# ICU Mortality Prediction API - Usage Guide

## Overview
The API now supports **both Basic and SOFA-enhanced models** for ICU mortality prediction.

## Installation
```bash
pip install fastapi uvicorn
```

## Starting the API
```bash
cd P3/api
uvicorn mlapi:app --reload --port 8000
```

Access the interactive documentation at: `http://localhost:8000/docs`

---

## Available Models

### Basic Models (AUC: 0.84)
- `xgboost`
- `random_forest`
- `logistic_regression`
- `lightgbm`

**Features:** 12 (demographics + comorbidities only)

### SOFA Models (AUC: 0.90) ⭐ **RECOMMENDED**
- `xgboost_sofa`
- `random_forest_sofa` ⭐ **BEST PERFORMER**
- `logistic_regression_sofa`
- `lightgbm_sofa`

**Features:** 30-35 (includes SOFA scores + organ function)

---

## API Endpoints

### 1. List Available Models
```http
GET /models
```

**Response:**
```json
{
  "basic_models": {
    "models": ["xgboost", "random_forest", "logistic_regression", "lightgbm"],
    "description": "Basic models using demographics and comorbidities only",
    "features": 12,
    "best_auc": 0.84
  },
  "sofa_models": {
    "models": ["xgboost_sofa", "random_forest_sofa", "logistic_regression_sofa", "lightgbm_sofa"],
    "description": "Enhanced models including SOFA scores and organ function",
    "features": "30-35",
    "best_auc": 0.90
  },
  "recommended": "random_forest_sofa"
}
```

---

### 2. Predict with Basic Model
```http
POST /predict/basic/{model_name}
```

**Example:** `POST /predict/basic/random_forest`

**Request Body:**
```json
{
  "GENDER": "M",
  "AGE": 65,
  "ETHNICITY_GROUP": "WHITE",
  "diabetes_comorbidity": 1,
  "hypertension_comorbidity": 1,
  "ckd_comorbidity": 0,
  "chf_comorbidity": 1,
  "copd_comorbidity": 0,
  "cancer_comorbidity": 0,
  "diagnosis_group": "CIRCULATORY",
  "resp_procedure": 1
}
```

**Response:**
```json
{
  "model_type": "basic",
  "model": "random_forest",
  "prediction": "HIGH RISK",
  "mortality_probability": 45.23,
  "risk_level": "moderate"
}
```

---

### 3. Predict with SOFA Model ⭐
```http
POST /predict/sofa/{model_name}
```

**Example:** `POST /predict/sofa/random_forest_sofa`

**Request Body:**
```json
{
  "gender": "M",
  "age": 65,
  "ethnicity_group": "WHITE",
  "flag_diabetes": 1,
  "flag_hypertension": 1,
  "flag_ckd": 0,
  "flag_chf": 1,
  "flag_copd": 0,
  "flag_cancer": 0,
  "total_sofa": 8,
  "resp_score": 2,
  "cv_score": 3,
  "liver_score": 1,
  "cns_score": 1,
  "sofa_coag": 0,
  "renal_score": 1,
  "bmi_baseline": 28.5,
  "any_dobutamine": 0,
  "any_norepi": 1,
  "any_vaso": 1,
  "resp_procedure": 1,
  "diagnosis_group": "CIRCULATORY"
}
```

**Response:**
```json
{
  "model_type": "sofa",
  "model": "random_forest_sofa",
  "prediction": "HIGH RISK",
  "mortality_probability": 52.8,
  "risk_level": "high",
  "sofa_severity": "moderate",
  "organ_failures": 2
}
```

---

## Python Client Examples

### Using Basic Model
```python
import requests

url = "http://localhost:8000/predict/basic/random_forest"
patient = {
    "GENDER": "M",
    "AGE": 65,
    "ETHNICITY_GROUP": "WHITE",
    "diabetes_comorbidity": 1,
    "hypertension_comorbidity": 1,
    "ckd_comorbidity": 0,
    "chf_comorbidity": 1,
    "copd_comorbidity": 0,
    "cancer_comorbidity": 0,
    "diagnosis_group": "CIRCULATORY",
    "resp_procedure": 1
}

response = requests.post(url, json=patient)
print(response.json())
```

### Using SOFA Model (Recommended)
```python
import requests

url = "http://localhost:8000/predict/sofa/random_forest_sofa"
patient = {
    "gender": "M",
    "age": 65,
    "ethnicity_group": "WHITE",
    "flag_diabetes": 1,
    "flag_hypertension": 1,
    "flag_ckd": 0,
    "flag_chf": 1,
    "flag_copd": 0,
    "flag_cancer": 0,
    "total_sofa": 8,
    "resp_score": 2,
    "cv_score": 3,
    "liver_score": 1,
    "cns_score": 1,
    "sofa_coag": 0,
    "renal_score": 1,
    "bmi_baseline": 28.5,
    "any_dobutamine": 0,
    "any_norepi": 1,
    "any_vaso": 1,
    "resp_procedure": 1,
    "diagnosis_group": "CIRCULATORY"
}

response = requests.post(url, json=patient)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Mortality Probability: {result['mortality_probability']}%")
print(f"Risk Level: {result['risk_level']}")
print(f"SOFA Severity: {result['sofa_severity']}")
print(f"Organ Failures: {result['organ_failures']}")
```

---

## Risk Level Interpretation

| Risk Level | Mortality Probability | Clinical Action |
|------------|----------------------|-----------------|
| **Low** | < 30% | Standard monitoring |
| **Moderate** | 30-50% | Enhanced monitoring |
| **High** | 50-70% | Intensive care required |
| **Critical** | ≥ 70% | Urgent intervention needed |

---

## SOFA Severity Categories

| Category | Total SOFA Score | Interpretation |
|----------|-----------------|----------------|
| **Low** | 0-6 | Minimal organ dysfunction |
| **Moderate** | 7-10 | Moderate organ dysfunction |
| **High** | 11-15 | Severe organ dysfunction |
| **Critical** | 16-24 | Life-threatening organ failure |

---

## Feature Requirements Comparison

### Basic Model (12 features)
✅ Easy to collect at admission
✅ No specialized assessments needed
✅ Fast prediction
⚠️ Lower accuracy (AUC 0.84)

**Required fields:**
- Demographics: Gender, Age, Ethnicity
- Comorbidities: 6 binary flags
- Clinical: Diagnosis group, Respiratory procedure

### SOFA Model (30-35 features)
✅ High accuracy (AUC 0.90)
✅ Clinically validated
✅ Provides organ-specific insights
⚠️ Requires SOFA assessment

**Required fields:**
- All Basic Model fields
- SOFA scores (7 subscores + total)
- BMI measurement
- Vasopressor use (3 flags)

---

## Model Performance Summary

| Model | Type | AUC | Accuracy | Precision | Recall | Recommendation |
|-------|------|-----|----------|-----------|---------|----------------|
| Random Forest | Basic | 0.84 | 77.5% | 26.9% | 75.2% | Good for screening |
| **Random Forest SOFA** | **SOFA** | **0.90** | **91.4%** | **53.2%** | **64.1%** | **Best overall** ⭐ |
| XGBoost | Basic | 0.84 | 73.1% | 24.2% | 81.0% | High sensitivity |
| XGBoost SOFA | SOFA | 0.90 | 87.6% | 40.9% | 73.8% | Good balance |
| LightGBM SOFA | SOFA | 0.88 | 91.1% | 53.0% | 42.7% | High precision |

---

## Backward Compatibility

The legacy endpoint is still supported:
```http
POST /predict/{model_name}
```
This endpoint uses the Basic model format for backward compatibility.

---

## Error Handling

### Model Not Found
```json
{
  "detail": "Model not found: invalid_model"
}
```

### Invalid Input
FastAPI will automatically validate input and return detailed error messages for missing or incorrect fields.

---

## Notes

1. **SOFA models provide superior performance** (6.9% improvement in AUC)
2. **Recommended for production:** `random_forest_sofa`
3. **Use Basic models when:** SOFA scores are not available
4. **BMI is optional** in SOFA models (defaults to 25.0 if not provided)
5. **All engineered features** are automatically calculated by the API

---

## Contact & Support

For issues or questions, refer to the project documentation or the model training scripts:
- `mortality_prediction_models.py` - Basic models
- `mortality_prediction_models_sofa.py` - SOFA models
