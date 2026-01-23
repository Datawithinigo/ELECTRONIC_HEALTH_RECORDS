# ICU Mortality Prediction

Minimal project structure for ICU mortality prediction using machine learning.

## 📁 Project Structure

```
P3/
├── mortality_prediction_models.py    # Main training script (single source of truth)
├── api/
│   ├── mlapi.py                      # FastAPI server
│   ├── index.html                    # Web interface
│   └── models/                       # Saved models (auto-generated)
│       ├── xgboost.pkl
│       ├── random_forest.pkl
│       └── logistic_regression.pkl
├── results_images/                   # Visualizations (auto-generated)
│   ├── feature_importance_xgboost.png
│   ├── feature_importance_random_forest.png
│   ├── feature_importance_logistic_regression.png
│   ├── roc_curves_comparison.png
│   └── model_comparison.png
└── ../resources_p3/
    └── df_a3_andrea_v2.csv          # Training data
```

## 🚀 Usage

### 1. Train Models
```bash
cd P3
python mortality_prediction_models.py
```

This will:
- Train 3 models (XGBoost, Random Forest, Logistic Regression)
- Generate performance visualizations
- Save models to `api/models/`

### 2. Start API
```bash
cd api
uvicorn mlapi:app --reload --port 8000
```

Visit: http://localhost:8000

### 3. Make Predictions

**Web Interface:**
- Open http://localhost:8000

**API Endpoint:**
```bash
curl -X POST http://localhost:8000/predict/random_forest \
  -H "Content-Type: application/json" \
  -d '{
    "GENDER": "M",
    "AGE": 65,
    "ETHNICITY_GROUP": "WHITE",
    "diabetes_comorbidity": 1,
    "hypertension_comorbidity": 1,
    "ckd_comorbidity": 0,
    "chf_comorbidity": 0,
    "copd_comorbidity": 0,
    "cancer_comorbidity": 0,
    "diagnosis_group": "cardiovascular",
    "resp_procedure": 0
  }'
```

## 📦 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost fastapi uvicorn
```

## 🎯 Model Performance

| Model | AUC | Accuracy |
|-------|-----|----------|
| Random Forest | 0.842 | 77.5% |
| XGBoost | 0.841 | 73.1% |
| Logistic Regression | 0.831 | 73.7% |

## 📊 Key Features

Top 3 predictors:
1. **Respiratory procedure** (78-99% importance)
2. **CKD comorbidity** (11-49% importance)
3. **Age** (1-45% importance)
