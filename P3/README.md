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
../.venv/bin/python mortality_prediction_models.py
```

This will:
- Train 3 models (XGBoost, Random Forest, Logistic Regression)
- Generate performance visualizations
- Save models to `api/models/`

### 2. Start API
```bash
../.venv/bin/uvicorn api.mlapi:app --reload --port 8000
```

> **Note:** Run from `P3/` directory

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




* P1(15%): 9.20 / 10.00,  Document (45%): 8.5; Presentation (35%): 9.5; Answers (20%): 10
* P2(30%): 7.15 / 10.00,  a) 5,6, b) 8,25, c) 9
* P3(45%): 7.15 / 10.00,  a) 5,6, b) 8,25, c) 9


This project is splited in 3 areas, db extraction, modeling and api. now we obtain extra data in the file /Users/arriazui/Desktop/master/ELECTRONIC_HEALTH_RECORDS/P3/resources_p3/df_icu.csv, i want that you create a new file and improve the model with the new states like sofa, ibm ... try to extract as much information as possible and you have to impove the results that /Users/arriazui/Desktop/master/ELECTRONIC_HEALTH_RECORDS/P3/mortality_prediction_models.py

PROMPT 1: 
explain me without changing anything what this doc does /Users/arriazui/Desktop/master/ELECTRONIC_HEALTH_RECORDS/P3/A3_GroupM (1).ipynb



PROMTP 2: 
okey with that dataframe what woudl be the best way to create a predictive model of mortalitiy using the maximum variables possibles?



## prompt 3 
Yo need to create a prediction mortality model of the 24h after enter in the icu. for that task: 
1. read the 5 first lines of the dataset to understand its columns and format /Users/arriazui/Desktop/master/ELECTRONIC_HEALTH_RECORDS/P3/db_processing/A3_v2_all_sofa_extra.ipynb 

2. that dataset would be the input for the model 
3. select the best prediction model for that structure of the dataset 
4. in a new file, using python generate the prediction model 

Some suggestions: 
Enhanced Clinical Model__ (Add First 24h Physiology)
focus in the strucutre of the dataset and generate the model 
__Add these SOFA-related variables:__

- __Respiratory:__ pf_ratio, resp_score
- __Cardiovascular:__ map_min, cv_score, any_vaso
- __Liver:__ bilirubin, liver_score
- __CNS:__ gcs, cns_score
- __Coagulation:__ platelets, sofa_coag
- __Renal:__ creatinine_mgdl, urine_ml_24h, renal_score
- __Total severity:__ total_sofa