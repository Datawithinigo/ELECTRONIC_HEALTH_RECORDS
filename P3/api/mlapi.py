"""
ICU Mortality Prediction API
=============================
FastAPI serving trained models for ICU mortality prediction
Supports both Basic and SOFA-enhanced models

Run: uvicorn mlapi:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import joblib
from pathlib import Path
import numpy as np

app = FastAPI(title="ICU Mortality Prediction API", version="2.0.0")

MODELS_DIR = Path(__file__).parent / "models"
models_cache = {}

def load_model(model_name: str):
    """Lazy load models"""
    if model_name not in models_cache:
        model_path = MODELS_DIR / f"{model_name}.pkl"
        if not model_path.exists():
            raise HTTPException(404, f"Model not found: {model_name}")
        models_cache[model_name] = joblib.load(model_path)
    return models_cache[model_name]


class PatientDataBasic(BaseModel):
    """Basic model patient data (no SOFA scores)"""
    GENDER: str
    AGE: int
    ETHNICITY_GROUP: str
    diabetes_comorbidity: int
    hypertension_comorbidity: int
    ckd_comorbidity: int
    chf_comorbidity: int
    copd_comorbidity: int
    cancer_comorbidity: int
    diagnosis_group: str
    resp_procedure: int


class PatientDataSOFA(BaseModel):
    """SOFA-enhanced model patient data"""
    # Basic demographics
    gender: str
    age: int
    ethnicity_group: str
    
    # Comorbidities
    flag_diabetes: int
    flag_hypertension: int
    flag_ckd: int
    flag_chf: int
    flag_copd: int
    flag_cancer: int
    
    # SOFA scores
    total_sofa: float
    resp_score: float
    cv_score: float
    liver_score: float
    cns_score: float
    sofa_coag: float
    renal_score: float
    
    # Clinical measurements
    bmi_baseline: Optional[float] = None
    
    # Interventions
    any_dobutamine: int = 0
    any_norepi: int = 0
    any_vaso: int = 0
    resp_procedure: int = 0
    
    # Diagnosis
    diagnosis_group: str


@app.get("/")
def read_root():
    index_path = Path(__file__).parent / "index.html"
    return FileResponse(index_path)


@app.get("/models")
def list_models():
    """List all available models"""
    return {
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


@app.post("/predict/basic/{model_name}")
def predict_basic(model_name: str, patient: PatientDataBasic):
    """Make prediction using Basic model (no SOFA scores required)"""
    bundle = load_model(model_name)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    label_encoders = bundle["label_encoders"]
    scaler = bundle.get("scaler")
    
    patient_dict = patient.dict()
    patient_dict["comorbidity_count"] = sum([
        patient_dict["diabetes_comorbidity"],
        patient_dict["hypertension_comorbidity"],
        patient_dict["ckd_comorbidity"],
        patient_dict["chf_comorbidity"],
        patient_dict["copd_comorbidity"],
        patient_dict["cancer_comorbidity"]
    ])
    
    for col, encoder in label_encoders.items():
        if col in patient_dict:
            try:
                patient_dict[col] = encoder.transform([str(patient_dict[col])])[0]
            except ValueError:
                patient_dict[col] = 0
    
    X = np.array([[patient_dict[f] for f in feature_names]])
    if scaler is not None:
        X = scaler.transform(X)
    
    mortality_prob = float(model.predict_proba(X)[0, 1])
    mortality_pred = int(mortality_prob >= 0.5)
    
    return {
        "model_type": "basic",
        "model": model_name,
        "prediction": "HIGH RISK" if mortality_pred == 1 else "LOW RISK",
        "mortality_probability": round(mortality_prob * 100, 2),
        "risk_level": (
            "critical" if mortality_prob >= 0.7 else 
            "high" if mortality_prob >= 0.5 else 
            "moderate" if mortality_prob >= 0.3 else 
            "low"
        )
    }


@app.post("/predict/sofa/{model_name}")
def predict_sofa(model_name: str, patient: PatientDataSOFA):
    """Make prediction using SOFA-enhanced model (requires SOFA scores)"""
    bundle = load_model(model_name)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    label_encoders = bundle["label_encoders"]
    scaler = bundle.get("scaler")
    
    patient_dict = patient.dict()
    
    # Create engineered features
    # 1. Comorbidity count
    patient_dict["comorbidity_count"] = sum([
        patient_dict["flag_diabetes"],
        patient_dict["flag_hypertension"],
        patient_dict["flag_ckd"],
        patient_dict["flag_chf"],
        patient_dict["flag_copd"],
        patient_dict["flag_cancer"]
    ])
    
    # 2. BMI category
    if patient_dict.get("bmi_baseline"):
        bmi = patient_dict["bmi_baseline"]
        if bmi < 18.5:
            patient_dict["bmi_category"] = 0  # underweight
        elif bmi < 25:
            patient_dict["bmi_category"] = 1  # normal
        elif bmi < 30:
            patient_dict["bmi_category"] = 2  # overweight
        else:
            patient_dict["bmi_category"] = 3  # obese
    else:
        patient_dict["bmi_baseline"] = 25.0  # median
        patient_dict["bmi_category"] = 1
    
    # 3. Age group
    age = patient_dict["age"]
    if age < 40:
        patient_dict["age_group"] = 0  # young
    elif age < 60:
        patient_dict["age_group"] = 1  # middle
    elif age < 75:
        patient_dict["age_group"] = 2  # elderly
    else:
        patient_dict["age_group"] = 3  # very_elderly
    
    # 4. SOFA severity
    total_sofa = patient_dict["total_sofa"]
    if total_sofa <= 6:
        patient_dict["sofa_severity"] = 0  # low
    elif total_sofa <= 10:
        patient_dict["sofa_severity"] = 1  # moderate
    elif total_sofa <= 15:
        patient_dict["sofa_severity"] = 2  # high
    else:
        patient_dict["sofa_severity"] = 3  # critical
    
    # 5. Organ failures (SOFA subscores > 2)
    patient_dict["organ_failures"] = sum([
        patient_dict["resp_score"] > 2,
        patient_dict["cv_score"] > 2,
        patient_dict["liver_score"] > 2,
        patient_dict["cns_score"] > 2,
        patient_dict["sofa_coag"] > 2,
        patient_dict["renal_score"] > 2
    ])
    
    # 6. Any vasopressor
    patient_dict["any_vasopressor"] = int(
        patient_dict["any_dobutamine"] or patient_dict["any_norepi"]
    )
    
    # 7. Interaction terms
    patient_dict["age_sofa_interaction"] = patient_dict["age"] * patient_dict["total_sofa"]
    patient_dict["bmi_age_interaction"] = patient_dict["bmi_baseline"] * patient_dict["age"]
    
    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in patient_dict:
            try:
                patient_dict[col] = encoder.transform([str(patient_dict[col])])[0]
            except ValueError:
                patient_dict[col] = 0
    
    # Prepare features in correct order
    X = np.array([[patient_dict.get(f, 0) for f in feature_names]])
    if scaler is not None:
        X = scaler.transform(X)
    
    mortality_prob = float(model.predict_proba(X)[0, 1])
    mortality_pred = int(mortality_prob >= 0.5)
    
    return {
        "model_type": "sofa",
        "model": model_name,
        "prediction": "HIGH RISK" if mortality_pred == 1 else "LOW RISK",
        "mortality_probability": round(mortality_prob * 100, 2),
        "risk_level": (
            "critical" if mortality_prob >= 0.7 else 
            "high" if mortality_prob >= 0.5 else 
            "moderate" if mortality_prob >= 0.3 else 
            "low"
        ),
        "sofa_severity": ["low", "moderate", "high", "critical"][patient_dict["sofa_severity"]],
        "organ_failures": patient_dict["organ_failures"]
    }


# Backward compatibility - keep old endpoint
@app.post("/predict/{model_name}")
def predict(model_name: str, patient: PatientDataBasic):
    """Legacy endpoint - uses basic model"""
    return predict_basic(model_name, patient)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
