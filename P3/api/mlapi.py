"""
ICU Mortality Prediction API
=============================
Minimal FastAPI serving trained models from mortality_prediction_models.py

Run: uvicorn mlapi:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
from pathlib import Path
import numpy as np

app = FastAPI(title="ICU Mortality Prediction API", version="1.0.0")

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


class PatientData(BaseModel):
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


@app.get("/")
def read_root():
    return FileResponse("index.html")


@app.get("/models")
def list_models():
    return {
        "models": ["xgboost", "random_forest", "logistic_regression"],
        "default": "random_forest"
    }


@app.post("/predict/{model_name}")
def predict(model_name: str, patient: PatientData):
    """Make prediction using specified model"""
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
