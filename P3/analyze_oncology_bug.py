import joblib
import numpy as np

bundle = joblib.load('api/models/lightgbm.pkl')
model = bundle['model']
feature_names = bundle['feature_names']
label_encoders = bundle['label_encoders']

# Test both scenarios
scenarios = [
    ("NO Cancer + Oncology", {
        'gender': 'M', 'ethnicity_group': 'WHITE', 'age': 64,
        'flag_diabetes': 0, 'flag_hypertension': 0, 'flag_ckd': 0,
        'flag_chf': 0, 'flag_copd': 0, 'flag_cancer': 0,
        'diagnosis_group': 'oncology', 'resp_procedure': 1,
        'comorbidity_count': 0
    }),
    ("YES Cancer + Oncology", {
        'gender': 'M', 'ethnicity_group': 'WHITE', 'age': 64,
        'flag_diabetes': 0, 'flag_hypertension': 0, 'flag_ckd': 0,
        'flag_chf': 0, 'flag_copd': 0, 'flag_cancer': 1,
        'diagnosis_group': 'oncology', 'resp_procedure': 1,
        'comorbidity_count': 1
    })
]

print("="*70)
print("DIAGNOSIS-COMORBIDITY INTERACTION ANALYSIS")
print("="*70)
print()

for name, data in scenarios:
    # Encode
    encoded = data.copy()
    for col, encoder in label_encoders.items():
        if col in encoded:
            encoded[col] = encoder.transform([str(encoded[col])])[0]
    
    X = np.array([[encoded.get(f, 0) for f in feature_names]])
    prob = model.predict_proba(X)[0, 1]
    
    print(f"{name}:")
    print(f"  Features: {dict(zip(feature_names, X[0]))}")
    print(f"  Mortality: {prob*100:.2f}%")
    print()

# Check feature importance
print("="*70)
print("FEATURE IMPORTANCES:")
print("="*70)
importance = model.feature_importances_
for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
    print(f"  {name:25s}: {imp:.4f}")

"""


This is a data quality or model training issue, not an API bug. The model learned this counter-intuitive pattern from the training data. This happens because:

Oncology diagnosis already implies cancer-related condition
When patients have "oncology" diagnosis WITHOUT the cancer comorbidity flag, they might be sicker (maybe undiagnosed or more advanced cancer)
When they have "oncology" WITH the cancer flag explicitly set, they might be better documented/managed cases
This is a model artifact from the training data. Let me check other models to see if they have the same issue:

Found the problem! This is a model training issue, not an API bug:

LightGBM: Cancer DECREASES mortality by 4.81% ❌
Random Forest: Cancer DECREASES mortality by 12.20% ❌
XGBoost: Cancer increases mortality by 4.72% ✓
Logistic Regression: Cancer increases mortality by 14.02% ✓
Root Cause
When the diagnosis is "oncology" (cancer-related), LightGBM and Random Forest learned a counter-intuitive pattern from the training data where having the cancer comorbidity flag explicitly set actually predicts LOWER mortality. This likely happened because:

Data artifact: Patients with oncology diagnosis but NO cancer flag might be more acutely ill or have worse outcomes
Documentation bias: Documented cancer cases (flag=1) might be better managed
Selection bias in the training data
Solution
The models need to be retrained with better data or feature engineering to fix this. For now, you have two options:

Use XGBoost or Logistic Regression for oncology patients (they behave correctly)
Retrain the models with:
Feature interaction terms (diagnosis × cancer)
Better data cleaning
Or remove redundant features
This is a limitation of the trained models, not the API implementation. The API is correctly passing all parameters - the models themselves learned this incorrect pattern.


--- ####################################################
SOFA Models Results:
LightGBM SOFA: Cancer increases mortality by 0.32% ✓ (small but correct direction)
Random Forest SOFA: Cancer increases mortality by 0.73% ✓ (small but correct direction)
XGBoost SOFA: Cancer DECREASES mortality by 0.92% ❌ (wrong but minimal impact)
Logistic Regression SOFA: Cancer increases mortality by 11.69% ✓ (correct and strong)
Summary Comparison:
Basic Models (without SOFA):

LightGBM: -4.81% ❌
Random Forest: -12.20% ❌
XGBoost: +4.72% ✓
Logistic Regression: +14.02% ✓
SOFA Models:

LightGBM: +0.32% ✓
Random Forest: +0.73% ✓
XGBoost: -0.92% ❌ (minimal)
Logistic Regression: +11.69% ✓
The SOFA models are much more reliable because they have additional clinical features (SOFA scores, organ failures, etc.) that provide better context for the diagnosis-comorbidity relationship. The basic models only have demographics and comorbidities, making them more susceptible to confounding patterns in the data.

Recommendation: For oncology patients, prefer using the SOFA-enhanced models, especially Logistic Regression SOFA or Random Forest SOFA




"""