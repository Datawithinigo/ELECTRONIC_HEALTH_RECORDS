"""
ICU Mortality Prediction Model
================================
Complete pipeline implementing three models:
1. XGBoost (best performance)
2. Random Forest (good balance)
3. Logistic Regression (most interpretable)

Author: Generated for EHR Analysis
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =====================================================================
# 1. DATA LOADING AND PREPROCESSING
# =====================================================================

def load_and_preprocess_data(filepath):
    """Load CSV and perform initial preprocessing"""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    
    # Display mortality distribution
    print(f"\nMortality distribution:")
    print(df['MORTALITY'].value_counts())
    mortality_rate = df['MORTALITY'].mean() * 100
    print(f"Mortality rate: {mortality_rate:.2f}%")
    
    return df


def feature_engineering(df):
    """Create additional features and prepare for modeling"""
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    
    df_model = df.copy()
    
    # Remove temporal variables that cause data leakage
    # (ICU_LOS_HOURS and HOSP_LOS_HOURS are known only after discharge)
    leakage_cols = ['ICU_DISCH', 'HOSP_DISCH', 'ICU_LOS_HOURS', 'HOSP_LOS_HOURS']
    df_model = df_model.drop(columns=[col for col in leakage_cols if col in df_model.columns])
    
    # Remove ID columns and dates
    id_cols = ['SUBJECT_ID', 'HADM_ID', 'ICU_ADMIT', 'HOSP_ADMIT', 'DOB']
    df_model = df_model.drop(columns=[col for col in id_cols if col in df_model.columns])
    
    # Remove ICD9_CODE and LONG_TITLE (already captured in diagnosis_group)
    if 'ICD9_CODE' in df_model.columns:
        df_model = df_model.drop(columns=['ICD9_CODE'])
    if 'LONG_TITLE' in df_model.columns:
        df_model = df_model.drop(columns=['LONG_TITLE'])
    
    # Create comorbidity count
    comorbidity_cols = [
        'diabetes_comorbidity', 'hypertension_comorbidity', 
        'ckd_comorbidity', 'chf_comorbidity', 
        'copd_comorbidity', 'cancer_comorbidity'
    ]
    
    available_comorbidities = [col for col in comorbidity_cols if col in df_model.columns]
    if available_comorbidities:
        df_model['comorbidity_count'] = df_model[available_comorbidities].sum(axis=1)
        print(f"\nComorbidity count created from: {available_comorbidities}")
    
    # Handle missing values
    print(f"\nMissing values before imputation:")
    missing = df_model.isnull().sum()
    print(missing[missing > 0])
    
    # Impute numeric columns with median
    numeric_cols = df_model.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'MORTALITY']
    
    for col in numeric_cols:
        if df_model[col].isnull().any():
            median_val = df_model[col].median()
            df_model[col].fillna(median_val, inplace=True)
            print(f"  - {col}: filled with median ({median_val:.2f})")
    
    # Impute categorical columns with mode
    categorical_cols = df_model.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_model[col].isnull().any():
            mode_val = df_model[col].mode()[0] if not df_model[col].mode().empty else 'UNKNOWN'
            df_model[col].fillna(mode_val, inplace=True)
            print(f"  - {col}: filled with mode ({mode_val})")
    
    print(f"\nFinal dataset shape: {df_model.shape}")
    print(f"Features: {[col for col in df_model.columns if col != 'MORTALITY']}")
    
    return df_model


def prepare_train_test_split(df_model):
    """Split data into train and test sets"""
    print("\n" + "=" * 70)
    print("TRAIN-TEST SPLIT")
    print("=" * 70)
    
    # Separate features and target
    X = df_model.drop('MORTALITY', axis=1)
    y = df_model['MORTALITY']
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nCategorical features to encode: {categorical_cols}")
    
    X_encoded = X.copy()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"  - {col}: {len(le.classes_)} categories")
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train mortality rate: {y_train.mean()*100:.2f}%")
    print(f"Test mortality rate: {y_test.mean()*100:.2f}%")
    
    # Feature scaling for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_encoded.columns.tolist()


# =====================================================================
# 2. MODEL TRAINING
# =====================================================================

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model"""
    print("\n" + "=" * 70)
    print("MODEL 1: XGBoost (BEST PERFORMANCE)")
    print("=" * 70)
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nClass imbalance ratio: {scale_pos_weight:.2f}")
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'random_state': RANDOM_STATE,
        'n_estimators': 200,
        'early_stopping_rounds': 20
    }
    
    print(f"\nTraining XGBoost with parameters:")
    for key, val in params.items():
        print(f"  - {key}: {val}")
    
    # Train model
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    print(f"\nBest iteration: {xgb_model.best_iteration}")
    
    return xgb_model


def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("\n" + "=" * 70)
    print("MODEL 2: Random Forest (GOOD BALANCE)")
    print("=" * 70)
    
    # Calculate class weights
    n_samples = len(y_train)
    n_positive = (y_train == 1).sum()
    n_negative = (y_train == 0).sum()
    
    class_weight = {
        0: n_samples / (2 * n_negative),
        1: n_samples / (2 * n_positive)
    }
    
    print(f"\nClass weights: {class_weight}")
    
    # Train model
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    
    print("\nTraining Random Forest...")
    rf_model.fit(X_train, y_train)
    print("Training complete!")
    
    return rf_model


def train_logistic_regression(X_train_scaled, y_train):
    """Train Logistic Regression model"""
    print("\n" + "=" * 70)
    print("MODEL 3: Logistic Regression (MOST INTERPRETABLE)")
    print("=" * 70)
    
    # Calculate class weights
    n_samples = len(y_train)
    n_positive = (y_train == 1).sum()
    n_negative = (y_train == 0).sum()
    
    class_weight = {
        0: n_samples / (2 * n_negative),
        1: n_samples / (2 * n_positive)
    }
    
    print(f"\nClass weights: {class_weight}")
    
    # Train model
    lr_model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver='lbfgs'
    )
    
    print("\nTraining Logistic Regression...")
    lr_model.fit(X_train_scaled, y_train)
    print("Training complete!")
    
    return lr_model


# =====================================================================
# 3. MODEL EVALUATION
# =====================================================================

def evaluate_model(model, X_test, y_test, model_name, X_test_scaled=None):
    """Comprehensive model evaluation"""
    print(f"\n{'='*70}")
    print(f"EVALUATION: {model_name}")
    print('='*70)
    
    # Use scaled data for logistic regression
    X_eval = X_test_scaled if X_test_scaled is not None else X_test
    
    # Predictions
    y_pred_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  • AUC-ROC:    {auc:.4f}")
    print(f"  • Accuracy:   {accuracy:.4f}")
    print(f"  • Precision:  {precision:.4f}")
    print(f"  • Recall:     {recall:.4f}")
    print(f"  • F1-Score:   {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📋 Confusion Matrix:")
    print(f"              Predicted")
    print(f"              0      1")
    print(f"  Actual 0  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"         1  {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Classification report
    print(f"\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Survived', 'Died']))
    
    return {
        'model_name': model_name,
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred_proba': y_pred_proba
    }


def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """Plot feature importance"""
    print(f"\n{'='*70}")
    print(f"FEATURE IMPORTANCE: {model_name}")
    print('='*70)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Feature importance not available for this model.")
        return
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} most important features:")
    print(importance_df.head(top_n).to_string(index=False))
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=importance_df.head(top_n),
        x='importance',
        y='feature',
        palette='viridis'
    )
    plt.title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n💾 Feature importance plot saved: {filename}")
    plt.close()
    
    return importance_df


def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models"""
    print(f"\n{'='*70}")
    print("ROC CURVES COMPARISON")
    print('='*70)
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, result in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        auc = result['auc']
        
        plt.plot(
            fpr, tpr,
            color=colors[i],
            lw=2,
            label=f"{result['model_name']} (AUC = {auc:.3f})"
        )
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - ICU Mortality Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = "roc_curves_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n💾 ROC curves saved: {filename}")
    plt.close()


def plot_model_comparison(results):
    """Plot comparison of model metrics"""
    print(f"\n{'='*70}")
    print("MODEL COMPARISON SUMMARY")
    print('='*70)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'AUC': r['auc'],
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1-Score': r['f1']
        }
        for r in results
    ])
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        sns.barplot(
            data=comparison_df,
            x='Model',
            y=metric,
            ax=ax,
            palette='Set2'
        )
        
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(comparison_df[metric]):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    # Hide the last subplot
    axes[1, 2].axis('off')
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save plot
    filename = "model_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n💾 Model comparison plot saved: {filename}")
    plt.close()
    
    # Identify best model
    best_model = comparison_df.loc[comparison_df['AUC'].idxmax(), 'Model']
    best_auc = comparison_df['AUC'].max()
    
    print(f"\n🏆 BEST MODEL: {best_model} (AUC = {best_auc:.4f})")
    
    return comparison_df


# =====================================================================
# 4. MAIN PIPELINE
# =====================================================================

def main():
    """Main execution pipeline"""
    print("\n" + "="*70)
    print("ICU MORTALITY PREDICTION - COMPLETE PIPELINE")
    print("="*70)
    
    # File path
    filepath = "../resources_p3/df_a3_andrea_v2.csv"
    
    # 1. Load and preprocess data
    df = load_and_preprocess_data(filepath)
    df_model = feature_engineering(df)
    
    # 2. Prepare train-test split
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_names = \
        prepare_train_test_split(df_model)
    
    # Get label encoders and scaler for API
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    X = df_model.drop('MORTALITY', axis=1)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(X[col].astype(str))
        label_encoders[col] = le
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # 3. Train models
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    rf_model = train_random_forest(X_train, y_train)
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    
    # 4. Evaluate models
    results = []
    
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    results.append(xgb_results)
    
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results.append(rf_results)
    
    lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression", X_test_scaled)
    results.append(lr_results)
    
    # 5. Feature importance
    plot_feature_importance(xgb_model, feature_names, "XGBoost", top_n=20)
    plot_feature_importance(rf_model, feature_names, "Random Forest", top_n=20)
    plot_feature_importance(lr_model, feature_names, "Logistic Regression", top_n=20)
    
    # 6. Visualizations
    plot_roc_curves(results, y_test)
    comparison_df = plot_model_comparison(results)
    
    # 7. Save models for API
    save_models_for_api(xgb_model, rf_model, lr_model, feature_names, label_encoders, scaler)
    
    # 8. Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\n📁 Generated files:")
    print("  • feature_importance_xgboost.png")
    print("  • feature_importance_random_forest.png")
    print("  • feature_importance_logistic_regression.png")
    print("  • roc_curves_comparison.png")
    print("  • model_comparison.png")
    print("\n✅ All models trained and evaluated successfully!")
    print("\n🚀 API models saved and ready for deployment!")
    print("   Run: cd api && uvicorn mlapi:app --reload --port 8000")
    
    return xgb_model, rf_model, lr_model, results, comparison_df


def save_models_for_api(xgb_model, rf_model, lr_model, feature_names, label_encoders, scaler):
    """Save trained models with all components for API deployment"""
    import joblib
    from pathlib import Path
    
    models_dir = Path(__file__).parent / "api" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving models to: {models_dir}")
    
    # Save model bundles
    bundles = {
        "xgboost.pkl": {"model": xgb_model, "feature_names": feature_names, 
                        "label_encoders": label_encoders, "scaler": None},
        "random_forest.pkl": {"model": rf_model, "feature_names": feature_names,
                              "label_encoders": label_encoders, "scaler": None},
        "logistic_regression.pkl": {"model": lr_model, "feature_names": feature_names,
                                    "label_encoders": label_encoders, "scaler": scaler}
    }
    
    for filename, bundle in bundles.items():
        joblib.dump(bundle, models_dir / filename)
        print(f"✅ Saved: {filename}")
    
    return models_dir


if __name__ == "__main__":
    xgb_model, rf_model, lr_model, results, comparison_df = main()
