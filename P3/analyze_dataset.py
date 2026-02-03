"""
Comprehensive ICU Mortality Dataset Analysis
This script analyzes mortality patterns across various demographic and clinical factors
in the ICU dataset and generates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# File paths
DATA_PATH = "/Users/arriazui/Desktop/master/ELECTRONIC_HEALTH_RECORDS/P3/resources_p3/df_icu.csv"
OUTPUT_DIR = "/Users/arriazui/Desktop/master/ELECTRONIC_HEALTH_RECORDS/P3/resources_p3/dataset_analysis_images"

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load data
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# Basic statistics
print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Total patients: {len(df)}")
print(f"Total deaths: {df['MORTALITY'].sum()}")
print(f"Overall mortality rate: {df['MORTALITY'].mean()*100:.2f}%\n")

# ============================================================================
# 1. MORTALITY RATE OVERVIEW
# ============================================================================
print("Generating mortality overview...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Overall mortality pie chart
mortality_counts = df['MORTALITY'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].pie(mortality_counts, labels=['Survived', 'Died'], autopct='%1.1f%%', 
           colors=colors, startangle=90)
axes[0].set_title(f'Overall Mortality Distribution\nTotal Patients: {len(df)}', 
                 fontsize=14, fontweight='bold')

# Mortality count bar
axes[1].bar(['Survived', 'Died'], mortality_counts.values, color=colors, alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Number of Patients', fontsize=12)
axes[1].set_title('Patient Count by Outcome', fontsize=14, fontweight='bold')
for i, v in enumerate(mortality_counts.values):
    axes[1].text(i, v + 50, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_overall_mortality.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. MORTALITY BY AGE
# ============================================================================
print("Analyzing mortality by age...")

# Age statistics
print("\nAge Statistics:")
print(df['AGE'].describe())

# Create age groups
df['age_group'] = pd.cut(df['AGE'], 
                         bins=[0, 30, 40, 50, 60, 70, 80, 120],
                         labels=['<30', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'])

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Age distribution
axes[0, 0].hist(df['AGE'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Age (years)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Age Distribution of All Patients', fontsize=14, fontweight='bold')
axes[0, 0].axvline(df['AGE'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["AGE"].mean():.1f}')
axes[0, 0].legend()

# Mortality rate by age group
age_mortality = df.groupby('age_group')['MORTALITY'].agg(['sum', 'count', 'mean'])
age_mortality['mortality_rate'] = age_mortality['mean'] * 100
axes[0, 1].bar(age_mortality.index.astype(str), age_mortality['mortality_rate'], 
              color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Age Group', fontsize=12)
axes[0, 1].set_ylabel('Mortality Rate (%)', fontsize=12)
axes[0, 1].set_title('Mortality Rate by Age Group', fontsize=14, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)
for i, v in enumerate(age_mortality['mortality_rate']):
    axes[0, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

# Age distribution: survivors vs deceased
axes[1, 0].hist([df[df['MORTALITY']==0]['AGE'], df[df['MORTALITY']==1]['AGE']], 
               bins=30, label=['Survived', 'Died'], color=['green', 'red'], 
               alpha=0.6, edgecolor='black')
axes[1, 0].set_xlabel('Age (years)', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Age Distribution: Survivors vs Deceased', fontsize=14, fontweight='bold')
axes[1, 0].legend()

# Patient count by age group
age_counts = df.groupby('age_group').size()
axes[1, 1].bar(age_counts.index.astype(str), age_counts.values, 
              color='lightblue', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Age Group', fontsize=12)
axes[1, 1].set_ylabel('Number of Patients', fontsize=12)
axes[1, 1].set_title('Patient Count by Age Group', fontsize=14, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)
for i, v in enumerate(age_counts.values):
    axes[1, 1].text(i, v + 20, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_mortality_by_age.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. MORTALITY BY GENDER
# ============================================================================
print("Analyzing mortality by gender...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Gender distribution
gender_counts = df['gender'].value_counts()
axes[0].bar(gender_counts.index, gender_counts.values, color=['lightblue', 'pink'], 
           alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Gender', fontsize=12)
axes[0].set_ylabel('Number of Patients', fontsize=12)
axes[0].set_title('Patient Distribution by Gender', fontsize=14, fontweight='bold')
for i, v in enumerate(gender_counts.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

# Mortality rate by gender
gender_mortality = df.groupby('gender')['MORTALITY'].agg(['sum', 'count', 'mean'])
gender_mortality['mortality_rate'] = gender_mortality['mean'] * 100
axes[1].bar(gender_mortality.index, gender_mortality['mortality_rate'], 
           color=['lightblue', 'pink'], alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Gender', fontsize=12)
axes[1].set_ylabel('Mortality Rate (%)', fontsize=12)
axes[1].set_title('Mortality Rate by Gender', fontsize=14, fontweight='bold')
for i, v in enumerate(gender_mortality['mortality_rate']):
    axes[1].text(i, v + 0.3, f'{v:.2f}%', ha='center', fontweight='bold')

# Stacked bar chart
gender_outcome = pd.crosstab(df['gender'], df['MORTALITY'])
gender_outcome.plot(kind='bar', stacked=True, ax=axes[2], color=['green', 'red'], 
                   alpha=0.7, edgecolor='black')
axes[2].set_xlabel('Gender', fontsize=12)
axes[2].set_ylabel('Number of Patients', fontsize=12)
axes[2].set_title('Patient Outcomes by Gender', fontsize=14, fontweight='bold')
axes[2].legend(['Survived', 'Died'], loc='upper right')
axes[2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_mortality_by_gender.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. MORTALITY BY ETHNICITY
# ============================================================================
print("Analyzing mortality by ethnicity...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Ethnicity distribution
ethnicity_counts = df['ethnicity_group'].value_counts()
axes[0, 0].barh(ethnicity_counts.index, ethnicity_counts.values, 
               color='lightgreen', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Number of Patients', fontsize=12)
axes[0, 0].set_ylabel('Ethnicity', fontsize=12)
axes[0, 0].set_title('Patient Distribution by Ethnicity', fontsize=14, fontweight='bold')
for i, v in enumerate(ethnicity_counts.values):
    axes[0, 0].text(v + 50, i, str(v), va='center', fontweight='bold')

# Mortality rate by ethnicity
ethnicity_mortality = df.groupby('ethnicity_group')['MORTALITY'].agg(['sum', 'count', 'mean'])
ethnicity_mortality['mortality_rate'] = ethnicity_mortality['mean'] * 100
ethnicity_mortality = ethnicity_mortality.sort_values('mortality_rate', ascending=False)
axes[0, 1].barh(ethnicity_mortality.index, ethnicity_mortality['mortality_rate'], 
               color='salmon', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Mortality Rate (%)', fontsize=12)
axes[0, 1].set_ylabel('Ethnicity', fontsize=12)
axes[0, 1].set_title('Mortality Rate by Ethnicity', fontsize=14, fontweight='bold')
for i, v in enumerate(ethnicity_mortality['mortality_rate']):
    axes[0, 1].text(v + 0.3, i, f'{v:.2f}%', va='center', fontweight='bold')

# Stacked bar chart by ethnicity
ethnicity_outcome = pd.crosstab(df['ethnicity_group'], df['MORTALITY'])
ethnicity_outcome.plot(kind='barh', stacked=True, ax=axes[1, 0], 
                      color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Number of Patients', fontsize=12)
axes[1, 0].set_ylabel('Ethnicity', fontsize=12)
axes[1, 0].set_title('Patient Outcomes by Ethnicity', fontsize=14, fontweight='bold')
axes[1, 0].legend(['Survived', 'Died'], loc='lower right')

# Ethnicity pie chart
axes[1, 1].pie(ethnicity_counts.values, labels=ethnicity_counts.index, autopct='%1.1f%%',
              startangle=90)
axes[1, 1].set_title('Ethnicity Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_mortality_by_ethnicity.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. MORTALITY BY COMORBIDITIES
# ============================================================================
print("Analyzing mortality by comorbidities...")

comorbidity_cols = ['flag_diabetes', 'flag_hypertension', 'flag_ckd', 
                    'flag_chf', 'flag_copd', 'flag_cancer']
comorbidity_names = ['Diabetes', 'Hypertension', 'CKD', 'CHF', 'COPD', 'Cancer']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (col, name) in enumerate(zip(comorbidity_cols, comorbidity_names)):
    # Calculate mortality rate for patients with and without the condition
    mortality_with = df[df[col] == 1]['MORTALITY'].mean() * 100
    mortality_without = df[df[col] == 0]['MORTALITY'].mean() * 100
    
    count_with = df[df[col] == 1].shape[0]
    count_without = df[df[col] == 0].shape[0]
    
    # Create grouped bar chart
    categories = [f'With {name}\n(n={count_with})', f'Without {name}\n(n={count_without})']
    mortality_rates = [mortality_with, mortality_without]
    
    colors_comorb = ['#e74c3c' if mortality_with > mortality_without else '#3498db',
                     '#3498db' if mortality_with > mortality_without else '#e74c3c']
    
    axes[idx].bar(categories, mortality_rates, color=colors_comorb, alpha=0.7, edgecolor='black')
    axes[idx].set_ylabel('Mortality Rate (%)', fontsize=11)
    axes[idx].set_title(f'{name} Impact on Mortality', fontsize=13, fontweight='bold')
    axes[idx].tick_params(axis='x', rotation=15, labelsize=9)
    
    for i, v in enumerate(mortality_rates):
        axes[idx].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_mortality_by_comorbidities.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. COMORBIDITY PREVALENCE
# ============================================================================
print("Analyzing comorbidity prevalence...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Prevalence of each comorbidity
comorbidity_prevalence = []
for col in comorbidity_cols:
    prevalence = (df[col].sum() / len(df)) * 100
    comorbidity_prevalence.append(prevalence)

axes[0].barh(comorbidity_names, comorbidity_prevalence, color='teal', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Prevalence (%)', fontsize=12)
axes[0].set_title('Comorbidity Prevalence in ICU Population', fontsize=14, fontweight='bold')
for i, v in enumerate(comorbidity_prevalence):
    axes[0].text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')

# Number of comorbidities per patient
df['num_comorbidities'] = df[comorbidity_cols].sum(axis=1)
comorbidity_counts = df['num_comorbidities'].value_counts().sort_index()
axes[1].bar(comorbidity_counts.index, comorbidity_counts.values, 
           color='purple', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Number of Comorbidities', fontsize=12)
axes[1].set_ylabel('Number of Patients', fontsize=12)
axes[1].set_title('Distribution of Comorbidity Count', fontsize=14, fontweight='bold')
for i, v in enumerate(comorbidity_counts.values):
    axes[1].text(comorbidity_counts.index[i], v + 50, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_comorbidity_prevalence.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. MORTALITY BY NUMBER OF COMORBIDITIES
# ============================================================================
print("Analyzing mortality by comorbidity burden...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Mortality rate by number of comorbidities
comorbidity_mortality = df.groupby('num_comorbidities')['MORTALITY'].agg(['sum', 'count', 'mean'])
comorbidity_mortality['mortality_rate'] = comorbidity_mortality['mean'] * 100

axes[0].plot(comorbidity_mortality.index, comorbidity_mortality['mortality_rate'], 
            marker='o', linewidth=2, markersize=10, color='red')
axes[0].set_xlabel('Number of Comorbidities', fontsize=12)
axes[0].set_ylabel('Mortality Rate (%)', fontsize=12)
axes[0].set_title('Mortality Rate vs Number of Comorbidities', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
for i, v in zip(comorbidity_mortality.index, comorbidity_mortality['mortality_rate']):
    axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

# Stacked bar: outcomes by comorbidity count
comorbidity_outcome = pd.crosstab(df['num_comorbidities'], df['MORTALITY'])
comorbidity_outcome.plot(kind='bar', stacked=True, ax=axes[1], 
                        color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Number of Comorbidities', fontsize=12)
axes[1].set_ylabel('Number of Patients', fontsize=12)
axes[1].set_title('Patient Outcomes by Comorbidity Count', fontsize=14, fontweight='bold')
axes[1].legend(['Survived', 'Died'], loc='upper left')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_mortality_by_comorbidity_count.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. MORTALITY BY DIAGNOSIS GROUP
# ============================================================================
print("Analyzing mortality by diagnosis group...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Diagnosis distribution
diagnosis_counts = df['diagnosis_group'].value_counts()
axes[0, 0].barh(diagnosis_counts.index, diagnosis_counts.values, 
               color='orange', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Number of Patients', fontsize=12)
axes[0, 0].set_ylabel('Diagnosis Group', fontsize=12)
axes[0, 0].set_title('Patient Distribution by Diagnosis Group', fontsize=14, fontweight='bold')
for i, v in enumerate(diagnosis_counts.values):
    axes[0, 0].text(v + 50, i, str(v), va='center', fontweight='bold')

# Mortality rate by diagnosis
diagnosis_mortality = df.groupby('diagnosis_group')['MORTALITY'].agg(['sum', 'count', 'mean'])
diagnosis_mortality['mortality_rate'] = diagnosis_mortality['mean'] * 100
diagnosis_mortality = diagnosis_mortality.sort_values('mortality_rate', ascending=False)
axes[0, 1].barh(diagnosis_mortality.index, diagnosis_mortality['mortality_rate'], 
               color='darkred', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Mortality Rate (%)', fontsize=12)
axes[0, 1].set_ylabel('Diagnosis Group', fontsize=12)
axes[0, 1].set_title('Mortality Rate by Diagnosis Group', fontsize=14, fontweight='bold')
for i, v in enumerate(diagnosis_mortality['mortality_rate']):
    axes[0, 1].text(v + 0.5, i, f'{v:.2f}%', va='center', fontweight='bold')

# Stacked bar chart
diagnosis_outcome = pd.crosstab(df['diagnosis_group'], df['MORTALITY'])
diagnosis_outcome.plot(kind='barh', stacked=True, ax=axes[1, 0], 
                      color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Number of Patients', fontsize=12)
axes[1, 0].set_ylabel('Diagnosis Group', fontsize=12)
axes[1, 0].set_title('Patient Outcomes by Diagnosis Group', fontsize=14, fontweight='bold')
axes[1, 0].legend(['Survived', 'Died'], loc='lower right')

# Diagnosis pie chart
axes[1, 1].pie(diagnosis_counts.values, labels=diagnosis_counts.index, autopct='%1.1f%%',
              startangle=90)
axes[1, 1].set_title('Diagnosis Group Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_mortality_by_diagnosis.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 9. MORTALITY BY BMI
# ============================================================================
print("Analyzing mortality by BMI...")

# Remove NA values for BMI analysis
df_bmi = df[df['bmi_baseline'].notna()].copy()

# Create BMI categories
df_bmi['bmi_category'] = pd.cut(df_bmi['bmi_baseline'], 
                                bins=[0, 18.5, 25, 30, 35, 100],
                                labels=['Underweight\n(<18.5)', 
                                       'Normal\n(18.5-25)', 
                                       'Overweight\n(25-30)', 
                                       'Obese\n(30-35)', 
                                       'Severely Obese\n(>35)'])

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# BMI distribution
axes[0, 0].hist(df_bmi['bmi_baseline'], bins=40, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('BMI', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('BMI Distribution', fontsize=14, fontweight='bold')
axes[0, 0].axvline(df_bmi['bmi_baseline'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df_bmi["bmi_baseline"].mean():.1f}')
axes[0, 0].legend()

# Mortality rate by BMI category
bmi_mortality = df_bmi.groupby('bmi_category')['MORTALITY'].agg(['sum', 'count', 'mean'])
bmi_mortality['mortality_rate'] = bmi_mortality['mean'] * 100
axes[0, 1].bar(range(len(bmi_mortality)), bmi_mortality['mortality_rate'], 
              color='indianred', alpha=0.7, edgecolor='black')
axes[0, 1].set_xticks(range(len(bmi_mortality)))
axes[0, 1].set_xticklabels(bmi_mortality.index, rotation=15)
axes[0, 1].set_ylabel('Mortality Rate (%)', fontsize=12)
axes[0, 1].set_title('Mortality Rate by BMI Category', fontsize=14, fontweight='bold')
for i, v in enumerate(bmi_mortality['mortality_rate']):
    axes[0, 1].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold')

# BMI distribution: survivors vs deceased
axes[1, 0].hist([df_bmi[df_bmi['MORTALITY']==0]['bmi_baseline'], 
                df_bmi[df_bmi['MORTALITY']==1]['bmi_baseline']], 
               bins=30, label=['Survived', 'Died'], color=['green', 'red'], 
               alpha=0.6, edgecolor='black')
axes[1, 0].set_xlabel('BMI', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('BMI Distribution: Survivors vs Deceased', fontsize=14, fontweight='bold')
axes[1, 0].legend()

# Patient count by BMI category
bmi_counts = df_bmi.groupby('bmi_category').size()
axes[1, 1].bar(range(len(bmi_counts)), bmi_counts.values, 
              color='lightblue', alpha=0.7, edgecolor='black')
axes[1, 1].set_xticks(range(len(bmi_counts)))
axes[1, 1].set_xticklabels(bmi_counts.index, rotation=15)
axes[1, 1].set_ylabel('Number of Patients', fontsize=12)
axes[1, 1].set_title('Patient Count by BMI Category', fontsize=14, fontweight='bold')
for i, v in enumerate(bmi_counts.values):
    axes[1, 1].text(i, v + 20, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_mortality_by_bmi.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 10. MORTALITY BY ICU LENGTH OF STAY
# ============================================================================
print("Analyzing mortality by ICU length of stay...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ICU LOS distribution (capped for visualization)
los_capped = df['icu_los_hours'].clip(upper=500)
axes[0, 0].hist(los_capped, bins=50, color='teal', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('ICU Length of Stay (hours)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('ICU Length of Stay Distribution', fontsize=14, fontweight='bold')
axes[0, 0].axvline(df['icu_los_hours'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["icu_los_hours"].mean():.1f}h')
axes[0, 0].legend()

# LOS categories
df['los_category'] = pd.cut(df['icu_los_hours'], 
                            bins=[0, 24, 48, 72, 168, 10000],
                            labels=['<24h', '24-48h', '48-72h', '72h-1wk', '>1wk'])

# Mortality rate by LOS category
los_mortality = df.groupby('los_category')['MORTALITY'].agg(['sum', 'count', 'mean'])
los_mortality['mortality_rate'] = los_mortality['mean'] * 100
axes[0, 1].bar(los_mortality.index.astype(str), los_mortality['mortality_rate'], 
              color='darkslategray', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('ICU Length of Stay', fontsize=12)
axes[0, 1].set_ylabel('Mortality Rate (%)', fontsize=12)
axes[0, 1].set_title('Mortality Rate by ICU Length of Stay', fontsize=14, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=15)
for i, v in enumerate(los_mortality['mortality_rate']):
    axes[0, 1].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold')

# LOS: survivors vs deceased (capped)
axes[1, 0].hist([df[df['MORTALITY']==0]['icu_los_hours'].clip(upper=500), 
                df[df['MORTALITY']==1]['icu_los_hours'].clip(upper=500)], 
               bins=40, label=['Survived', 'Died'], color=['green', 'red'], 
               alpha=0.6, edgecolor='black')
axes[1, 0].set_xlabel('ICU Length of Stay (hours, capped at 500)', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('ICU LOS: Survivors vs Deceased', fontsize=14, fontweight='bold')
axes[1, 0].legend()

# Patient count by LOS category
los_counts = df.groupby('los_category').size()
axes[1, 1].bar(los_counts.index.astype(str), los_counts.values, 
              color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('ICU Length of Stay', fontsize=12)
axes[1, 1].set_ylabel('Number of Patients', fontsize=12)
axes[1, 1].set_title('Patient Count by ICU Length of Stay', fontsize=14, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=15)
for i, v in enumerate(los_counts.values):
    axes[1, 1].text(i, v + 50, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_mortality_by_icu_los.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 11. MORTALITY BY SOFA SCORE
# ============================================================================
print("Analyzing mortality by SOFA score...")

df_sofa = df[df['total_sofa'].notna()].copy()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# SOFA score distribution
axes[0, 0].hist(df_sofa['total_sofa'], bins=20, color='crimson', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Total SOFA Score', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('SOFA Score Distribution', fontsize=14, fontweight='bold')
axes[0, 0].axvline(df_sofa['total_sofa'].mean(), color='darkred', linestyle='--', 
                   label=f'Mean: {df_sofa["total_sofa"].mean():.1f}')
axes[0, 0].legend()

# SOFA categories
df_sofa['sofa_category'] = pd.cut(df_sofa['total_sofa'], 
                                  bins=[0, 5, 10, 15, 25],
                                  labels=['Low (0-5)', 'Moderate (6-10)', 
                                         'High (11-15)', 'Very High (>15)'])

# Mortality rate by SOFA category
sofa_mortality = df_sofa.groupby('sofa_category')['MORTALITY'].agg(['sum', 'count', 'mean'])
sofa_mortality['mortality_rate'] = sofa_mortality['mean'] * 100
axes[0, 1].bar(sofa_mortality.index.astype(str), sofa_mortality['mortality_rate'], 
              color='maroon', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('SOFA Score Category', fontsize=12)
axes[0, 1].set_ylabel('Mortality Rate (%)', fontsize=12)
axes[0, 1].set_title('Mortality Rate by SOFA Score', fontsize=14, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=15)
for i, v in enumerate(sofa_mortality['mortality_rate']):
    axes[0, 1].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')

# SOFA: survivors vs deceased
axes[1, 0].hist([df_sofa[df_sofa['MORTALITY']==0]['total_sofa'], 
                df_sofa[df_sofa['MORTALITY']==1]['total_sofa']], 
               bins=20, label=['Survived', 'Died'], color=['green', 'red'], 
               alpha=0.6, edgecolor='black')
axes[1, 0].set_xlabel('Total SOFA Score', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('SOFA Score: Survivors vs Deceased', fontsize=14, fontweight='bold')
axes[1, 0].legend()

# Scatter: SOFA vs Age colored by mortality
scatter = axes[1, 1].scatter(df_sofa['AGE'], df_sofa['total_sofa'], 
                            c=df_sofa['MORTALITY'], cmap='RdYlGn_r', 
                            alpha=0.5, edgecolor='black', linewidth=0.5)
axes[1, 1].set_xlabel('Age (years)', fontsize=12)
axes[1, 1].set_ylabel('Total SOFA Score', fontsize=12)
axes[1, 1].set_title('SOFA Score vs Age (colored by mortality)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=axes[1, 1])
cbar.set_label('Mortality', fontsize=11)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Survived', 'Died'])

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_mortality_by_sofa.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 12. MORTALITY BY VASOPRESSOR USE
# ============================================================================
print("Analyzing mortality by vasopressor use...")

# Check if vasopressor columns exist
if 'any_vaso' in df.columns:
    df_vaso = df[df['any_vaso'].notna()].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Vasopressor use distribution
    vaso_counts = df_vaso['any_vaso'].value_counts()
    axes[0].bar(['No Vasopressor', 'Vasopressor Used'], vaso_counts.values, 
               color=['lightgreen', 'lightcoral'], alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Number of Patients', fontsize=12)
    axes[0].set_title('Vasopressor Use Distribution', fontsize=14, fontweight='bold')
    for i, v in enumerate(vaso_counts.values):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # Mortality rate by vasopressor use
    vaso_mortality = df_vaso.groupby('any_vaso')['MORTALITY'].agg(['sum', 'count', 'mean'])
    vaso_mortality['mortality_rate'] = vaso_mortality['mean'] * 100
    axes[1].bar(['No Vasopressor', 'Vasopressor Used'], vaso_mortality['mortality_rate'].values, 
               color=['lightgreen', 'lightcoral'], alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Mortality Rate (%)', fontsize=12)
    axes[1].set_title('Mortality Rate by Vasopressor Use', fontsize=14, fontweight='bold')
    for i, v in enumerate(vaso_mortality['mortality_rate'].values):
        axes[1].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')
    
    # Stacked bar
    vaso_outcome = pd.crosstab(df_vaso['any_vaso'], df_vaso['MORTALITY'])
    vaso_outcome.plot(kind='bar', stacked=True, ax=axes[2], 
                     color=['green', 'red'], alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Vasopressor Use', fontsize=12)
    axes[2].set_ylabel('Number of Patients', fontsize=12)
    axes[2].set_title('Patient Outcomes by Vasopressor Use', fontsize=14, fontweight='bold')
    axes[2].legend(['Survived', 'Died'], loc='upper left')
    axes[2].set_xticklabels(['No Vasopressor', 'Vasopressor Used'], rotation=15)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/12_mortality_by_vasopressor.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# 13. CORRELATION HEATMAP
# ============================================================================
print("Generating correlation heatmap...")

# Select numerical columns for correlation
numerical_cols = ['AGE', 'MORTALITY', 'flag_diabetes', 'flag_hypertension', 
                 'flag_ckd', 'flag_chf', 'flag_copd', 'flag_cancer',
                 'icu_los_hours', 'bmi_baseline', 'num_comorbidities']

# Filter to available columns
numerical_cols = [col for col in numerical_cols if col in df.columns]

df_corr = df[numerical_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
           square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix of Key Variables', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 14. MULTI-FACTOR ANALYSIS: AGE, GENDER, COMORBIDITIES
# ============================================================================
print("Generating multi-factor analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Mortality by age and gender
age_gender_mortality = df.groupby(['age_group', 'gender'])['MORTALITY'].mean() * 100
age_gender_mortality = age_gender_mortality.unstack()
age_gender_mortality.plot(kind='bar', ax=axes[0, 0], color=['lightblue', 'pink'], 
                         alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Age Group', fontsize=12)
axes[0, 0].set_ylabel('Mortality Rate (%)', fontsize=12)
axes[0, 0].set_title('Mortality Rate by Age Group and Gender', fontsize=14, fontweight='bold')
axes[0, 0].legend(title='Gender', loc='upper left')
axes[0, 0].tick_params(axis='x', rotation=45)

# Mortality by ethnicity and age
ethnicity_age_counts = df.groupby(['ethnicity_group', 'age_group']).size().unstack(fill_value=0)
ethnicity_age_counts.plot(kind='bar', stacked=True, ax=axes[0, 1], 
                         colormap='viridis', alpha=0.8, edgecolor='black')
axes[0, 1].set_xlabel('Ethnicity', fontsize=12)
axes[0, 1].set_ylabel('Number of Patients', fontsize=12)
axes[0, 1].set_title('Patient Distribution: Ethnicity by Age Group', fontsize=14, fontweight='bold')
axes[0, 1].legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 1].tick_params(axis='x', rotation=45)

# Comorbidities by age group
age_comorbidity = df.groupby('age_group')[comorbidity_cols].mean() * 100
age_comorbidity.plot(kind='bar', ax=axes[1, 0], colormap='Set2', 
                    alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Age Group', fontsize=12)
axes[1, 0].set_ylabel('Prevalence (%)', fontsize=12)
axes[1, 0].set_title('Comorbidity Prevalence by Age Group', fontsize=14, fontweight='bold')
axes[1, 0].legend(comorbidity_names, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
axes[1, 0].tick_params(axis='x', rotation=45)

# Diagnosis by gender
diagnosis_gender = pd.crosstab(df['diagnosis_group'], df['gender'], normalize='columns') * 100
diagnosis_gender.plot(kind='bar', ax=axes[1, 1], color=['lightblue', 'pink'], 
                     alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Diagnosis Group', fontsize=12)
axes[1, 1].set_ylabel('Percentage (%)', fontsize=12)
axes[1, 1].set_title('Diagnosis Distribution by Gender', fontsize=14, fontweight='bold')
axes[1, 1].legend(title='Gender', loc='upper right')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/14_multi_factor_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 15. SUMMARY STATISTICS TABLE
# ============================================================================
print("Generating summary statistics...")

summary_stats = {
    'Total Patients': len(df),
    'Total Deaths': int(df['MORTALITY'].sum()),
    'Overall Mortality Rate (%)': f"{df['MORTALITY'].mean()*100:.2f}",
    'Mean Age (years)': f"{df['AGE'].mean():.1f}",
    'Age Range': f"{df['AGE'].min():.0f} - {df['AGE'].max():.0f}",
    'Male Patients (%)': f"{(df['gender']=='M').sum()/len(df)*100:.1f}",
    'Female Patients (%)': f"{(df['gender']=='F').sum()/len(df)*100:.1f}",
    'Mean ICU LOS (hours)': f"{df['icu_los_hours'].mean():.1f}",
    'Median ICU LOS (hours)': f"{df['icu_los_hours'].median():.1f}",
    'Mean BMI': f"{df['bmi_baseline'].mean():.1f}",
    'Patients with Diabetes (%)': f"{(df['flag_diabetes']==1).sum()/len(df)*100:.1f}",
    'Patients with Hypertension (%)': f"{(df['flag_hypertension']==1).sum()/len(df)*100:.1f}",
}

# Create visualization of summary stats
fig, ax = plt.subplots(figsize=(12, 10))
ax.axis('tight')
ax.axis('off')

table_data = [[key, value] for key, value in summary_stats.items()]
table = ax.table(cellText=table_data, colLabels=['Statistic', 'Value'],
                cellLoc='left', loc='center', colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.5)

# Style header
for i in range(2):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data) + 1):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('white')

plt.title('Dataset Summary Statistics', fontsize=16, fontweight='bold', pad=20)
plt.savefig(f"{OUTPUT_DIR}/15_summary_statistics.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# SAVE DETAILED STATISTICS TO CSV
# ============================================================================
print("Saving detailed statistics to CSV...")

# Age group statistics
age_stats = df.groupby('age_group').agg({
    'MORTALITY': ['count', 'sum', 'mean']
}).round(4)
age_stats.columns = ['Total_Patients', 'Deaths', 'Mortality_Rate']
age_stats['Mortality_Rate_%'] = age_stats['Mortality_Rate'] * 100
age_stats.to_csv(f"{OUTPUT_DIR}/stats_age_groups.csv")

# Gender statistics
gender_stats = df.groupby('gender').agg({
    'MORTALITY': ['count', 'sum', 'mean']
}).round(4)
gender_stats.columns = ['Total_Patients', 'Deaths', 'Mortality_Rate']
gender_stats['Mortality_Rate_%'] = gender_stats['Mortality_Rate'] * 100
gender_stats.to_csv(f"{OUTPUT_DIR}/stats_gender.csv")

# Ethnicity statistics
ethnicity_stats = df.groupby('ethnicity_group').agg({
    'MORTALITY': ['count', 'sum', 'mean']
}).round(4)
ethnicity_stats.columns = ['Total_Patients', 'Deaths', 'Mortality_Rate']
ethnicity_stats['Mortality_Rate_%'] = ethnicity_stats['Mortality_Rate'] * 100
ethnicity_stats.to_csv(f"{OUTPUT_DIR}/stats_ethnicity.csv")

# Diagnosis statistics
diagnosis_stats = df.groupby('diagnosis_group').agg({
    'MORTALITY': ['count', 'sum', 'mean']
}).round(4)
diagnosis_stats.columns = ['Total_Patients', 'Deaths', 'Mortality_Rate']
diagnosis_stats['Mortality_Rate_%'] = diagnosis_stats['Mortality_Rate'] * 100
diagnosis_stats.to_csv(f"{OUTPUT_DIR}/stats_diagnosis.csv")

# Comorbidity statistics
comorbidity_stats = pd.DataFrame({
    'Comorbidity': comorbidity_names,
    'Prevalence_%': comorbidity_prevalence,
    'Mortality_With_%': [df[df[col] == 1]['MORTALITY'].mean() * 100 for col in comorbidity_cols],
    'Mortality_Without_%': [df[df[col] == 0]['MORTALITY'].mean() * 100 for col in comorbidity_cols]
})
comorbidity_stats.to_csv(f"{OUTPUT_DIR}/stats_comorbidities.csv", index=False)

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll visualizations have been saved to:")
print(f"{OUTPUT_DIR}/")
print(f"\nGenerated files:")
print(f"  - 15 visualization images (PNG)")
print(f"  - 5 statistical summary files (CSV)")
print("\nKey Findings:")
print(f"  - Total Patients: {len(df)}")
print(f"  - Overall Mortality Rate: {df['MORTALITY'].mean()*100:.2f}%")
print(f"  - Mean Age: {df['AGE'].mean():.1f} years")
print(f"  - Mean ICU LOS: {df['icu_los_hours'].mean():.1f} hours")
print("\n" + "="*80)
