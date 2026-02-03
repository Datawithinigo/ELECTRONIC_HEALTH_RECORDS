"""
Auto-update README.md with latest model metrics from CSV files
"""
import pandas as pd
from pathlib import Path


def format_table_row(row, has_specificity=False):
    """Format a single model row for markdown table"""
    if has_specificity:
        return (
            f"| {row['Model']} | "
            f"{row['AUC']:.3f} | "
            f"{row['Accuracy']*100:.1f}% | "
            f"{row['Precision']:.3f} | "
            f"{row['Recall']:.3f} | "
            f"{row['Specificity']:.3f} | "
            f"{row['F1-Score']:.3f} |"
        )
    else:
        return (
            f"| {row['Model']} | "
            f"{row['AUC']:.3f} | "
            f"{row['Accuracy']*100:.1f}% | "
            f"{row['Precision']:.3f} | "
            f"{row['Recall']:.3f} | "
            f"{row['F1-Score']:.3f} |"
        )


def generate_metrics_table(csv_path):
    """Generate markdown table from CSV metrics"""
    df = pd.read_csv(csv_path)
    
    # Check if Specificity column exists
    has_specificity = 'Specificity' in df.columns
    
    # Header
    if has_specificity:
        table = "| Model | AUC | Accuracy | Precision | Recall | Specificity | F1-Score |\n"
        table += "|-------|-----|----------|-----------|--------|-------------|----------|\n"
    else:
        table = "| Model | AUC | Accuracy | Precision | Recall | F1-Score |\n"
        table += "|-------|-----|----------|-----------|--------|----------|\n"
    
    # Sort by AUC descending
    df = df.sort_values('AUC', ascending=False)
    
    # Data rows
    for _, row in df.iterrows():
        table += format_table_row(row, has_specificity) + "\n"
    
    return table


def update_readme():
    """Update README.md with latest metrics from CSV files"""
    script_dir = Path(__file__).parent
    readme_path = script_dir.parent / "README.md"
    
    # Read CSV files
    baseline_csv = script_dir / "results_images" / "model_metrics_summary.csv"
    sofa_csv = script_dir / "results_images" / "model_metrics_summary_sofa.csv"
    
    if not baseline_csv.exists():
        print(f"⚠️  Warning: {baseline_csv} not found. Run mortality_prediction_models.py first.")
        return False
    
    if not sofa_csv.exists():
        print(f"⚠️  Warning: {sofa_csv} not found. Run mortality_prediction_models_sofa.py first.")
        return False
    
    # Generate tables
    print("📊 Generating metrics tables from CSV files...")
    baseline_table = generate_metrics_table(baseline_csv)
    sofa_table = generate_metrics_table(sofa_csv)
    
    # Find best model from SOFA table
    sofa_df = pd.read_csv(sofa_csv)
    best_model = sofa_df.loc[sofa_df['AUC'].idxmax()]
    best_summary = f"**Best Model:** {best_model['Model']} with SOFA score (AUC: {best_model['AUC']:.3f}, Accuracy: {best_model['Accuracy']*100:.1f}%)"
    
    # Read current README
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Find and replace baseline table
    baseline_start = content.find("### Baseline Models (Without SOFA Score)")
    baseline_end = content.find("### Enhanced Models (With SOFA Score)")
    
    if baseline_start == -1 or baseline_end == -1:
        print("❌ Error: Could not find baseline models section in README")
        return False
    
    baseline_section = (
        "### Baseline Models (Without SOFA Score)\n\n"
        + baseline_table + "\n"
    )
    
    # Find and replace SOFA table
    sofa_end = content.find("**Best Model:**", baseline_end)
    
    if sofa_end == -1:
        print("❌ Error: Could not find SOFA models section in README")
        return False
    
    sofa_section = (
        "### Enhanced Models (With SOFA Score)\n\n"
        + sofa_table + "\n"
        + best_summary + "\n"
    )
    
    # Reconstruct README
    best_model_end = content.find("\n", sofa_end + 10)
    
    new_content = (
        content[:baseline_start] +
        baseline_section +
        sofa_section +
        content[best_model_end+1:]
    )
    
    # Write updated README
    with open(readme_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ README.md updated successfully!")
    print(f"   📍 Location: {readme_path}")
    print(f"\n🏆 Best Model: {best_model['Model']}")
    print(f"   • AUC: {best_model['AUC']:.3f}")
    print(f"   • Accuracy: {best_model['Accuracy']*100:.1f}%")
    print(f"   • Precision: {best_model['Precision']:.3f}")
    print(f"   • Recall: {best_model['Recall']:.3f}")
    print(f"   • F1-Score: {best_model['F1-Score']:.3f}")
    
    return True


if __name__ == "__main__":
    print("="*70)
    print("README.md AUTO-UPDATER")
    print("="*70)
    print("\nReading model metrics from CSV files...")
    
    success = update_readme()
    
    if success:
        print("\n" + "="*70)
        print("✅ UPDATE COMPLETE")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ UPDATE FAILED")
        print("="*70)
        print("\nMake sure to run both training scripts first:")
        print("  1. python mortality_prediction_models.py")
        print("  2. python mortality_prediction_models_sofa.py")
