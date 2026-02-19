"""
SHAP Explainability Analysis - Vehicle Registration Prediction

This script explains HOW the model makes predictions using SHAP values.
Focus: Show which features have the most impact on vehicle registration predictions.

Uses CatBoost model optimized via Bayesian Optimization (R² = 0.9760, MAE = 306.68)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

try:
    import shap
except ImportError:
    os.system('pip install shap')
    import shap

try:
    from catboost import CatBoostRegressor
except ImportError:
    os.system('pip install catboost')
    from catboost import CatBoostRegressor


def run_shap_analysis():
    """Run clean SHAP explainability analysis."""
    
    print("=" * 60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)
    
    # Paths
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'catboost_model.bin')
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'master_dataset.csv')
    
    # Load model
    print("\n📊 Loading model...")
    model = CatBoostRegressor()
    model.load_model(model_path)
    print("✓ Model loaded")
    
    # Load data
    print("📊 Loading data...")
    df = pd.read_csv(data_path)
    
    # Prepare features (exclude identifiers and target)
    target_col = 'New_Registration'
    exclude_cols = ['Standard_Category', 'Month', 'Year', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    X = X.fillna(X.mean(numeric_only=True))
    y = df[target_col].copy()
    
    print(f"✓ Data loaded: {X.shape[0]} records, {X.shape[1]} features")
    print(f"  Target: {target_col} (Monthly vehicle registrations)")
    
    # Calculate SHAP values
    print("\n🧮 Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print("✓ SHAP values computed")
    
    # Feature importance
    print("\n⭐ TOP 10 MOST IMPORTANT FEATURES")
    print("-" * 60)
    print("These features have the biggest impact on registration predictions:\n")
    
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Impact': np.abs(shap_values).mean(axis=0)
    }).sort_values('Impact', ascending=False)
    
    for i, (_, row) in enumerate(importance.head(10).iterrows(), 1):
        bar = "█" * int(row['Impact'] / importance['Impact'].max() * 30)
        print(f"{i:2d}. {row['Feature']:30s} {bar} {row['Impact']:.4f}")
    
    # Create visualization
    print("\n📈 Creating visualization...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP Summary - Vehicle Registration Feature Importance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(model_path), 'shap_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualization saved: {output_path}")
    
    # Explain one prediction
    print("\n🎯 EXPLAINING ONE REGISTRATION PREDICTION")
    print("-" * 60)
    
    idx = 0
    pred = model.predict(X.iloc[[idx]])[0]
    baseline = explainer.expected_value
    shap_vals = shap_values[idx]
    features = X.iloc[idx]
    
    print(f"\nMonth Record #{idx}")
    print(f"Predicted New Registrations: {pred:.0f} vehicles")
    print(f"Average Monthly Registrations: {baseline:.0f} vehicles")
    print(f"Difference from Average: {pred - baseline:+.0f} vehicles\n")
    
    # Top contributors
    contributions = pd.DataFrame({
        'Feature': X.columns,
        'Value': features.values,
        'Impact': shap_vals
    }).sort_values('Impact', ascending=False)
    
    print("Top factors INCREASING registrations:")
    for _, row in contributions[contributions['Impact'] > 0].head(3).iterrows():
        print(f"  ↑ {row['Feature']:30s} (+{row['Impact']:.0f} vehicles)")
    
    print("\nTop factors DECREASING registrations:")
    for _, row in contributions[contributions['Impact'] < 0].head(3).iterrows():
        print(f"  ↓ {row['Feature']:30s} ({row['Impact']:.0f} vehicles)")
    
    print("\n" + "=" * 60)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    run_shap_analysis()
