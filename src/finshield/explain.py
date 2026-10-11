"""
Model explainability using SHAP for FinShield fraud detection.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path


def generate_shap_explanations(model, X_sample, feature_names, max_display=20, 
                               output_dir='reports/figures'):
    """
    Generate SHAP explanations for the fraud detection model.
    
    Parameters:
    -----------
    model : trained model
        XGBoost model
    X_sample : pd.DataFrame or np.array
        Sample data for SHAP analysis (typically a subset of test data)
    feature_names : list
        List of feature names
    max_display : int
        Maximum number of features to display
    output_dir : str
        Directory to save SHAP visualizations
    """
    print("\n" + "="*60)
    print("GENERATING SHAP EXPLANATIONS")
    print("="*60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nComputing SHAP values for {len(X_sample)} samples...")
    print("This may take a few minutes...")
    
    # Create SHAP explainer (TreeExplainer is fast for tree-based models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    print("SHAP values computed successfully!")
    
    # 1. Summary Plot (Feature Importance)
    print("\nGenerating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                     max_display=max_display, show=False)
    plt.tight_layout()
    summary_path = output_path / 'shap_summary_plot.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"SHAP summary plot saved to: {summary_path}")
    plt.close()
    
    # 2. Bar Plot (Mean absolute SHAP values)
    print("Generating SHAP bar plot...")
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    bar_path = output_path / 'shap_bar_plot.png'
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    print(f"SHAP bar plot saved to: {bar_path}")
    plt.close()
    
    # 3. Waterfall plot for a fraud case
    print("Generating SHAP waterfall plot for fraud example...")
    # Find a fraud case
    if isinstance(X_sample, pd.DataFrame):
        fraud_idx = None
        # We need the actual labels to find fraud, so we'll use the first sample
        fraud_idx = 0  # Just use first sample as example
    else:
        fraud_idx = 0
    
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_values[fraud_idx], max_display=15, show=False)
    plt.tight_layout()
    waterfall_path = output_path / 'shap_waterfall_plot.png'
    plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
    print(f"SHAP waterfall plot saved to: {waterfall_path}")
    plt.close()
    
    # 4. Force plot (interactive HTML)
    print("Generating SHAP force plot...")
    try:
        shap.initjs()
        force_plot = shap.force_plot(
            explainer.expected_value, 
            shap_values.values[0], 
            X_sample.iloc[0] if isinstance(X_sample, pd.DataFrame) else X_sample[0],
            feature_names=feature_names,
            matplotlib=False
        )
        force_path = output_path / 'shap_force_plot.html'
        shap.save_html(str(force_path), force_plot)
        print(f"SHAP force plot saved to: {force_path}")
    except Exception as e:
        print(f"Could not generate force plot: {e}")
    
    # 5. Dependence plots for top features
    print("\nGenerating SHAP dependence plots for top 3 features...")
    # Get top 3 most important features
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-3:][::-1]
    
    for i, feat_idx in enumerate(top_features_idx):
        feat_name = feature_names[feat_idx]
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feat_idx, 
            shap_values.values, 
            X_sample,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        dep_path = output_path / f'shap_dependence_{i+1}_{feat_name}.png'
        plt.savefig(dep_path, dpi=300, bbox_inches='tight')
        print(f"SHAP dependence plot for '{feat_name}' saved to: {dep_path}")
        plt.close()
    
    print("\n" + "="*60)
    print("SHAP EXPLANATIONS COMPLETE!")
    print(f"All SHAP visualizations saved to: {output_path}")
    print("="*60)
    
    return shap_values, explainer


def explain_single_prediction(model, explainer, X_single, feature_names, 
                              output_dir='reports/figures'):
    """
    Explain a single prediction in detail.
    
    Parameters:
    -----------
    model : trained model
    explainer : SHAP explainer
    X_single : single sample (1D array or DataFrame row)
    feature_names : list of feature names
    output_dir : str
        Directory to save visualization
    """
    # Compute SHAP values
    shap_values = explainer(X_single)
    
    # Get prediction
    prediction = model.predict(X_single)[0]
    probability = model.predict_proba(X_single)[0, 1]
    
    print("\n" + "="*60)
    print("SINGLE PREDICTION EXPLANATION")
    print("="*60)
    print(f"Prediction: {'FRAUD' if prediction == 1 else 'NON-FRAUD'}")
    print(f"Fraud Probability: {probability:.4f}")
    print("="*60)
    
    # Show top contributing features
    shap_vals = shap_values.values[0]
    contributions = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_vals,
        'abs_shap': np.abs(shap_vals)
    }).sort_values('abs_shap', ascending=False).head(10)
    
    print("\nTop 10 Contributing Features:")
    print(contributions[['feature', 'shap_value']].to_string(index=False))
    
    # Generate waterfall plot
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    waterfall_path = output_path / 'shap_single_prediction.png'
    plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
    print(f"\nWaterfall plot saved to: {waterfall_path}")
    plt.close()
    
    return contributions


if __name__ == "__main__":
    from data import load_transaction_data
    from preprocess import FraudPreprocessor, create_time_aware_split
    
    print("="*60)
    print("FINSHIELD MODEL EXPLAINABILITY (SHAP)")
    print("="*60)
    
    # Load model and preprocessor
    print("\nLoading model and preprocessor...")
    model = joblib.load('models/xgboost_baseline_latest.joblib')
    preprocessor = joblib.load('models/preprocessor_latest.joblib')
    
    # Load and prepare data
    print("Loading data...")
    df = load_transaction_data()
    train_df, test_df = create_time_aware_split(df, test_size=0.2)
    
    print("Preprocessing data...")
    X_test, y_test = preprocessor.fit_transform(test_df)
    
    # Use a sample of test data for SHAP (to speed up computation)
    sample_size = 1000
    print(f"\nUsing {sample_size} samples for SHAP analysis...")
    
    # Sample with stratification to include both fraud and non-fraud
    fraud_indices = np.where(y_test == 1)[0]
    non_fraud_indices = np.where(y_test == 0)[0]
    
    # Take 500 fraud and 500 non-fraud
    n_fraud = min(500, len(fraud_indices))
    n_non_fraud = min(500, len(non_fraud_indices))
    
    sample_indices = np.concatenate([
        np.random.choice(fraud_indices, n_fraud, replace=False),
        np.random.choice(non_fraud_indices, n_non_fraud, replace=False)
    ])
    
    X_sample = X_test.iloc[sample_indices] if isinstance(X_test, pd.DataFrame) else X_test[sample_indices]
    y_sample = y_test.iloc[sample_indices] if isinstance(y_test, pd.Series) else y_test[sample_indices]
    
    print(f"Sample contains {(y_sample == 1).sum()} fraud cases and {(y_sample == 0).sum()} non-fraud cases")
    
    # Generate SHAP explanations
    shap_values, explainer = generate_shap_explanations(
        model, 
        X_sample, 
        preprocessor.feature_columns,
        max_display=20
    )
    
    # Explain a specific fraud case
    print("\n" + "="*60)
    print("EXPLAINING A SPECIFIC FRAUD CASE")
    print("="*60)
    fraud_sample_idx = np.where(y_sample == 1)[0][0]
    X_fraud = X_sample.iloc[[fraud_sample_idx]] if isinstance(X_sample, pd.DataFrame) else X_sample[[fraud_sample_idx]]
    
    contributions = explain_single_prediction(
        model, 
        explainer, 
        X_fraud, 
        preprocessor.feature_columns
    )
