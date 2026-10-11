"""
Model evaluation and visualization for FinShield fraud detection.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, 
    auc, 
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
import joblib
from pathlib import Path


def plot_roc_curve(y_true, y_pred_proba, save_path='reports/figures/roc_curve.png'):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred_proba : array
        Predicted probabilities
    save_path : str
        Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to: {save_path}")
    plt.close()


def plot_precision_recall_curve(y_true, y_pred_proba, save_path='reports/figures/precision_recall_curve.png'):
    """
    Plot Precision-Recall curve.
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred_proba : array
        Predicted probabilities
    save_path : str
        Path to save the plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=11)
    plt.grid(alpha=0.3)
    
    # Add baseline (fraud rate)
    fraud_rate = y_true.mean()
    plt.axhline(y=fraud_rate, color='r', linestyle='--', label=f'Baseline (fraud rate = {fraud_rate:.4f})')
    plt.legend(loc="best", fontsize=11)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curve saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path='reports/figures/confusion_matrix.png'):
    """
    Plot confusion matrix heatmap.
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    save_path : str
        Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = (cm[i, j] / total) * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, save_path='reports/figures/feature_importance.png'):
    """
    Plot top N important features.
    
    Parameters:
    -----------
    model : trained model
        XGBoost model
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to plot
    save_path : str
        Path to save the plot
    """
    # Get feature importance
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance_df, y='feature', x='importance', palette='viridis')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {save_path}")
    plt.close()
    
    return feature_importance_df


def plot_prediction_distribution(y_true, y_pred_proba, save_path='reports/figures/prediction_distribution.png'):
    """
    Plot distribution of predicted probabilities for fraud and non-fraud.
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred_proba : array
        Predicted probabilities
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot distributions
    plt.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.6, label='Non-Fraud', color='blue', density=True)
    plt.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.6, label='Fraud', color='red', density=True)
    
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction distribution plot saved to: {save_path}")
    plt.close()


def generate_evaluation_report(model, X_test, y_test, feature_names, output_dir='reports'):
    """
    Generate comprehensive evaluation report with all visualizations.
    
    Parameters:
    -----------
    model : trained model
    X_test : test features
    y_test : test labels
    feature_names : list of feature names
    output_dir : str
        Directory to save reports
    """
    print("\n" + "="*60)
    print("GENERATING EVALUATION REPORT")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create figures directory
    figures_dir = Path(output_dir) / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    print("\nGenerating visualizations...")
    plot_roc_curve(y_test, y_pred_proba, figures_dir / 'roc_curve.png')
    plot_precision_recall_curve(y_test, y_pred_proba, figures_dir / 'precision_recall_curve.png')
    plot_confusion_matrix(y_test, y_pred, figures_dir / 'confusion_matrix.png')
    feature_importance_df = plot_feature_importance(model, feature_names, top_n=20, 
                                                   save_path=figures_dir / 'feature_importance.png')
    plot_prediction_distribution(y_test, y_pred_proba, figures_dir / 'prediction_distribution.png')
    
    # Save feature importance to CSV
    feature_importance_path = Path(output_dir) / 'feature_importance.csv'
    feature_importance_df.to_csv(feature_importance_path, index=False)
    print(f"\nFeature importance saved to: {feature_importance_path}")
    
    print("\n" + "="*60)
    print("EVALUATION REPORT COMPLETE!")
    print(f"All visualizations saved to: {figures_dir}")
    print("="*60)
    
    return {
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'feature_importance': feature_importance_df
    }


if __name__ == "__main__":
    from data import load_transaction_data
    from preprocess import FraudPreprocessor, create_time_aware_split
    
    print("="*60)
    print("FINSHIELD MODEL EVALUATION")
    print("="*60)
    
    # Load the latest model and preprocessor
    print("\nLoading model and preprocessor...")
    model = joblib.load('models/xgboost_baseline_latest.joblib')
    preprocessor = joblib.load('models/preprocessor_latest.joblib')
    
    # Load and prepare test data
    print("\nLoading test data...")
    df = load_transaction_data()
    train_df, test_df = create_time_aware_split(df, test_size=0.2)
    
    print("\nPreprocessing test data...")
    X_test, y_test = preprocessor.fit_transform(test_df)
    
    # Generate evaluation report
    results = generate_evaluation_report(
        model, 
        X_test, 
        y_test, 
        preprocessor.feature_columns
    )
    
    print("\nTop 10 Most Important Features:")
    print(results['feature_importance'].head(10).to_string(index=False))
