"""
Model training script for FinShield fraud detection.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib
from pathlib import Path
import time

from data import load_transaction_data
from preprocess import FraudPreprocessor, create_time_aware_split


def train_xgboost_baseline(X_train, y_train, X_test, y_test):
    """
    Train XGBoost classifier for fraud detection.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    
    Returns:
    --------
    model : trained XGBoost model
    metrics : dict of evaluation metrics
    """
    print("\n" + "="*60)
    print("TRAINING XGBOOST BASELINE MODEL")
    print("="*60)
    
    # Calculate scale_pos_weight for imbalanced data
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance ratio: {fraud_ratio:.2f}")
    
    # XGBoost parameters optimized for fraud detection
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': fraud_ratio,  # Handle imbalance
        'random_state': 42,
        'tree_method': 'hist',  # CPU-efficient
        'n_jobs': -1
    }
    
    print("\nModel parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Initialize and train
    print("\nTraining model...")
    start_time = time.time()
    
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'training_time': training_time
    }
    
    return model, metrics


def print_evaluation_metrics(metrics, y_test):
    """
    Print comprehensive evaluation metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of evaluation metrics
    y_test : array
        True labels
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'ROC-AUC':<20} {metrics['roc_auc']:.4f}")
    print(f"{'Precision':<20} {metrics['precision']:.4f}")
    print(f"{'Recall':<20} {metrics['recall']:.4f}")
    print(f"{'F1-Score':<20} {metrics['f1']:.4f}")
    
    print(f"\n{'Confusion Matrix':<20}")
    print("-" * 35)
    cm = metrics['confusion_matrix']
    print(f"                    Predicted")
    print(f"                 Non-Fraud  Fraud")
    print(f"Actual Non-Fraud   {cm[0][0]:>6}    {cm[0][1]:>5}")
    print(f"Actual Fraud       {cm[1][0]:>6}    {cm[1][1]:>5}")
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{'Detailed Metrics':<20}")
    print("-" * 35)
    print(f"True Negatives:  {tn:,}")
    print(f"False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")
    print(f"True Positives:  {tp:,}")
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\nSpecificity:     {specificity:.4f}")
    print(f"Sensitivity:     {metrics['recall']:.4f}")
    
    # Business metrics
    print(f"\n{'Business Impact':<20}")
    print("-" * 35)
    fraud_detected = tp
    fraud_missed = fn
    fraud_detection_rate = (fraud_detected / (fraud_detected + fraud_missed)) * 100 if (fraud_detected + fraud_missed) > 0 else 0
    false_alarm_rate = (fp / (tn + fp)) * 100 if (tn + fp) > 0 else 0
    
    print(f"Fraud Detection Rate: {fraud_detection_rate:.2f}%")
    print(f"False Alarm Rate:     {false_alarm_rate:.2f}%")
    print(f"Frauds Caught:        {fraud_detected:,} / {fraud_detected + fraud_missed:,}")
    
    print("="*60)


def save_model_and_preprocessor(model, preprocessor, model_dir='models'):
    """
    Save trained model and preprocessor.
    
    Parameters:
    -----------
    model : trained model
    preprocessor : fitted preprocessor
    model_dir : str
        Directory to save model artifacts
    """
    # Create models directory if it doesn't exist
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Save with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = Path(model_dir) / f'xgboost_baseline_{timestamp}.joblib'
    preprocessor_path = Path(model_dir) / f'preprocessor_{timestamp}.joblib'
    
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    
    # Also save as latest
    latest_model = Path(model_dir) / 'xgboost_baseline_latest.joblib'
    latest_preprocessor = Path(model_dir) / 'preprocessor_latest.joblib'
    
    joblib.dump(model, latest_model)
    joblib.dump(preprocessor, latest_preprocessor)
    
    print(f"Latest model saved to: {latest_model}")
    print(f"Latest preprocessor saved to: {latest_preprocessor}")


if __name__ == "__main__":
    print("="*60)
    print("FINSHIELD FRAUD DETECTION - BASELINE MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = load_transaction_data()
    
    # Create time-aware split
    print("\nCreating time-aware train/test split...")
    train_df, test_df = create_time_aware_split(df, test_size=0.2)
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = FraudPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test, y_test = preprocessor.fit_transform(test_df)
    
    # Train model
    model, metrics = train_xgboost_baseline(X_train, y_train, X_test, y_test)
    
    # Print evaluation
    print_evaluation_metrics(metrics, y_test)
    
    # Save model
    save_model_and_preprocessor(model, preprocessor)
    
    print("\n" + "="*60)
    print("BASELINE MODEL TRAINING COMPLETE!")
    print("="*60)
