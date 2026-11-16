import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from pathlib import Path
from finshield.data import load_transaction_data
from finshield.preprocess import FraudPreprocessor, create_time_aware_split


class VotingEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        if weights is None:
            self.weights = {name: 1/len(models) for name in models.keys()}
        else:
            total = sum(weights.values())
            self.weights = {name: w/total for name, w in weights.items()}
    def predict_proba(self, X):
        predictions = []
        for name, model in self.models.items():
            pred = model.predict_proba(X)[:, 1]
            weighted_pred = pred * self.weights[name]
            predictions.append(weighted_pred)
        
        ensemble_pred = np.sum(predictions, axis=0)
        
        return ensemble_pred
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

def evaluate_ensemble(ensemble, X_test, y_test, threshold=0.5):
    y_pred_proba = ensemble.predict_proba(X_test)
    y_pred = (y_pred_proba >= threshold).astype(int)
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return metrics


def print_ensemble_results(metrics):
    print("\n" + "="*60)
    print("ENSEMBLE MODEL EVALUATION")
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
    
    tn, fp, fn, tp = cm.ravel()
    fraud_detection_rate = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    false_alarm_rate = (fp / (tn + fp)) * 100 if (tn + fp) > 0 else 0
    
    print(f"\n{'Business Impact':<20}")
    print("-" * 35)
    print(f"Fraud Detection Rate: {fraud_detection_rate:.2f}%")
    print(f"False Alarm Rate:     {false_alarm_rate:.2f}%")
    print(f"Frauds Caught:        {tp:,} / {tp + fn:,}")
    
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("FINSHIELD ENSEMBLE MODEL")
    print("="*60)
    
    print("\nLoading tuned models...")
    xgb_model = joblib.load('models/xgboost_tuned_latest.joblib')
    lgb_model = joblib.load('models/lightgbm_tuned_latest.joblib')
    cat_model = joblib.load('models/catboost_tuned_latest.joblib')
    preprocessor = joblib.load('models/preprocessor_tuned_latest.joblib')
    
    models = {
        'xgboost': xgb_model,
        'lightgbm': lgb_model,
        'catboost': cat_model
    }
    
    print("\nLoading test data...")
    df = load_transaction_data()
    train_df, temp_df = create_time_aware_split(df, test_size=0.2)
    val_df, test_df = create_time_aware_split(temp_df, test_size=0.5)
    
    print("Preprocessing...")
    X_test, y_test = preprocessor.fit_transform(test_df)
    
    print(f"\nTest set: {len(X_test):,} samples")
    print(f"Fraud rate: {y_test.mean()*100:.2f}%")
    
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*60)
    
    individual_scores = {}
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        individual_scores[name] = auc
        print(f"{name.upper():<15} AUC: {auc:.4f}")
    
    print("\n" + "="*60)
    print("CREATING WEIGHTED ENSEMBLE")
    print("="*60)
    
    weights = {
        'xgboost': 0.9145,    # Best validation AUC
        'lightgbm': 0.9068,
        'catboost': 0.9017
    }
    
    ensemble = VotingEnsemble(models, weights)
    
    print("\nEnsemble weights:")
    for name, weight in ensemble.weights.items():
        print(f"  {name}: {weight:.4f}")
    
    print("\nEvaluating ensemble...")
    metrics = evaluate_ensemble(ensemble, X_test, y_test)
    print_ensemble_results(metrics)
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"\n{'Model':<15} {'ROC-AUC':<15} {'Improvement'}")
    print("-" * 45)
    for name, auc in individual_scores.items():
        improvement = ((metrics['roc_auc'] - auc) / auc) * 100
        print(f"{name.upper():<15} {auc:.4f}         {improvement:+.2f}%")
    print(f"{'ENSEMBLE':<15} {metrics['roc_auc']:.4f}         -")
    
    print("\n" + "="*60)
    print("SAVING ENSEMBLE MODEL")
    print("="*60)
    
    ensemble_path = Path('models') / 'ensemble_latest.joblib'
    joblib.dump(ensemble, ensemble_path)
    print(f"Ensemble saved to: {ensemble_path}")
    
    print("\n" + "="*60)
    print("ENSEMBLE MODEL COMPLETE!")
    print("="*60)
