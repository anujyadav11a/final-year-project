"""
Hyperparameter tuning for FinShield fraud detection using Optuna.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from sklearn.metrics import roc_auc_score
import joblib
from pathlib import Path
import time

from data import load_transaction_data
from preprocess import FraudPreprocessor, create_time_aware_split


def objective_xgboost(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for XGBoost hyperparameter tuning.
    """
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'random_state': 42,
        'tree_method': 'hist',
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    
    return auc


def objective_lightgbm(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for LightGBM hyperparameter tuning.
    """
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    
    return auc


def objective_catboost(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for CatBoost hyperparameter tuning.
    """
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'random_seed': 42,
        'verbose': False,
        'task_type': 'CPU'
    }
    
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    
    return auc


def tune_model(model_name, X_train, y_train, X_val, y_val, n_trials=50):
    """
    Tune hyperparameters for a specific model.
    
    Parameters:
    -----------
    model_name : str
        'xgboost', 'lightgbm', or 'catboost'
    n_trials : int
        Number of optimization trials
    """
    print(f"\n{'='*60}")
    print(f"TUNING {model_name.upper()} HYPERPARAMETERS")
    print(f"{'='*60}")
    print(f"Number of trials: {n_trials}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    
    start_time = time.time()
    
    # Create study
    study = optuna.create_study(direction='maximize', study_name=f'{model_name}_tuning')
    
    # Select objective function
    if model_name == 'xgboost':
        objective_func = lambda trial: objective_xgboost(trial, X_train, y_train, X_val, y_val)
    elif model_name == 'lightgbm':
        objective_func = lambda trial: objective_lightgbm(trial, X_train, y_train, X_val, y_val)
    elif model_name == 'catboost':
        objective_func = lambda trial: objective_catboost(trial, X_train, y_train, X_val, y_val)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Optimize
    study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)
    
    tuning_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"TUNING RESULTS FOR {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Best ROC-AUC: {study.best_value:.4f}")
    print(f"Tuning time: {tuning_time/60:.2f} minutes")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params, study.best_value


def train_best_model(model_name, best_params, X_train, y_train):
    """
    Train model with best hyperparameters.
    """
    print(f"\nTraining final {model_name} model with best parameters...")
    
    # Add fixed parameters
    if model_name == 'xgboost':
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': -1
        })
        model = xgb.XGBClassifier(**best_params)
        
    elif model_name == 'lightgbm':
        best_params.update({
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        })
        model = lgb.LGBMClassifier(**best_params)
        
    elif model_name == 'catboost':
        best_params.update({
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'random_seed': 42,
            'verbose': False,
            'task_type': 'CPU'
        })
        model = CatBoostClassifier(**best_params)
    
    model.fit(X_train, y_train, verbose=False)
    print(f"Model trained successfully!")
    
    return model


if __name__ == "__main__":
    print("="*60)
    print("FINSHIELD HYPERPARAMETER TUNING")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = load_transaction_data()
    
    # Create splits (80% train, 10% validation, 10% test)
    train_df, temp_df = create_time_aware_split(df, test_size=0.2)
    val_df, test_df = create_time_aware_split(temp_df, test_size=0.5)
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Validation: {len(val_df):,} samples")
    print(f"  Test: {len(test_df):,} samples")
    
    # Preprocess
    print("\nPreprocessing...")
    preprocessor = FraudPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.fit_transform(val_df)
    X_test, y_test = preprocessor.fit_transform(test_df)
    
    # Tune each model
    results = {}
    
    for model_name in ['xgboost', 'lightgbm', 'catboost']:
        best_params, best_score = tune_model(
            model_name, 
            X_train, y_train, 
            X_val, y_val,
            n_trials=50
        )
        
        # Train best model
        best_model = train_best_model(model_name, best_params, X_train, y_train)
        
        # Evaluate on test
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n{model_name.upper()} Test AUC: {test_auc:.4f}")
        
        # Save model
        model_path = Path('models') / f'{model_name}_tuned_latest.joblib'
        joblib.dump(best_model, model_path)
        print(f"Model saved to: {model_path}")
        
        results[model_name] = {
            'best_params': best_params,
            'val_auc': best_score,
            'test_auc': test_auc,
            'model': best_model
        }
    
    # Save preprocessor
    joblib.dump(preprocessor, 'models/preprocessor_tuned_latest.joblib')
    
    # Print summary
    print("\n" + "="*60)
    print("TUNING SUMMARY")
    print("="*60)
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Validation AUC: {result['val_auc']:.4f}")
        print(f"  Test AUC: {result['test_auc']:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_auc'])
    print(f"\n🏆 Best model: {best_model_name.upper()} with {results[best_model_name]['test_auc']:.4f} AUC")
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("="*60)
