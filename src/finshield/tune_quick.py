import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path
import time
from data import load_transaction_data
from preprocess import FraudPreprocessor, create_time_aware_split


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=10):
    print(f"\n{'='*60}")
    print("TUNING XGBOOST")
    print(f"{'='*60}")
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 5, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 150, 350),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 5, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 2, 6),
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': -1
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred_proba)
    
    start = time.time()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Time: {(time.time() - start)/60:.2f} minutes")
    print("Best params:", study.best_params)
    
    return study.best_params, study.best_value


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=10):
    print(f"\n{'='*60}")
    print("TUNING LIGHTGBM")
    print(f"{'='*60}")
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': trial.suggest_int('num_leaves', 30, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 150, 350),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 3, 8),
            'reg_lambda': trial.suggest_float('reg_lambda', 2, 6),
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred_proba)
    
    start = time.time()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Time: {(time.time() - start)/60:.2f} minutes")
    print("Best params:", study.best_params)
    
    return study.best_params, study.best_value


def tune_catboost(X_train, y_train, X_val, y_val, n_trials=10):
    print(f"\n{'='*60}")
    print("TUNING CATBOOST")
    print(f"{'='*60}")
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 150, 350),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
            'depth': trial.suggest_int('depth', 5, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2, 8),
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'random_seed': 42,
            'verbose': False,
            'task_type': 'CPU'
        }
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred_proba)
    
    start = time.time()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Time: {(time.time() - start)/60:.2f} minutes")
    print("Best params:", study.best_params)
    
    return study.best_params, study.best_value


def train_final_model(model_name, params, X_train, y_train):
    print(f"\nTraining final {model_name} model...")
    if model_name == 'xgboost':
        params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': -1
        })
        model = xgb.XGBClassifier(**params)
        
    elif model_name == 'lightgbm':
        params.update({
            'objective': 'binary',
            'metric': 'auc',
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        })
        model = lgb.LGBMClassifier(**params)
        
    elif model_name == 'catboost':
        params.update({
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'random_seed': 42,
            'verbose': False,
            'task_type': 'CPU'
        })
        model = CatBoostClassifier(**params)
    
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    print("="*60)
    print("FINSHIELD QUICK HYPERPARAMETER TUNING")
    print("="*60)
    
    print("\nLoading data...")
    df = load_transaction_data()
    
    train_df, temp_df = create_time_aware_split(df, test_size=0.2)
    val_df, test_df = create_time_aware_split(temp_df, test_size=0.5)
    
    print("\nPreprocessing...")
    preprocessor = FraudPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.fit_transform(val_df)
    X_test, y_test = preprocessor.fit_transform(test_df)
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train):,}")
    print(f"  Val: {len(X_val):,}")
    print(f"  Test: {len(X_test):,}")
    
    results = {}
    
    xgb_params, xgb_val_auc = tune_xgboost(X_train, y_train, X_val, y_val, n_trials=10)
    xgb_model = train_final_model('xgboost', xgb_params, X_train, y_train)
    xgb_test_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    results['xgboost'] = {'model': xgb_model, 'val_auc': xgb_val_auc, 'test_auc': xgb_test_auc}
    
    lgb_params, lgb_val_auc = tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=10)
    lgb_model = train_final_model('lightgbm', lgb_params, X_train, y_train)
    lgb_test_auc = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])
    results['lightgbm'] = {'model': lgb_model, 'val_auc': lgb_val_auc, 'test_auc': lgb_test_auc}
    
    cat_params, cat_val_auc = tune_catboost(X_train, y_train, X_val, y_val, n_trials=10)
    cat_model = train_final_model('catboost', cat_params, X_train, y_train)
    cat_test_auc = roc_auc_score(y_test, cat_model.predict_proba(X_test)[:, 1])
    results['catboost'] = {'model': cat_model, 'val_auc': cat_val_auc, 'test_auc': cat_test_auc}
    
    print("\n" + "="*60)
    print("TUNING SUMMARY")
    print("="*60)
    for name, result in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Val AUC: {result['val_auc']:.4f}")
        print(f"  Test AUC: {result['test_auc']:.4f}")
        
        model_path = Path('models') / f'{name}_tuned_latest.joblib'
        joblib.dump(result['model'], model_path)
        print(f"  Saved: {model_path}")
    
    joblib.dump(preprocessor, 'models/preprocessor_tuned_latest.joblib')
    
    best_name = max(results.keys(), key=lambda k: results[k]['test_auc'])
    print(f"\n🏆 Best: {best_name.upper()} - {results[best_name]['test_auc']:.4f} AUC")
    
    print("\n" + "="*60)
    print("TUNING COMPLETE!")
    print("="*60)
