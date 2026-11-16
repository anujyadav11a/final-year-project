import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class FraudPreprocessor:
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = None
        self.categorical_columns = None
        self.numeric_columns = None
        
    def fit_transform(self, df, target_col='isFraud'):
        print("Starting preprocessing...")
        
        y = df[target_col].copy()
        X = df.drop([target_col], axis=1)
        
        if 'TransactionID' in X.columns:
            X = X.drop(['TransactionID'], axis=1)
        
        print(f"Initial shape: {X.shape}")
        
        self.categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        self.numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Categorical columns: {len(self.categorical_columns)}")
        print(f"Numeric columns: {len(self.numeric_columns)}")
        
        for col in self.categorical_columns:
            le = LabelEncoder()
            X[col] = X[col].fillna('missing')
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        print(f"Encoded {len(self.categorical_columns)} categorical features")
        
        for col in self.numeric_columns:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
        
        print(f"Imputed missing values in numeric features")
        
        self.feature_columns = X.columns.tolist()
        
        print(f"Final shape: {X.shape}")
        print("Preprocessing complete!")
        
        return X, y
    
    def transform(self, df):
        X = df.copy()
        
        if 'TransactionID' in X.columns:
            X = X.drop(['TransactionID'], axis=1)
        
        if 'isFraud' in X.columns:
            X = X.drop(['isFraud'], axis=1)
        
        for col in self.categorical_columns:
            if col in X.columns:
                X[col] = X[col].fillna('missing')
                X[col] = X[col].apply(lambda x: x if x in self.label_encoders[col].classes_ else 'missing')
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        for col in self.numeric_columns:
            if col in X.columns and X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
        
        return X


def create_time_aware_split(df, test_size=0.2, time_col='TransactionDT', target_col='isFraud'):
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    print(f"\nTime-aware split:")
    print(f"Train: {len(train_df):,} transactions ({train_df[target_col].mean()*100:.2f}% fraud)")
    print(f"Test: {len(test_df):,} transactions ({test_df[target_col].mean()*100:.2f}% fraud)")
    print(f"Time range - Train: {train_df[time_col].min():.0f} to {train_df[time_col].max():.0f}")
    print(f"Time range - Test: {test_df[time_col].min():.0f} to {test_df[time_col].max():.0f}")
    
    return train_df, test_df


if __name__ == "__main__":
    from data import load_transaction_data
    
    df = load_transaction_data()
    train_df, test_df = create_time_aware_split(df, test_size=0.2)
    
    preprocessor = FraudPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    
    print(f"\nProcessed training data:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_train fraud rate: {y_train.mean()*100:.2f}%")
