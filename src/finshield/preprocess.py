"""
Data preprocessing pipeline for FinShield fraud detection.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class FraudPreprocessor:
    """
    Preprocessing pipeline for fraud detection data.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = None
        self.categorical_columns = None
        self.numeric_columns = None
        
    def fit_transform(self, df, target_col='isFraud'):
        """
        Fit the preprocessor and transform the data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw transaction data
        target_col : str
            Name of the target column
            
        Returns:
        --------
        X : pd.DataFrame
            Processed features
        y : pd.Series
            Target variable
        """
        print("Starting preprocessing...")
        
        # Separate target and remove TransactionID
        y = df[target_col].copy()
        X = df.drop([target_col], axis=1)
        
        # Remove TransactionID if present (not a feature)
        if 'TransactionID' in X.columns:
            X = X.drop(['TransactionID'], axis=1)
        
        print(f"Initial shape: {X.shape}")
        
        # Identify column types
        self.categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        self.numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Categorical columns: {len(self.categorical_columns)}")
        print(f"Numeric columns: {len(self.numeric_columns)}")
        
        # Handle categorical features with Label Encoding
        for col in self.categorical_columns:
            le = LabelEncoder()
            # Handle missing values
            X[col] = X[col].fillna('missing')
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        print(f"Encoded {len(self.categorical_columns)} categorical features")
        
        # Handle missing values in numeric features
        # For baseline: simple median imputation
        for col in self.numeric_columns:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
        
        print(f"Imputed missing values in numeric features")
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        print(f"Final shape: {X.shape}")
        print("Preprocessing complete!")
        
        return X, y
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessor.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw transaction data
            
        Returns:
        --------
        X : pd.DataFrame
            Processed features
        """
        X = df.copy()
        
        # Remove TransactionID if present
        if 'TransactionID' in X.columns:
            X = X.drop(['TransactionID'], axis=1)
        
        # Remove target if present
        if 'isFraud' in X.columns:
            X = X.drop(['isFraud'], axis=1)
        
        # Encode categorical features
        for col in self.categorical_columns:
            if col in X.columns:
                X[col] = X[col].fillna('missing')
                # Handle unseen categories
                X[col] = X[col].apply(lambda x: x if x in self.label_encoders[col].classes_ else 'missing')
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Impute numeric features
        for col in self.numeric_columns:
            if col in X.columns and X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
        
        return X


def create_time_aware_split(df, test_size=0.2, time_col='TransactionDT', target_col='isFraud'):
    """
    Create train/test split based on transaction time to avoid data leakage.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction data with time column
    test_size : float
        Proportion of data for testing
    time_col : str
        Name of the time column
    target_col : str
        Name of the target column
        
    Returns:
    --------
    train_df, test_df : pd.DataFrame
        Train and test dataframes
    """
    # Sort by time
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    
    # Split based on time
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
    # Test the preprocessing pipeline
    from data import load_transaction_data
    
    # Load data
    df = load_transaction_data()
    
    # Create time-aware split
    train_df, test_df = create_time_aware_split(df, test_size=0.2)
    
    # Fit preprocessor on training data
    preprocessor = FraudPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    
    print(f"\nProcessed training data:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_train fraud rate: {y_train.mean()*100:.2f}%")
