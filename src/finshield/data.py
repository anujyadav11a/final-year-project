"""
Data loading utilities for FinShield fraud detection.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_transaction_data(data_path='data/raw/train_transaction.csv'):
    """
    Load the IEEE-CIS fraud detection transaction data.
    
    Parameters:
    -----------
    data_path : str
        Path to the train_transaction.csv file
        
    Returns:
    --------
    pd.DataFrame
        Transaction data with all 394 columns
    """
    df = pd.read_csv(data_path)
    print(f"Loaded transaction data: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}... (showing first 10)")
    return df


def get_data_info(df):
    """
    Get comprehensive information about the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction dataframe
        
    Returns:
    --------
    dict
        Dictionary containing dataset statistics
    """
    info = {
        'shape': df.shape,
        'total_transactions': len(df),
        'fraud_count': df['isFraud'].sum(),
        'fraud_percentage': (df['isFraud'].sum() / len(df)) * 100,
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.value_counts().to_dict()
    }
    
    return info


def print_data_summary(df):
    """
    Print a comprehensive summary of the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction dataframe
    """
    info = get_data_info(df)
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Shape: {info['shape']}")
    print(f"Total Transactions: {info['total_transactions']:,}")
    print(f"\nFraud Statistics:")
    print(f"  Fraud Cases: {info['fraud_count']:,}")
    print(f"  Non-Fraud Cases: {info['total_transactions'] - info['fraud_count']:,}")
    print(f"  Fraud Rate: {info['fraud_percentage']:.2f}%")
    print(f"\nMissing Values:")
    print(f"  Total Missing: {info['missing_values']:,}")
    print(f"  Missing Percentage: {info['missing_percentage']:.2f}%")
    print(f"\nData Types:")
    for dtype, count in info['dtypes'].items():
        print(f"  {dtype}: {count} columns")
    print("="*60)
    
    # Show columns with high missing rates
    missing_pct = (df.isnull().sum() / len(df)) * 100
    high_missing = missing_pct[missing_pct > 50].sort_values(ascending=False)
    
    if len(high_missing) > 0:
        print(f"\nColumns with >50% missing values: {len(high_missing)}")
        print(high_missing.head(10))
    
    # Show target distribution
    print(f"\nTarget Distribution (isFraud):")
    print(df['isFraud'].value_counts())
    print(df['isFraud'].value_counts(normalize=True) * 100)
    
    return info


if __name__ == "__main__":
    # Load and explore the data
    df = load_transaction_data()
    info = print_data_summary(df)
