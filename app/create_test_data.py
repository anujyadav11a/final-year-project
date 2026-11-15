"""
Create sample test data for FinShield interface testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.finshield.data import load_transaction_data

def create_sample_test_data(n_samples=100, output_path='app/sample_transactions.csv'):
    """
    Create a sample test CSV with real data structure
    
    Parameters:
    -----------
    n_samples : int
        Number of sample transactions to include
    output_path : str
        Where to save the sample CSV
    """
    
    print(f"Creating sample test data with {n_samples} transactions...")
    
    # Load the full dataset
    df = load_transaction_data()
    
    # Sample mix of fraud and legitimate transactions
    fraud_samples = df[df['isFraud'] == 1].sample(n=min(20, len(df[df['isFraud'] == 1])), random_state=42)
    legit_samples = df[df['isFraud'] == 0].sample(n=n_samples-len(fraud_samples), random_state=42)
    
    # Combine
    sample_df = pd.concat([fraud_samples, legit_samples]).sample(frac=1, random_state=42)
    
    # Drop the target variable for prediction
    if 'isFraud' in sample_df.columns:
        sample_df = sample_df.drop('isFraud', axis=1)
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(exist_ok=True)
    sample_df.to_csv(output_file, index=False)
    
    print(f"✅ Sample data created: {output_file}")
    print(f"   Shape: {sample_df.shape}")
    print(f"   Size: {output_file.stat().st_size / 1024:.2f} KB")
    print(f"\n📋 You can use this file to test the Streamlit interface!")
    
    return sample_df

if __name__ == "__main__":
    create_sample_test_data()
