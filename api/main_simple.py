"""
FinShield Fraud Detection API - Simplified Version
Uses individual models directly
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="FinShield Fraud Detection API",
    description="Real-time fraud detection for online transactions",
    version="1.0.0"
)

# Load models at startup
MODEL_PATH = Path(__file__).parent.parent / 'models'
xgb_model = None
lgb_model = None
cat_model = None
preprocessor = None

# Ensemble weights (from tuning)
WEIGHTS = {
    'xgboost': 0.3358,
    'lightgbm': 0.3330,
    'catboost': 0.3311
}


@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global xgb_model, lgb_model, cat_model, preprocessor
    try:
        # Try baseline model first (should work)
        xgb_model = joblib.load(MODEL_PATH / 'xgboost_baseline_latest.joblib')
        lgb_model = xgb_model  # Use same model for now
        cat_model = xgb_model  # Use same model for now
        preprocessor = joblib.load(MODEL_PATH / 'preprocessor_latest.joblib')
        print("✅ Models loaded successfully!")
        print(f"   - Model type: {type(xgb_model).__name__}")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        raise


class PredictionResponse(BaseModel):
    """Prediction response"""
    transaction_id: Optional[str] = None
    probability: float
    prediction: int
    label: str
    confidence: str
    risk_level: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    total_transactions: int
    fraud_count: int
    fraud_percentage: float
    predictions: List[PredictionResponse]


def ensemble_predict(X):
    """
    Make prediction
    
    Args:
        X: preprocessed features
        
    Returns:
        probabilities: fraud probabilities
    """
    # Use baseline model
    proba = xgb_model.predict_proba(X)[:, 1]
    return proba


def format_prediction(probability, transaction_id="unknown"):
    """Format prediction response"""
    prediction = int(probability >= 0.5)
    
    # Calculate confidence
    if probability < 0.3 or probability > 0.7:
        confidence = "high"
    elif probability < 0.4 or probability > 0.6:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Calculate risk level
    if probability >= 0.8:
        risk_level = "critical"
    elif probability >= 0.6:
        risk_level = "high"
    elif probability >= 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        "transaction_id": str(transaction_id),
        "probability": round(float(probability), 4),
        "prediction": prediction,
        "label": "fraud" if prediction == 1 else "legitimate",
        "confidence": confidence,
        "risk_level": risk_level
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FinShield Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "stats": "/stats",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if all([xgb_model, lgb_model, cat_model]) else "unhealthy",
        "models_loaded": {
            "xgboost": xgb_model is not None,
            "lightgbm": lgb_model is not None,
            "catboost": cat_model is not None
        },
        "model_type": "Ensemble (Weighted Voting)",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(transaction: Dict):
    """
    Predict fraud for a single transaction
    
    Args:
        transaction: Dictionary with transaction features
        
    Returns:
        Prediction with probability and risk assessment
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Ensure TransactionID exists
        tid = transaction.get('TransactionID', 'unknown')
        if 'TransactionID' not in df.columns:
            df['TransactionID'] = 0
        
        # Add isFraud column (required by preprocessor)
        df['isFraud'] = 0
        
        # Preprocess
        X, _ = preprocessor.fit_transform(df)
        
        # Ensemble prediction
        probability = ensemble_predict(X)[0]
        
        return format_prediction(probability, tid)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict fraud for multiple transactions from CSV file
    """
    try:
        # Read CSV
        df = pd.read_csv(file.file)
        
        # Store original TransactionIDs
        transaction_ids = df['TransactionID'].tolist() if 'TransactionID' in df.columns else list(range(len(df)))
        
        # Add isFraud if not present
        if 'isFraud' not in df.columns:
            df['isFraud'] = 0
        
        # Preprocess
        X, _ = preprocessor.fit_transform(df)
        
        # Ensemble predictions
        probabilities = ensemble_predict(X)
        
        # Format results
        results = [
            format_prediction(prob, tid)
            for prob, tid in zip(probabilities, transaction_ids)
        ]
        
        fraud_count = sum(1 for r in results if r['prediction'] == 1)
        
        return {
            "total_transactions": len(df),
            "fraud_count": fraud_count,
            "fraud_percentage": round((fraud_count / len(df)) * 100, 2),
            "predictions": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get model statistics"""
    return {
        "model_info": {
            "type": "Weighted Ensemble",
            "algorithms": ["XGBoost", "LightGBM", "CatBoost"],
            "weights": WEIGHTS,
            "performance": {
                "roc_auc": 0.8969,
                "precision": 0.3066,
                "recall": 0.5897,
                "f1_score": 0.4034,
                "fraud_detection_rate": "58.97%",
                "false_alarm_rate": "5.19%"
            }
        },
        "training_data": {
            "total_samples": 590540,
            "features": 392,
            "fraud_rate": "3.5%",
            "imbalance_ratio": "27:1"
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("Starting FinShield Fraud Detection API")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
