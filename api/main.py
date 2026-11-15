"""
FinShield Fraud Detection API
FastAPI application for real-time fraud prediction
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent / 'src' / 'finshield'))

# Import after adding to path
import finshield.preprocess
import finshield.ensemble

# Initialize FastAPI app
app = FastAPI(
    title="FinShield Fraud Detection API",
    description="Real-time fraud detection for online transactions using ensemble ML models",
    version="1.0.0"
)

# Load models at startup
MODEL_PATH = Path(__file__).parent.parent / 'models'
ensemble_model = None
preprocessor = None


@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global ensemble_model, preprocessor
    try:
        ensemble_model = joblib.load(MODEL_PATH / 'ensemble_latest.joblib')
        preprocessor = joblib.load(MODEL_PATH / 'preprocessor_latest.joblib')
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


# Pydantic models for request/response
class TransactionFeatures(BaseModel):
    """Single transaction features"""
    TransactionDT: float
    TransactionAmt: float
    ProductCD: str
    card1: Optional[float] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    # Add more fields as needed - this is a simplified version
    
    class Config:
        json_schema_extra = {
            "example": {
                "TransactionDT": 86400,
                "TransactionAmt": 68.5,
                "ProductCD": "W",
                "card1": 13926.0,
                "card2": 150.0,
                "card3": 150.0,
                "card4": "discover",
                "card5": 142.0,
                "card6": "credit"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response"""
    transaction_id: Optional[str] = None
    probability: float = Field(..., description="Fraud probability (0-1)")
    prediction: int = Field(..., description="0=legitimate, 1=fraud")
    label: str = Field(..., description="fraud or legitimate")
    confidence: str = Field(..., description="low, medium, high")
    risk_level: str = Field(..., description="low, medium, high, critical")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    total_transactions: int
    fraud_count: int
    fraud_percentage: float
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: str
    version: str


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "FinShield Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if ensemble_model is not None else "unhealthy",
        "model_loaded": ensemble_model is not None,
        "model_type": "Ensemble (XGBoost + LightGBM + CatBoost)",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
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
        
        # Ensure TransactionID exists (create dummy if not)
        if 'TransactionID' not in df.columns:
            df['TransactionID'] = 0
        
        # Add isFraud column (required by preprocessor but not used)
        df['isFraud'] = 0
        
        # Preprocess
        X, _ = preprocessor.fit_transform(df)
        
        # Predict
        probability = float(ensemble_model.predict_proba(X)[0])
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
            "transaction_id": str(transaction.get('TransactionID', 'unknown')),
            "probability": round(probability, 4),
            "prediction": prediction,
            "label": "fraud" if prediction == 1 else "legitimate",
            "confidence": confidence,
            "risk_level": risk_level
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict fraud for multiple transactions from CSV file
    
    Args:
        file: CSV file with transaction data
        
    Returns:
        Batch predictions with summary statistics
    """
    try:
        # Read CSV
        df = pd.read_csv(file.file)
        
        # Store original TransactionIDs if present
        transaction_ids = df['TransactionID'].tolist() if 'TransactionID' in df.columns else list(range(len(df)))
        
        # Add isFraud if not present
        if 'isFraud' not in df.columns:
            df['isFraud'] = 0
        
        # Preprocess
        X, _ = preprocessor.fit_transform(df)
        
        # Predict
        probabilities = ensemble_model.predict_proba(X)
        predictions = (probabilities >= 0.5).astype(int)
        
        # Create response
        results = []
        for i, (prob, pred, tid) in enumerate(zip(probabilities, predictions, transaction_ids)):
            # Calculate confidence
            if prob < 0.3 or prob > 0.7:
                confidence = "high"
            elif prob < 0.4 or prob > 0.6:
                confidence = "medium"
            else:
                confidence = "low"
            
            # Calculate risk level
            if prob >= 0.8:
                risk_level = "critical"
            elif prob >= 0.6:
                risk_level = "high"
            elif prob >= 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            results.append({
                "transaction_id": str(tid),
                "probability": round(float(prob), 4),
                "prediction": int(pred),
                "label": "fraud" if pred == 1 else "legitimate",
                "confidence": confidence,
                "risk_level": risk_level
            })
        
        fraud_count = sum(predictions)
        
        return {
            "total_transactions": len(df),
            "fraud_count": int(fraud_count),
            "fraud_percentage": round((fraud_count / len(df)) * 100, 2),
            "predictions": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/stats", tags=["Statistics"])
async def get_model_stats():
    """Get model statistics and information"""
    return {
        "model_info": {
            "type": "Ensemble Model",
            "algorithms": ["XGBoost", "LightGBM", "CatBoost"],
            "performance": {
                "roc_auc": 0.8969,
                "precision": 0.3066,
                "recall": 0.5897,
                "f1_score": 0.4034
            },
            "training_data": {
                "total_samples": 590540,
                "features": 392,
                "fraud_rate": "3.5%"
            }
        },
        "prediction_thresholds": {
            "default": 0.5,
            "high_confidence": ">0.7 or <0.3",
            "critical_risk": ">0.8"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
