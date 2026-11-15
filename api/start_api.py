"""
Start the FinShield API server
"""
import uvicorn

if __name__ == "__main__":
    print("="*60)
    print("Starting FinShield Fraud Detection API")
    print("="*60)
    print("API will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop")
    print("="*60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
