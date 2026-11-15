# FinShield Deployment Guide

Complete guide for deploying and running the FinShield fraud detection system.

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Running the Application](#running-the-application)
4. [Architecture Overview](#architecture-overview)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)
7. [Production Deployment](#production-deployment)

---

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.10 or higher
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 5 GB free space
- **Internet**: Required for initial setup

### Software Dependencies
- **Conda** (recommended) or pip for package management
- **Git** for version control
- Modern web browser (Chrome, Firefox, Edge)

---

## 📦 Installation

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd final-year-project
```

### Step 2: Install Dependencies

#### Option A: Using Conda (Recommended)
```bash
# Create environment (optional)
conda create -n finshield python=3.10
conda activate finshield

# Install packages
conda install numpy pandas scikit-learn xgboost lightgbm catboost joblib -y
conda install matplotlib seaborn shap optuna imbalanced-learn -y
conda install jupyter ipykernel plotly -y
conda install fastapi uvicorn pydantic python-multipart -y
conda install streamlit requests -y
```

#### Option B: Using pip
```bash
# Create virtual environment (optional)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
# Check Python version
python --version

# Check key packages
python -c "import xgboost; print('XGBoost:', xgboost.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
```

---

## 🚀 Running the Application

### Quick Start (All Components)

#### Terminal 1: Start API Backend
```bash
python api\start_api.py
```

Expected output:
```
🛡️  Starting FinShield Fraud Detection API...
✅ Model loaded successfully!
📍 API running at: http://localhost:8000
📚 API documentation: http://localhost:8000/docs
```

#### Terminal 2: Start Web Interface
```bash
python app\start_streamlit.py
```

Expected output:
```
🛡️  Starting FinShield Web Interface...
🌐 Server will start at: http://localhost:8501
You can now view your Streamlit app in your browser.
```

#### Access Points
- **Web Interface**: http://localhost:8501
- **API Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FinShield System                      │
└─────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   Browser    │         │   Streamlit  │         │   FastAPI    │
│  (User UI)   │◄────────│   Frontend   │◄────────│   Backend    │
│  Port 8501   │  HTTP   │  Port 8501   │  HTTP   │  Port 8000   │
└──────────────┘         └──────────────┘         └──────────────┘
                                                           │
                                                           ▼
                                                  ┌──────────────┐
                                                  │  ML Models   │
                                                  │  (XGBoost)   │
                                                  └──────────────┘
                                                           │
                                                           ▼
                                                  ┌──────────────┐
                                                  │ Preprocessor │
                                                  │   Pipeline   │
                                                  └──────────────┘
```

### Component Details

#### 1. Web Interface (Streamlit)
- **Location**: `app/streamlit_app.py`
- **Port**: 8501
- **Purpose**: User-facing web application
- **Features**: File upload, visualization, results download

#### 2. API Backend (FastAPI)
- **Location**: `api/main_simple.py`
- **Port**: 8000
- **Purpose**: ML inference service
- **Endpoints**: `/predict`, `/predict/batch`, `/health`, `/stats`

#### 3. ML Models
- **Location**: `models/`
- **Type**: XGBoost, LightGBM, CatBoost
- **Features**: 392 transaction features
- **Output**: Fraud probability (0.0 to 1.0)

---

## 🧪 Testing

### Test API Backend

#### Manual Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test stats endpoint
curl http://localhost:8000/stats
```

#### Automated Testing
```bash
# Run API test suite
python api\test_api.py
```

Expected output:
```
Testing FinShield Fraud Detection API
================================================================================
✅ Test 1: Root endpoint - PASSED
✅ Test 2: Health check - PASSED
✅ Test 3: Batch predictions - PASSED
...
Overall: 5/6 tests passed (83.33%)
```

### Test Web Interface

1. **Start both services** (API + Streamlit)
2. **Open browser** to http://localhost:8501
3. **Upload test data**: Use `app/sample_transactions.csv`
4. **Click "Analyze Transactions"**
5. **Verify results** are displayed correctly

### Generate Test Data
```bash
python app\create_test_data.py
```

This creates `app/sample_transactions.csv` with 100 sample transactions.

---

## 🐛 Troubleshooting

### Common Issues

#### 1. API Not Starting
**Error**: `Address already in use` or `Port 8000 is busy`

**Solution**:
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (use PID from above)
taskkill /PID <pid> /F

# Or change port in api/start_api.py
```

#### 2. Model Not Found
**Error**: `FileNotFoundError: models/xgboost_baseline_latest.joblib`

**Solution**:
```bash
# Train the model first
python src\finshield\train.py
```

#### 3. Import Errors
**Error**: `ModuleNotFoundError: No module named 'finshield'`

**Solution**:
```bash
# Set PYTHONPATH
$env:PYTHONPATH="d:\final-year-project\src"  # Windows PowerShell
export PYTHONPATH="/path/to/final-year-project/src"  # Linux/Mac
```

#### 4. Streamlit Email Prompt
**Error**: Streamlit asks for email on first run

**Solution**: Config file `.streamlit/config.toml` is already created. Just press Enter to skip.

#### 5. Slow Predictions
**Issue**: Predictions taking too long

**Solution**:
- Process smaller batches (< 1000 rows)
- Use faster model (baseline instead of ensemble)
- Increase system resources
- Check for memory issues

#### 6. Missing Features Error
**Error**: `Feature mismatch: Expected 392 features`

**Solution**:
- Ensure CSV has all required columns
- Use data from `data/raw/train_transaction.csv`
- Check `app/sample_transactions.csv` for reference

---

## 🌐 Production Deployment

### Option 1: Local Server

#### Setup
1. Use fixed ports (8000 for API, 8501 for UI)
2. Configure firewall rules
3. Set up auto-restart on failure

#### Example Startup Scripts

**Windows (start_all.bat)**:
```batch
@echo off
start "FinShield API" cmd /k "python api\start_api.py"
timeout /t 5
start "FinShield UI" cmd /k "python app\start_streamlit.py"
```

**Linux (start_all.sh)**:
```bash
#!/bin/bash
python api/start_api.py &
sleep 5
python app/start_streamlit.py &
```

### Option 2: Docker Container

#### Create Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

EXPOSE 8000 8501

CMD ["bash", "-c", "python api/start_api.py & python app/start_streamlit.py"]
```

#### Build and Run
```bash
docker build -t finshield .
docker run -p 8000:8000 -p 8501:8501 finshield
```

### Option 3: Cloud Deployment

#### AWS EC2
1. Launch EC2 instance (t2.medium or larger)
2. Install dependencies
3. Configure security groups (ports 8000, 8501)
4. Set up systemd services for auto-restart

#### Heroku
1. Create `Procfile`:
   ```
   web: streamlit run app/streamlit_app.py --server.port=$PORT
   ```
2. Deploy using Heroku CLI

#### Google Cloud Run
1. Containerize application
2. Push to Google Container Registry
3. Deploy to Cloud Run

---

## 📊 Performance Optimization

### Model Loading
- Models are loaded once at startup
- Keep models in memory for fast predictions
- Use joblib for efficient serialization

### API Performance
- **Throughput**: ~500-1000 transactions/second
- **Latency**: ~0.5-2 seconds per batch
- **Concurrency**: FastAPI handles multiple requests

### Streamlit Caching
- Results cached in session state
- No need to re-predict same data
- Clear cache with browser refresh

---

## 🔒 Security Considerations

### API Security
- Add authentication (API keys, OAuth)
- Implement rate limiting
- Use HTTPS in production
- Validate input data

### Data Privacy
- Don't log sensitive transaction data
- Clear uploaded files after processing
- Use encryption for data in transit
- Comply with data protection regulations

---

## 📈 Monitoring

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Check logs
tail -f api/logs/api.log  # if logging enabled
```

### Metrics to Monitor
- Request count
- Response time
- Error rate
- Model prediction distribution
- System resource usage (CPU, RAM)

---

## 🔄 Updates and Maintenance

### Updating Models
1. Train new model
2. Save to `models/` directory
3. Update model path in `api/main_simple.py`
4. Restart API

### Updating Dependencies
```bash
# Update specific package
conda update xgboost

# Update all packages
conda update --all

# Update requirements.txt
pip freeze > requirements.txt
```

### Backup
Important directories to backup:
- `models/` - Trained models
- `data/processed/` - Processed data
- `reports/` - Visualizations and results
- `notebooks/` - Analysis notebooks

---

## 📞 Support

### Getting Help
1. Check logs for error messages
2. Review troubleshooting section
3. Test with sample data
4. Verify all services are running

### Useful Commands
```bash
# Check running processes
netstat -ano | findstr :8000
netstat -ano | findstr :8501

# Test API
curl http://localhost:8000/health

# Check Python packages
pip list | grep xgboost
conda list | grep streamlit
```

---

## 📝 Notes

- Always start API before Streamlit interface
- Use sample data for testing
- Monitor system resources during large batch predictions
- Keep models and data out of git (already in .gitignore)

---

**FinShield v1.0** | Deployment Guide | Last Updated: October 2026
