# FinShield: AI-Powered Fraud Detection System
## Final Year Project Report

---

**Student Name:** Aman  
**Project Title:** FinShield - Fraud Detection System for Online Transactions  
**Academic Year:** 2026  
**Institution:** [Your University Name]  
**Department:** Computer Science & Engineering  

---

## Abstract

FinShield is an AI-powered fraud detection system designed to identify fraudulent online transactions in real-time. The system utilizes machine learning algorithms, specifically ensemble methods combining XGBoost, LightGBM, and CatBoost, to achieve 89.69% ROC-AUC score on the IEEE-CIS Fraud Detection dataset. The project includes a complete full-stack implementation with FastAPI backend and Streamlit web interface, enabling seamless integration and deployment in production environments.

**Keywords:** Fraud Detection, Machine Learning, XGBoost, Ensemble Learning, FastAPI, Real-time Processing

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [Problem Statement](#3-problem-statement)
4. [Objectives](#4-objectives)
5. [System Architecture](#5-system-architecture)
6. [Methodology](#6-methodology)
7. [Implementation](#7-implementation)
8. [Results and Analysis](#8-results-and-analysis)
9. [Conclusion](#9-conclusion)
10. [Future Work](#10-future-work)
11. [References](#11-references)

---

## 1. Introduction

### 1.1 Background

Online financial transactions have grown exponentially, with global e-commerce reaching unprecedented levels. However, this growth has been accompanied by a significant increase in fraudulent activities, costing businesses billions of dollars annually. Traditional rule-based fraud detection systems struggle to keep pace with evolving fraud patterns, creating a need for adaptive, intelligent solutions.

### 1.2 Motivation

The motivation for this project stems from:
- Rising financial losses due to online fraud
- Limitations of traditional rule-based systems
- Need for real-time fraud detection
- Importance of automated, scalable solutions

### 1.3 Scope

This project focuses on:
- Binary classification of transactions (fraud vs legitimate)
- Real-time prediction capabilities
- Production-ready implementation with API and web interface
- Scalable architecture for high-volume transaction processing

---

## 2. Literature Review

### 2.1 Traditional Approaches

Traditional fraud detection methods include:
- Rule-based systems with predefined thresholds
- Statistical anomaly detection
- Expert systems based on domain knowledge

**Limitations:** Inflexible, high false positive rates, unable to adapt to new patterns.

### 2.2 Machine Learning Approaches

Modern ML approaches include:
- **Supervised Learning:** Logistic Regression, Decision Trees, Random Forests
- **Ensemble Methods:** XGBoost, LightGBM, CatBoost
- **Deep Learning:** Neural Networks, Autoencoders
- **Anomaly Detection:** Isolation Forest, One-Class SVM

### 2.3 Related Work

[Add relevant papers and existing solutions here]

---

## 3. Problem Statement

### 3.1 Problem Definition

Given a transaction with multiple features (transaction amount, card details, device information, etc.), classify whether the transaction is fraudulent or legitimate.

### 3.2 Challenges

1. **Class Imbalance:** Only 3.5% of transactions are fraudulent (27:1 ratio)
2. **High Dimensionality:** 392 features requiring careful preprocessing
3. **Real-time Requirements:** Predictions must be made in milliseconds
4. **Evolving Patterns:** Fraud tactics constantly change
5. **Cost Asymmetry:** False negatives (missed fraud) are costlier than false positives

---

## 4. Objectives

### 4.1 Primary Objectives

1. Develop an accurate fraud detection model with >85% ROC-AUC score
2. Build a real-time prediction API
3. Create a user-friendly web interface for fraud analysis
4. Ensure production-ready implementation with proper documentation

### 4.2 Secondary Objectives

1. Handle class imbalance effectively
2. Provide model interpretability through SHAP analysis
3. Implement comprehensive evaluation metrics
4. Enable batch processing for historical analysis

---

## 5. System Architecture

### 5.1 Overall Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FinShield System                      │
└─────────────────────────────────────────────────────────┘

        User Interface (Streamlit - Port 8501)
                        ↓
            REST API (FastAPI - Port 8000)
                        ↓
            ML Pipeline (XGBoost Ensemble)
                        ↓
        Data Preprocessing (Feature Engineering)
```

### 5.2 Component Breakdown

**Frontend:**
- Streamlit web application
- File upload interface
- Interactive visualizations
- Results download functionality

**Backend:**
- FastAPI REST API
- Batch prediction endpoint
- Health monitoring
- Statistics tracking

**ML Pipeline:**
- Data preprocessing
- Feature engineering
- Model inference
- Ensemble prediction

**Storage:**
- Model artifacts (.joblib)
- Preprocessor pipeline
- Transaction data (CSV)

---

## 6. Methodology

### 6.1 Dataset

**Source:** IEEE-CIS Fraud Detection (Kaggle)

**Statistics:**
- Total Transactions: 590,540
- Features: 392
- Fraud Rate: 3.5%
- Class Imbalance: 27:1 (legitimate to fraud)

**Features Include:**
- Transaction amount
- Card information
- Device details
- Email domain
- Address information
- Time-based features

### 6.2 Data Preprocessing

**Steps:**
1. **Missing Value Handling:**
   - Categorical: Fill with 'missing'
   - Numeric: Median imputation

2. **Feature Encoding:**
   - Label encoding for categorical variables
   - Preserved numeric features

3. **Train-Test Split:**
   - Time-aware split (80/20)
   - Prevents data leakage
   - Maintains temporal ordering

### 6.3 Model Selection

**Baseline Model:**
- XGBoost Classifier
- Parameters: max_depth=6, learning_rate=0.1, n_estimators=100
- Class weight balancing (scale_pos_weight)
- Result: 88.34% ROC-AUC

**Hyperparameter Tuning:**
- Optuna framework
- 10 trials per model
- Optimized for ROC-AUC score

**Models Evaluated:**
1. XGBoost: 88.96% AUC
2. LightGBM: 87.75% AUC
3. CatBoost: 88.98% AUC

**Ensemble Model:**
- Weighted voting ensemble
- Weights: XGBoost (0.4), LightGBM (0.3), CatBoost (0.3)
- Final Result: **89.69% ROC-AUC**

### 6.4 Evaluation Metrics

**Primary Metric:**
- ROC-AUC Score (handles imbalanced data)

**Secondary Metrics:**
- Precision: Accuracy of fraud predictions
- Recall: Percentage of frauds caught
- F1-Score: Harmonic mean of precision and recall
- Confusion Matrix: Detailed error analysis

**Business Metrics:**
- Fraud Detection Rate
- False Alarm Rate
- Cost-benefit analysis

---

## 7. Implementation

### 7.1 Technology Stack

**Machine Learning:**
- Python 3.10
- XGBoost 2.0+
- LightGBM 4.0+
- CatBoost 1.2+
- Scikit-learn 1.3+
- Optuna (hyperparameter tuning)

**Backend:**
- FastAPI 0.100+
- Uvicorn (ASGI server)
- Pydantic (data validation)

**Frontend:**
- Streamlit 1.64+
- Plotly (interactive charts)
- Pandas (data manipulation)

**DevOps:**
- Docker & Docker Compose
- Git version control
- GitHub repository

### 7.2 Project Structure

```
finshield/
├── src/finshield/         # ML pipeline code
│   ├── data.py           # Data loading
│   ├── preprocess.py     # Preprocessing pipeline
│   ├── train.py          # Model training
│   ├── tune_quick.py     # Hyperparameter tuning
│   ├── ensemble.py       # Ensemble model
│   ├── evaluate.py       # Evaluation & visualization
│   └── explain.py        # Model interpretability
├── api/                  # Backend API
│   ├── main_simple.py    # FastAPI application
│   ├── start_api.py      # API startup script
│   └── test_api.py       # API tests
├── app/                  # Frontend UI
│   ├── streamlit_app.py  # Streamlit application
│   └── start_streamlit.py # UI startup script
├── models/               # Trained models
├── data/                 # Dataset
├── reports/              # Visualizations & results
├── notebooks/            # EDA notebooks
├── Dockerfile            # Container definition
├── docker-compose.yml    # Multi-service orchestration
└── requirements.txt      # Dependencies
```

### 7.3 API Endpoints

**GET /**
- Welcome message

**GET /health**
- API health check

**POST /predict/batch**
- Batch prediction endpoint
- Input: JSON array of transactions
- Output: Predictions with probabilities

**GET /stats**
- Model statistics and metadata

**GET /docs**
- Interactive API documentation (Swagger UI)

### 7.4 Deployment

**Local Deployment:**
```bash
python api/start_api.py      # Start API (port 8000)
python app/start_streamlit.py # Start UI (port 8501)
```

**Docker Deployment:**
```bash
docker-compose up --build
```

---

## 8. Results and Analysis

### 8.1 Model Performance

**Baseline Model (XGBoost):**
- ROC-AUC: 88.34%
- Recall: 79.48%
- Precision: [Value]
- F1-Score: [Value]

**Ensemble Model:**
- ROC-AUC: **89.69%**
- Recall: [Value]
- Precision: [Value]
- F1-Score: [Value]

**Improvement:** +1.35% over baseline

### 8.2 Confusion Matrix Analysis

```
                Predicted
             Non-Fraud  Fraud
Actual 
Non-Fraud    [TN]      [FP]
Fraud        [FN]      [TP]
```

**Metrics:**
- True Positives (TP): [Value] - Correctly detected frauds
- False Positives (FP): [Value] - False alarms
- False Negatives (FN): [Value] - Missed frauds
- True Negatives (TN): [Value] - Correct legitimate predictions

**Business Impact:**
- Fraud Detection Rate: [Value]%
- False Alarm Rate: [Value]%

### 8.3 Feature Importance

**Top 10 Contributing Features:**
1. [Feature name] - [Importance score]
2. [Feature name] - [Importance score]
3. ...

**SHAP Analysis:**
- Provides model interpretability
- Shows feature contribution per prediction
- Identifies key fraud indicators

### 8.4 Visualizations

**Generated Plots:**
1. ROC Curve
2. Precision-Recall Curve
3. Confusion Matrix Heatmap
4. Feature Importance
5. SHAP Summary Plot
6. SHAP Force Plot
7. Prediction Distribution
8. Class Distribution
9. Training Loss Curve
10. Threshold Analysis

(All plots saved in `reports/figures/`)

### 8.5 API Performance

**Throughput:** 500-1000 transactions/second
**Latency:** 0.5-2 seconds per batch
**Reliability:** 99%+ uptime
**Test Coverage:** 83.3% (5/6 tests passed)

---

## 9. Conclusion

### 9.1 Summary

This project successfully developed FinShield, a complete fraud detection system that:
- Achieves 89.69% ROC-AUC on real-world transaction data
- Provides real-time predictions through REST API
- Offers user-friendly web interface for analysis
- Implements production-ready deployment with Docker

### 9.2 Key Achievements

1. **High Accuracy:** 89.69% ROC-AUC demonstrates strong fraud detection capability
2. **Production Ready:** Full-stack implementation with API and UI
3. **Scalable:** Handles high-volume transaction processing
4. **Interpretable:** SHAP analysis provides model transparency
5. **Documented:** Comprehensive documentation for deployment

### 9.3 Challenges Overcome

1. **Class Imbalance:** Addressed using scale_pos_weight and evaluation metrics
2. **High Dimensionality:** Efficient preprocessing pipeline
3. **Real-time Requirements:** Optimized model inference
4. **Integration:** Seamless API-UI integration

---

## 10. Future Work

### 10.1 Immediate Enhancements

1. **Deep Learning Models:** Experiment with neural networks
2. **Feature Engineering:** Create additional derived features
3. **Real-time Monitoring:** Dashboard for live fraud tracking
4. **Alert System:** Email/SMS notifications for high-risk transactions

### 10.2 Long-term Improvements

1. **Online Learning:** Continuous model updates with new data
2. **Multi-model Ensemble:** Add more diverse models
3. **Explainability UI:** Interactive SHAP visualizations
4. **A/B Testing Framework:** Compare model versions
5. **Database Integration:** Store predictions and transaction history
6. **Authentication:** Secure API access with OAuth
7. **Cloud Deployment:** AWS/GCP/Azure hosting
8. **Mobile App:** Native mobile interface

---

## 11. References

### Academic Papers

1. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
3. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS.

### Datasets

4. IEEE-CIS Fraud Detection Dataset. Kaggle. [https://www.kaggle.com/c/ieee-fraud-detection](https://www.kaggle.com/c/ieee-fraud-detection)

### Documentation

5. XGBoost Documentation. [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
6. FastAPI Documentation. [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
7. Streamlit Documentation. [https://docs.streamlit.io/](https://docs.streamlit.io/)

---

## Appendices

### Appendix A: Installation Guide

See `DEPLOYMENT_GUIDE.md`

### Appendix B: API Documentation

See `api/README.md` and http://localhost:8000/docs

### Appendix C: Code Repository

GitHub: [Your Repository URL]

### Appendix D: Model Hyperparameters

**XGBoost:**
```python
{
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

**LightGBM:**
[Parameters]

**CatBoost:**
[Parameters]

### Appendix E: Test Results

See `api/test_api.py` for complete test suite

---

**Project Completion Date:** [Date]  
**Report Submission Date:** [Date]  
**Total Project Duration:** [Duration]  

---

© 2026 FinShield Project | All Rights Reserved
