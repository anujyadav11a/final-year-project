import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="FinShield - Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color:
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color:
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color:
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .fraud-badge {
        background-color:
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .legitimate-badge {
        background-color:
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    </style>
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False
def get_api_stats():
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None
def predict_batch(df):
    try:
        data = df.to_dict(orient='records')
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"transactions": data},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

def create_fraud_distribution_chart(results):
    fraud_counts = pd.Series([r['prediction'] for r in results]).value_counts()
    labels = ['Legitimate', 'Fraudulent']
    values = [fraud_counts.get(0, 0), fraud_counts.get(1, 0)]
    colors = ['#00cc00', '#ff4b4b']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo='label+percent+value',
        textfont=dict(size=14)
    )])
    
    fig.update_layout(
        title="Fraud Distribution",
        height=400,
        showlegend=True
    )
    
    return fig

def create_confidence_distribution(results):
    probabilities = [r['fraud_probability'] for r in results]
    fig = px.histogram(
        x=probabilities,
        nbins=50,
        title="Fraud Probability Distribution",
        labels={'x': 'Fraud Probability', 'y': 'Count'},
        color_discrete_sequence=['#1f77b4']
    )
    
    fig.update_layout(
        xaxis_title="Fraud Probability",
        yaxis_title="Number of Transactions",
        height=400,
        showlegend=False
    )
    
    return fig

def create_results_table(df, results):
    results_df = pd.DataFrame(results)
    if 'TransactionID' in df.columns:
        results_df.insert(0, 'TransactionID', df['TransactionID'].values)
    
    results_df['fraud_probability'] = results_df['fraud_probability'].round(4)
    results_df['prediction_label'] = results_df['prediction'].map({
        0: '✅ Legitimate',
        1: '🚨 Fraudulent'
    })
    
    return results_df

st.markdown('<div class="main-header">🛡️ FinShield</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Fraud Detection System</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ System Status")
    
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
        
        stats = get_api_stats()
        if stats:
            st.metric("Model Loaded", stats.get("model_type", "Unknown"))
            st.metric("API Version", stats.get("version", "Unknown"))
    else:
        st.error("❌ API Disconnected")
        st.warning("Please ensure the API is running at http://localhost:8000")
        st.code("python api\\start_api.py", language="bash")
    
    st.divider()
    
    st.header("📊 About")
    st.info("""
    **FinShield** uses machine learning to detect fraudulent online transactions.
    
    **Features:**
    - Real-time fraud detection
    - Batch processing
    - Probability scoring
    - Visual analytics
    
    **Model:** XGBoost Classifier
    **AUC Score:** 88.34%
    1. Upload a CSV file with transaction data
    2. Ensure it contains all required features
    3. Click "Analyze Transactions"
    4. View results and download predictions
        **Algorithm:** XGBoost (Extreme Gradient Boosting)
        **Performance Metrics:**
        - ROC-AUC Score: 88.34%
        - Recall: 79.48%
        - Precision: Optimized for fraud detection
        
        **Training Data:**
        - Dataset: IEEE-CIS Fraud Detection
        - Transactions: 590,540
        - Features: 392
        - Fraud Rate: 3.5%
        **Top Contributing Factors:**
        1. Transaction amount patterns
        2. Card information
        3. Device characteristics
        4. Email domain
        5. Transaction time patterns
        6. Address matching
        7. Purchase category
        8. Historical patterns
        **Preprocessing:**
        - Missing value imputation
        - Categorical encoding
        - Feature scaling
        - Time-aware validation split
        **Machine Learning**
        - XGBoost
        - LightGBM
        - CatBoost
        - Scikit-learn
        **Backend**
        - FastAPI
        - Uvicorn
        - Python 3.10
        - Pandas
        **Frontend**
        - Streamlit
        - Plotly
        - Requests
    FinShield is an AI-powered fraud detection system designed to identify fraudulent 
    online transactions in real-time. Built as a final-year project, it demonstrates 
    the application of modern machine learning techniques to solve real-world financial 
    security challenges.
    **Project Team:** Aman (ML Engineering)
    
    **Institution:** [Your University Name]
    
    **Academic Year:** 2026
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🛡️ FinShield v1.0 | Built with Streamlit & FastAPI | © 2026
    </div>