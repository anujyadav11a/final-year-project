"""
FinShield - Fraud Detection Web Interface
Streamlit application for detecting fraudulent transactions
"""

import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="FinShield - Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .fraud-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .legitimate-badge {
        background-color: #00cc00;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_api_stats():
    """Get API statistics"""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def predict_batch(df):
    """Send batch prediction request to API"""
    try:
        # Convert DataFrame to JSON
        data = df.to_dict(orient='records')
        
        # Send request
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
    """Create pie chart for fraud distribution"""
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
    """Create histogram for fraud probability distribution"""
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
    """Create interactive results table"""
    results_df = pd.DataFrame(results)
    
    # Add original transaction info if available
    if 'TransactionID' in df.columns:
        results_df.insert(0, 'TransactionID', df['TransactionID'].values)
    
    # Format columns
    results_df['fraud_probability'] = results_df['fraud_probability'].round(4)
    results_df['prediction_label'] = results_df['prediction'].map({
        0: '✅ Legitimate',
        1: '🚨 Fraudulent'
    })
    
    return results_df

# Header
st.markdown('<div class="main-header">🛡️ FinShield</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Fraud Detection System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ System Status")
    
    # API Health Check
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
        
        # Get API stats
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
    """)
    
    st.divider()
    
    st.header("📖 Instructions")
    st.markdown("""
    1. Upload a CSV file with transaction data
    2. Ensure it contains all required features
    3. Click "Analyze Transactions"
    4. View results and download predictions
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Predict", "📊 Analytics", "ℹ️ Model Info"])

with tab1:
    st.header("Upload Transaction Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing transaction data with all required features"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(df)} transactions loaded.")
            
            # Show preview
            with st.expander("📋 View Data Preview", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Transactions", f"{len(df):,}")
                with col2:
                    st.metric("Total Features", len(df.columns))
                with col3:
                    st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
            
            # Prediction button
            if st.button("🔍 Analyze Transactions", type="primary", use_container_width=True):
                if not api_healthy:
                    st.error("❌ Cannot make predictions: API is not available")
                else:
                    with st.spinner("Analyzing transactions..."):
                        progress_bar = st.progress(0)
                        
                        # Make prediction
                        results = predict_batch(df)
                        progress_bar.progress(100)
                        
                        if results and 'predictions' in results:
                            predictions = results['predictions']
                            
                            # Store results in session state
                            st.session_state['predictions'] = predictions
                            st.session_state['original_data'] = df
                            st.session_state['processing_time'] = results.get('processing_time', 0)
                            
                            st.success(f"✅ Analysis complete! Processed {len(predictions)} transactions in {results.get('processing_time', 0):.2f}s")
                            
                            # Display summary metrics
                            st.subheader("📈 Summary Statistics")
                            
                            fraud_count = sum(1 for p in predictions if p['prediction'] == 1)
                            legitimate_count = len(predictions) - fraud_count
                            avg_fraud_prob = sum(p['fraud_probability'] for p in predictions) / len(predictions)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "Total Transactions",
                                    f"{len(predictions):,}",
                                    delta=None
                                )
                            
                            with col2:
                                st.metric(
                                    "Fraudulent",
                                    f"{fraud_count:,}",
                                    delta=f"{fraud_count/len(predictions)*100:.1f}%",
                                    delta_color="inverse"
                                )
                            
                            with col3:
                                st.metric(
                                    "Legitimate",
                                    f"{legitimate_count:,}",
                                    delta=f"{legitimate_count/len(predictions)*100:.1f}%",
                                    delta_color="normal"
                                )
                            
                            with col4:
                                st.metric(
                                    "Avg Fraud Score",
                                    f"{avg_fraud_prob:.3f}",
                                    delta=None
                                )
                            
                            # Visualizations
                            st.subheader("📊 Visual Analysis")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig1 = create_fraud_distribution_chart(predictions)
                                st.plotly_chart(fig1, use_container_width=True)
                            
                            with col2:
                                fig2 = create_confidence_distribution(predictions)
                                st.plotly_chart(fig2, use_container_width=True)
                            
                            # Results table
                            st.subheader("📋 Detailed Results")
                            
                            results_df = create_results_table(df, predictions)
                            
                            # Filter options
                            col1, col2 = st.columns(2)
                            with col1:
                                filter_option = st.selectbox(
                                    "Filter by:",
                                    ["All Transactions", "Fraudulent Only", "Legitimate Only"]
                                )
                            
                            with col2:
                                sort_option = st.selectbox(
                                    "Sort by:",
                                    ["Fraud Probability (High to Low)", "Fraud Probability (Low to High)"]
                                )
                            
                            # Apply filters
                            display_df = results_df.copy()
                            
                            if filter_option == "Fraudulent Only":
                                display_df = display_df[display_df['prediction'] == 1]
                            elif filter_option == "Legitimate Only":
                                display_df = display_df[display_df['prediction'] == 0]
                            
                            # Apply sorting
                            if sort_option == "Fraud Probability (High to Low)":
                                display_df = display_df.sort_values('fraud_probability', ascending=False)
                            else:
                                display_df = display_df.sort_values('fraud_probability', ascending=True)
                            
                            # Display table
                            st.dataframe(
                                display_df[['prediction_label', 'fraud_probability', 'confidence']].head(100),
                                use_container_width=True,
                                height=400
                            )
                            
                            if len(display_df) > 100:
                                st.info(f"Showing top 100 of {len(display_df)} transactions")
                            
                            # Download results
                            st.subheader("💾 Download Results")
                            
                            # Prepare CSV for download
                            output_df = df.copy()
                            output_df['fraud_prediction'] = [p['prediction'] for p in predictions]
                            output_df['fraud_probability'] = [p['fraud_probability'] for p in predictions]
                            output_df['confidence'] = [p['confidence'] for p in predictions]
                            
                            csv = output_df.to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Download Predictions (CSV)",
                                data=csv,
                                file_name=f"finshield_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted and contains all required features.")

with tab2:
    st.header("📊 Analytics Dashboard")
    
    if 'predictions' in st.session_state:
        predictions = st.session_state['predictions']
        df = st.session_state['original_data']
        
        # Key metrics
        fraud_count = sum(1 for p in predictions if p['prediction'] == 1)
        legitimate_count = len(predictions) - fraud_count
        
        st.subheader("🎯 Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Detection Rate", f"{fraud_count/len(predictions)*100:.2f}%")
        
        with col2:
            st.metric("Processing Time", f"{st.session_state['processing_time']:.2f}s")
        
        with col3:
            st.metric("Throughput", f"{len(predictions)/st.session_state['processing_time']:.0f} tx/s")
        
        # Distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = create_fraud_distribution_chart(predictions)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = create_confidence_distribution(predictions)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Risk categorization
        st.subheader("⚠️ Risk Categories")
        
        high_risk = sum(1 for p in predictions if p['fraud_probability'] >= 0.7)
        medium_risk = sum(1 for p in predictions if 0.3 <= p['fraud_probability'] < 0.7)
        low_risk = sum(1 for p in predictions if p['fraud_probability'] < 0.3)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔴 High Risk", high_risk, help="Fraud probability ≥ 70%")
        
        with col2:
            st.metric("🟡 Medium Risk", medium_risk, help="Fraud probability 30-70%")
        
        with col3:
            st.metric("🟢 Low Risk", low_risk, help="Fraud probability < 30%")
        
    else:
        st.info("📤 Upload and analyze transactions to view analytics")

with tab3:
    st.header("ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Details")
        st.markdown("""
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
        """)
    
    with col2:
        st.subheader("🎯 Key Features")
        st.markdown("""
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
        """)
    
    st.divider()
    
    st.subheader("⚙️ Technical Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Machine Learning**
        - XGBoost
        - LightGBM
        - CatBoost
        - Scikit-learn
        """)
    
    with col2:
        st.markdown("""
        **Backend**
        - FastAPI
        - Uvicorn
        - Python 3.10
        - Pandas
        """)
    
    with col3:
        st.markdown("""
        **Frontend**
        - Streamlit
        - Plotly
        - Requests
        """)
    
    st.divider()
    
    st.subheader("📚 About FinShield")
    st.markdown("""
    FinShield is an AI-powered fraud detection system designed to identify fraudulent 
    online transactions in real-time. Built as a final-year project, it demonstrates 
    the application of modern machine learning techniques to solve real-world financial 
    security challenges.
    
    **Project Team:** Aman (ML Engineering)
    
    **Institution:** [Your University Name]
    
    **Academic Year:** 2026
    """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🛡️ FinShield v1.0 | Built with Streamlit & FastAPI | © 2026
    </div>
    """, unsafe_allow_html=True)
