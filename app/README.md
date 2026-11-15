# FinShield Web Interface

Interactive web application for fraud detection using Streamlit.

## 🚀 Quick Start

### Prerequisites

1. **API must be running**: Start the FastAPI backend first
   ```bash
   python api\start_api.py
   ```
   API will be available at: http://localhost:8000

2. **Dependencies installed**: Ensure all packages are installed
   ```bash
   conda install streamlit requests -y
   ```

### Starting the Application

```bash
python app\start_streamlit.py
```

The web interface will open automatically at: **http://localhost:8501**

## 📋 Features

### 1. Upload & Predict Tab
- **File Upload**: Upload CSV files with transaction data
- **Data Preview**: View uploaded data statistics
- **Batch Prediction**: Analyze all transactions at once
- **Summary Statistics**: View fraud detection metrics
- **Visual Analysis**: Interactive charts and graphs
- **Detailed Results**: Sortable and filterable results table
- **Download Results**: Export predictions as CSV

### 2. Analytics Dashboard Tab
- **Performance Metrics**: Detection rate, processing time, throughput
- **Distribution Charts**: Fraud vs legitimate breakdown
- **Risk Categories**: High/medium/low risk segmentation
- **Probability Distribution**: Fraud score histogram

### 3. Model Info Tab
- **Model Details**: Algorithm, performance metrics
- **Key Features**: Top contributing factors
- **Technical Stack**: Technologies used
- **Project Information**: About FinShield

## 📊 Using the Interface

### Step 1: Ensure API is Running
Before using the web interface, make sure the API is running:
```bash
python api\start_api.py
```

Check the sidebar for "✅ API Connected" status.

### Step 2: Upload Transaction Data
1. Click "Browse files" or drag & drop a CSV file
2. File should contain transaction features (393 columns expected)
3. View data preview to confirm upload

### Step 3: Analyze Transactions
1. Click "🔍 Analyze Transactions" button
2. Wait for processing (progress bar will show status)
3. View results in multiple formats:
   - Summary metrics
   - Pie chart (fraud distribution)
   - Histogram (probability distribution)
   - Detailed results table

### Step 4: Filter and Sort Results
- **Filter by**: All / Fraudulent Only / Legitimate Only
- **Sort by**: Fraud probability (high to low or low to high)
- View top 100 transactions in the table

### Step 5: Download Results
Click "📥 Download Predictions (CSV)" to save:
- All original transaction data
- Fraud prediction (0 or 1)
- Fraud probability (0.0 to 1.0)
- Confidence score

## 📁 Test Data

A sample test file is provided: `sample_transactions.csv`

This file contains 100 transactions (mix of fraud and legitimate) that you can use to test the interface.

To create a new sample file:
```bash
python app\create_test_data.py
```

## 🎨 Interface Components

### API Status Indicator
- ✅ **Green**: API connected and ready
- ❌ **Red**: API disconnected (check if API is running)

### Metrics Display
- **Total Transactions**: Number of transactions analyzed
- **Fraudulent**: Count and percentage of fraud cases
- **Legitimate**: Count and percentage of legitimate transactions
- **Avg Fraud Score**: Average fraud probability across all transactions

### Risk Categories
- 🔴 **High Risk**: Fraud probability ≥ 70%
- 🟡 **Medium Risk**: Fraud probability 30-70%
- 🟢 **Low Risk**: Fraud probability < 30%

## 🛠️ Technical Details

### Architecture
```
User Browser (Streamlit UI)
       ↓
    HTTP POST
       ↓
FastAPI Backend (localhost:8000)
       ↓
  ML Model (XGBoost)
       ↓
   Predictions
```

### Technology Stack
- **Frontend**: Streamlit 1.64.0
- **Visualization**: Plotly Express
- **HTTP Client**: Requests
- **Data Processing**: Pandas

### File Structure
```
app/
├── streamlit_app.py        # Main Streamlit application
├── start_streamlit.py      # Startup script
├── create_test_data.py     # Test data generation
├── sample_transactions.csv # Sample test data
└── README.md              # This file
```

## 🔧 Configuration

### API URL
Default: `http://localhost:8000`

To change the API URL, edit `streamlit_app.py`:
```python
API_URL = "http://localhost:8000"  # Change this
```

### Port Configuration
Default port: 8501

To change the port, edit `start_streamlit.py`:
```python
"--server.port=8501",  # Change this
```

Or run directly:
```bash
streamlit run app\streamlit_app.py --server.port=8502
```

## 🐛 Troubleshooting

### API Disconnected Error
**Problem**: Red "❌ API Disconnected" in sidebar

**Solution**:
1. Start the API: `python api\start_api.py`
2. Wait for "✅ Model loaded successfully"
3. Refresh the Streamlit page

### File Upload Error
**Problem**: "Error processing file"

**Solution**:
- Ensure CSV has all required features (393 columns)
- Check for proper CSV formatting
- Use `sample_transactions.csv` to verify system works

### Prediction Error
**Problem**: "API Error: 500"

**Solution**:
- Check API logs for detailed error
- Ensure data has correct feature names
- Verify no missing required columns

### Slow Performance
**Problem**: Predictions take too long

**Solution**:
- Process smaller batches (< 1000 transactions)
- Ensure API is running locally (not remote)
- Check system resources (CPU/RAM)

## 📈 Performance

### Typical Processing Times
- 100 transactions: ~0.5 seconds
- 1,000 transactions: ~2 seconds
- 10,000 transactions: ~15 seconds

### Throughput
- Average: ~500-1000 transactions/second
- Depends on: CPU speed, model complexity, data size

## 🎯 Use Cases

### 1. Real-time Monitoring
Upload recent transaction batches to identify fraud patterns in real-time.

### 2. Historical Analysis
Analyze past transactions to find missed fraud cases or validate model performance.

### 3. Risk Assessment
Categorize transactions by risk level for manual review prioritization.

### 4. Model Validation
Test model performance on new data before deployment.

## 📚 Additional Resources

### Related Files
- API Documentation: `api/README.md`
- Model Training: `src/finshield/train.py`
- Project Summary: `PROJECT_SUMMARY.md`

### Model Information
- Algorithm: XGBoost Classifier
- ROC-AUC Score: 88.34%
- Recall: 79.48%
- Training Data: IEEE-CIS Fraud Detection

## 🤝 Support

For issues or questions:
1. Check API logs: `api/start_api.py` output
2. Review error messages in Streamlit interface
3. Verify data format matches requirements
4. Test with provided `sample_transactions.csv`

## 📝 Notes

- **Data Privacy**: All processing happens locally on your machine
- **No Data Storage**: Predictions are not saved by the application
- **Download Results**: Use download button to save predictions
- **Browser Support**: Best viewed in Chrome, Firefox, or Edge

---

**FinShield v1.0** | Built with ❤️ using Streamlit & FastAPI
