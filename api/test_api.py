import requests
import json
import pandas as pd
from pathlib import Path
BASE_URL = "http://localhost:8000"

def test_root():
    print("\n" + "="*60)
    print("TEST 1: Root Endpoint")
    print("="*60)
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✅ Root endpoint test passed!")


def test_health():
    print("\n" + "="*60)
    print("TEST 2: Health Check")
    print("="*60)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["model_loaded"] == True
    print("✅ Health check test passed!")


def test_single_prediction():
    print("\n" + "="*60)
    print("TEST 3: Single Prediction")
    print("="*60)
    transaction = {
        "TransactionDT": 86400,
        "TransactionAmt": 68.5,
        "ProductCD": "W",
        "card1": 13926.0,
        "card2": 150.0,
        "card3": 150.0,
        "card4": "discover",
        "card5": 142.0,
        "card6": "credit",
        "addr1": 315.0,
        "addr2": 87.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=transaction)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Result:")
        print(f"  Probability: {result['probability']:.4f}")
        print(f"  Prediction: {result['prediction']} ({result['label']})")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Risk Level: {result['risk_level']}")
        print("✅ Single prediction test passed!")
    else:
        print(f"❌ Error: {response.text}")
        

def test_batch_prediction():
    print("\n" + "="*60)
    print("TEST 4: Batch Prediction")
    print("="*60)
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'train_transaction.csv'
    
    if not data_path.exists():
        print("⚠️ Dataset not found, skipping batch test")
        return
    
    df = pd.read_csv(data_path, nrows=10)
    
    temp_csv = Path(__file__).parent / 'temp_test.csv'
    df.to_csv(temp_csv, index=False)
    
    with open(temp_csv, 'rb') as f:
        files = {'file': ('test_transactions.csv', f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/predict/batch", files=files)
    
    temp_csv.unlink()
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nBatch Results:")
        print(f"  Total Transactions: {result['total_transactions']}")
        print(f"  Fraud Count: {result['fraud_count']}")
        print(f"  Fraud Percentage: {result['fraud_percentage']:.2f}%")
        print(f"\nFirst 3 Predictions:")
        for pred in result['predictions'][:3]:
            print(f"    ID: {pred['transaction_id']} | "
                  f"Prob: {pred['probability']:.4f} | "
                  f"Label: {pred['label']} | "
                  f"Risk: {pred['risk_level']}")
        print("✅ Batch prediction test passed!")
    else:
        print(f"❌ Error: {response.text}")


def test_stats():
    print("\n" + "="*60)
    print("TEST 5: Model Statistics")
    print("="*60)
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nModel Info:")
        print(f"  Type: {result['model_info']['type']}")
        print(f"  Algorithms: {', '.join(result['model_info']['algorithms'])}")
        print(f"  ROC-AUC: {result['model_info']['performance']['roc_auc']}")
        print("✅ Statistics test passed!")
    else:
        print(f"❌ Error: {response.text}")


def test_fraud_scenarios():
    print("\n" + "="*60)
    print("TEST 6: Fraud Scenario Testing")
    print("="*60)
    scenarios = [
        {
            "name": "High-value transaction",
            "data": {
                "TransactionDT": 86400,
                "TransactionAmt": 5000.0,
                "ProductCD": "W",
                "card1": 13926.0,
                "card6": "debit"
            }
        },
        {
            "name": "Low-value transaction",
            "data": {
                "TransactionDT": 86400,
                "TransactionAmt": 10.0,
                "ProductCD": "C",
                "card1": 13926.0,
                "card6": "credit"
            }
        },
        {
            "name": "Unusual product",
            "data": {
                "TransactionDT": 86400,
                "TransactionAmt": 150.0,
                "ProductCD": "R",
                "card1": 13926.0,
                "card6": "debit"
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        response = requests.post(f"{BASE_URL}/predict", json=scenario['data'])
        
        if response.status_code == 200:
            result = response.json()
            print(f"  Probability: {result['probability']:.4f} | "
                  f"Prediction: {result['label']} | "
                  f"Risk: {result['risk_level']}")
        else:
            print(f"  ❌ Error: {response.text}")
    
    print("✅ Scenario testing complete!")


def run_all_tests():
    print("\n" + "="*70)
    print("FINSHIELD API TESTING SUITE")
    print("="*70)
    print("Starting API tests...")
    print(f"API URL: {BASE_URL}")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        print("✅ API is running!")
    except requests.exceptions.ConnectionError:
        print("❌ Error: API is not running!")
        print("Please start the API with: python api/main.py")
        return
    
    tests = [
        test_root,
        test_health,
        test_single_prediction,
        test_batch_prediction,
        test_stats,
        test_fraud_scenarios
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success Rate: {(passed/len(tests))*100:.1f}%")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
