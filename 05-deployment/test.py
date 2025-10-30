import requests
import json

# Test server URL
url = 'http://localhost:8080/predict'

# ============================================
# TEST CASE: Realistic Churning Customer
# ============================================
# Customer: Female, new to service (1 month),
# month-to-month contract, low charges
# Expected: HIGH CHURN RISK (probability ~0.66)

customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

print("=" * 80)
print("🧪 TESTING CHURN PREDICTION API")
print("=" * 80)

print("\n📊 Test Customer Profile:")
print(f"  Gender: {customer['gender']}")
print(f"  Tenure: {customer['tenure']} months")
print(f"  Monthly Charges: ${customer['monthlycharges']}")
print(f"  Contract: {customer['contract']}")

print("\n🔗 Sending request to:", url)
print("\n📤 Request body:")
print(json.dumps(customer, indent=2))

try:
    response = requests.post(url, json=customer)
    
    print("\n📥 Response:")
    print(f"Status Code: {response.status_code}")
    
    predictions = response.json()
    print("\nResponse Body:")
    print(json.dumps(predictions, indent=2))
    
    # Parse results
    prob = predictions['churn_probability']
    will_churn = predictions['churn']
    
    print("\n" + "=" * 80)
    print("✅ PREDICTION RESULTS")
    print("=" * 80)
    print(f"Churn Probability: {prob:.4f} ({prob*100:.2f}%)")
    print(f"Will Churn: {will_churn}")
    
    if will_churn:
        print("\n🔴 Status: HIGH CHURN RISK")
        print("💡 Action: Send retention offer/discount")
    else:
        print("\n🟢 Status: LOW CHURN RISK")
        print("✅ Action: Maintain relationship")
    
    print("=" * 80 + "\n")
    
except requests.exceptions.ConnectionError:
    print("❌ ERROR: Could not connect to server!")
    print("Make sure the API is running: python predict.py")
except Exception as e:
    print(f"❌ ERROR: {e}")
