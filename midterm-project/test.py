#!/usr/bin/env python
# coding: utf-8

import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Test cases
test_cases = [
    {
        "name": "Normal Operation (Low Risk)",
        "data": {
            "type": "M",
            "air_temperature": 298.0,
            "process_temperature": 308.0,
            "rotational_speed": 1500,
            "torque": 40.0,
            "tool_wear": 50
        }
    },
    {
        "name": "High Risk - High Temperature",
        "data": {
            "type": "L",
            "air_temperature": 303.0,
            "process_temperature": 314.0,
            "rotational_speed": 1200,
            "torque": 55.0,
            "tool_wear": 200
        }
    },
    {
        "name": "High Risk - High Torque & Tool Wear",
        "data": {
            "type": "H",
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1100,
            "torque": 60.0,
            "tool_wear": 220
        }
    }
]

print("=" * 80)
print("TESTING PREDICTIVE MAINTENANCE API")
print("=" * 80)

# Test health endpoint
print("\n1. Testing Health Endpoint...")
try:
    response = requests.get("http://localhost:8000/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")
    print("   Make sure the API is running: python predict.py")
    exit(1)

# Test prediction endpoint
print("\n2. Testing Prediction Endpoint...")
for i, test_case in enumerate(test_cases, 1):
    print(f"\n   Test Case {i}: {test_case['name']}")
    print(f"   Input: {json.dumps(test_case['data'], indent=6)}")
    
    try:
        response = requests.post(url, json=test_case['data'])
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Prediction: {'FAILURE' if result['failure_prediction'] == 1 else 'NO FAILURE'}")
            print(f"   ✓ Probability: {result['failure_probability']:.2%}")
            print(f"   ✓ Risk Level: {result['risk_level']}")
            print(f"   ✓ Recommendation: {result['recommendation']}")
        else:
            print(f"   ✗ Error: {response.status_code}")
            print(f"   ✗ Message: {response.json()}")
    
    except Exception as e:
        print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
print("✅ TESTING COMPLETE!")
print("=" * 80)
