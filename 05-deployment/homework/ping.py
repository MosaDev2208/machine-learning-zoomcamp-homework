"""
Health check script for Lead Scoring API
Module 5: ML Zoomcamp Deployment
"""

import requests
import json

url = 'http://localhost:8000/'

print("🔍 Pinging API health check endpoint...")
try:
    response = requests.get(url, timeout=5)
    print(f"✅ Status Code: {response.status_code}")
    print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")
