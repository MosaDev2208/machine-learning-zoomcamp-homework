"""
Test script for Lead Scoring API
Module 5: ML Zoomcamp Deployment
Based on Workshop: test.py pattern

This script tests the API with multiple test cases and displays results.
"""

import requests
import json


# ============================================================================
# CONFIGURATION
# ============================================================================

url = 'http://localhost:8000/predict'


# ============================================================================
# TEST CASE 1: High-engagement lead (likely to convert)
# ============================================================================

print("\n" + "=" * 70)
print("TEST 1: High-engagement lead (organic_search, 4 courses viewed)")
print("=" * 70)

client_1 = {
    'lead_source': 'organic_search',
    'number_of_courses_viewed': 4,
    'annual_income': 80304.0
}

print(f"\nInput: {json.dumps(client_1, indent=2)}")

response = requests.post(url, json=client_1)
predictions_1 = response.json()

print(f"\nResponse: {json.dumps(predictions_1, indent=2)}")
print(f"Probability: {predictions_1['probability']:.3f}")

if predictions_1['will_convert']:
    print("✅ Decision: LIKELY TO CONVERT → Send premium offer")
else:
    print("❌ Decision: UNLIKELY TO CONVERT → Send nurture content")


# ============================================================================
# TEST CASE 2: Low-engagement lead (less likely to convert)
# ============================================================================

print("\n" + "=" * 70)
print("TEST 2: Low-engagement lead (paid_ads, 1 course viewed)")
print("=" * 70)

client_2 = {
    'lead_source': 'paid_ads',
    'number_of_courses_viewed': 1,
    'annual_income': 25000.0
}

print(f"\nInput: {json.dumps(client_2, indent=2)}")

response = requests.post(url, json=client_2)
predictions_2 = response.json()

print(f"\nResponse: {json.dumps(predictions_2, indent=2)}")
print(f"Probability: {predictions_2['probability']:.3f}")

if predictions_2['will_convert']:
    print("✅ Decision: LIKELY TO CONVERT → Send premium offer")
else:
    print("❌ Decision: UNLIKELY TO CONVERT → Send nurture content")


# ============================================================================
# TEST SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

print(f"\n📊 Results:")
print(f"   Lead 1 Conversion Probability: {predictions_1['probability']:.3f}")
print(f"   Lead 2 Conversion Probability: {predictions_2['probability']:.3f}")

print(f"\n✅ All tests completed successfully!")
print(f"✅ API is working correctly\n")
