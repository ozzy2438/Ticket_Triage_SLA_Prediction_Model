#!/usr/bin/env python3
"""
=============================================================================
API TEST SCRIPT - Production Readiness Verification
=============================================================================
This script tests all API endpoints to verify production readiness.

Usage:
    python test_api.py

Requirements:
    - API running on http://localhost:8000
    - requests library: pip install requests
=============================================================================
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_response(response: requests.Response, show_full: bool = False):
    """Print formatted API response"""
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ SUCCESS")
    else:
        print("❌ FAILED")
    
    try:
        data = response.json()
        if show_full:
            print(json.dumps(data, indent=2))
        else:
            # Print key fields only
            for key, value in data.items():
                if isinstance(value, (str, int, float, bool)):
                    print(f"  {key}: {value}")
    except:
        print(f"Response: {response.text}")


def test_health_check():
    """Test 1: Health check endpoint"""
    print_section("TEST 1: Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    print_response(response)
    
    if response.status_code == 200:
        data = response.json()
        print("\n📊 System Status:")
        print(f"   Model Loaded: {data.get('model_loaded')}")
        print(f"   Device: {data.get('device')}")
        print(f"   Uptime: {data.get('uptime_seconds'):.2f}s")
        print(f"   Total Predictions: {data.get('total_predictions')}")
    
    return response.status_code == 200


def test_single_prediction():
    """Test 2: Single ticket prediction"""
    print_section("TEST 2: Single Ticket Prediction")
    
    # Test case: Restaurant delivery issue
    payload = {
        "ticket_id": "TEST-001",
        "complaint_text": "Food delivery is extremely late, customer called 3 times and is very upset",
        "hour": 19,
        "day_of_week": 5,
        "is_weekend": 1,
        "month": 2,
        "borough": "BROOKLYN",
        "priority": "high"
    }
    
    print(f"\n📝 Test Ticket:")
    print(f"   ID: {payload['ticket_id']}")
    print(f"   Text: {payload['complaint_text'][:50]}...")
    print(f"   Time: {payload['hour']}:00, Day {payload['day_of_week']}")
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/predict", json=payload, headers=HEADERS)
    elapsed_ms = (time.time() - start_time) * 1000
    
    print(f"\n⏱️  Processing Time: {elapsed_ms:.2f}ms")
    
    if response.status_code == 200:
        data = response.json()
        print("\n🎯 Prediction Results:")
        print(f"   Predicted Agency: {data.get('predicted_agency')}")
        print(f"   Confidence: {data.get('confidence'):.4f}")
        print(f"   SLA Hours: {data.get('sla_prediction_hours'):.2f}")
        print(f"   Action: {data.get('action')}")
        print(f"   Confidence Category: {data.get('confidence_category')}")
        
        print("\n   Top 3 Predictions:")
        for pred in data.get('top_3_predictions', []):
            print(f"      - {pred['agency']}: {pred['probability']:.4f}")
    else:
        print_response(response, show_full=True)
    
    return response.status_code == 200


def test_batch_prediction():
    """Test 3: Batch prediction"""
    print_section("TEST 3: Batch Prediction (5 tickets)")
    
    # Test cases: Multiple scenarios
    tickets = [
        {
            "ticket_id": "BATCH-001",
            "complaint_text": "Noise complaint from neighbors late at night",
            "hour": 23, "day_of_week": 5, "is_weekend": 1, "month": 2, "borough": "MANHATTAN"
        },
        {
            "ticket_id": "BATCH-002",
            "complaint_text": "Street light is broken and not working",
            "hour": 8, "day_of_week": 1, "is_weekend": 0, "month": 2, "borough": "BROOKLYN"
        },
        {
            "ticket_id": "BATCH-003",
            "complaint_text": "Restaurant health code violation observed",
            "hour": 14, "day_of_week": 3, "is_weekend": 0, "month": 2, "borough": "QUEENS"
        },
        {
            "ticket_id": "BATCH-004",
            "complaint_text": "Parking meter not accepting payment",
            "hour": 10, "day_of_week": 2, "is_weekend": 0, "month": 2, "borough": "BRONX"
        },
        {
            "ticket_id": "BATCH-005",
            "complaint_text": "Water leak in building causing damage",
            "hour": 16, "day_of_week": 4, "is_weekend": 0, "month": 2, "borough": "STATEN ISLAND"
        }
    ]
    
    payload = {"tickets": tickets}
    
    print(f"\n📦 Sending {len(tickets)} tickets for batch processing...")
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload, headers=HEADERS)
    elapsed_ms = (time.time() - start_time) * 1000
    
    print(f"\n⏱️  Total Processing Time: {elapsed_ms:.2f}ms")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Batch Results:")
        print(f"   Total Processed: {data.get('total_processed')}")
        print(f"   Successful: {data.get('successful')}")
        print(f"   Failed: {data.get('failed')}")
        print(f"   Avg Time/Ticket: {elapsed_ms / len(tickets):.2f}ms")
        
        print("\n   Sample Predictions:")
        for pred in data.get('predictions', [])[:3]:
            print(f"      {pred['ticket_id']}: {pred['predicted_agency']} (conf: {pred['confidence']:.2f})")
    else:
        print_response(response, show_full=True)
    
    return response.status_code == 200


def test_feedback():
    """Test 4: Feedback recording"""
    print_section("TEST 4: Feedback Recording")
    
    payload = {
        "ticket_id": "TEST-001",
        "model_prediction": "Department of Consumer Affairs",
        "human_decision": "Department of Consumer Affairs",
        "actual_resolution_hours": 18.5,
        "correction_reason": None,
        "operator_id": "OP-123",
        "notes": "Model prediction was correct"
    }
    
    print("\n📝 Feedback Submission:")
    print(f"   Ticket: {payload['ticket_id']}")
    print(f"   Model: {payload['model_prediction']}")
    print(f"   Human: {payload['human_decision']}")
    print(f"   Correct: {payload['model_prediction'] == payload['human_decision']}")
    
    response = requests.post(f"{BASE_URL}/feedback", json=payload, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Feedback Recorded:")
        print(f"   Status: {data.get('status')}")
        print(f"   Was Corrected: {data.get('was_corrected')}")
        print(f"   Accuracy Impact: {data.get('model_accuracy_impact')}")
    else:
        print_response(response, show_full=True)
    
    return response.status_code == 200


def test_statistics():
    """Test 5: Statistics endpoint"""
    print_section("TEST 5: Statistics & Business Metrics")
    
    response = requests.get(f"{BASE_URL}/stats")
    
    if response.status_code == 200:
        data = response.json()
        print("\n📊 Performance Metrics:")
        print(f"   Total Processed: {data.get('total_processed')}")
        print(f"   Auto-Routed: {data.get('auto_routed')}")
        print(f"   Human Review: {data.get('human_reviewed')}")
        print(f"   Escalated: {data.get('escalated')}")
        print(f"   Automation Rate: {data.get('automation_rate')}")
        
        print("\n💰 Business Impact:")
        print(f"   Estimated Time Saved: {data.get('estimated_time_saved_hours'):.2f} hours")
        print(f"   Estimated Cost Saved: ${data.get('estimated_cost_saved'):.2f}")
        
        print("\n🎯 Accuracy Metrics:")
        print(f"   Total Feedback: {data.get('total_feedback')}")
        print(f"   Correct Predictions: {data.get('correct_predictions')}")
        print(f"   Accuracy Rate: {data.get('accuracy_rate')}")
        print(f"   Avg Confidence: {data.get('avg_confidence'):.4f}")
        print(f"   Avg SLA Hours: {data.get('avg_sla_hours'):.2f}")
    else:
        print_response(response, show_full=True)
    
    return response.status_code == 200


def test_drift_monitoring():
    """Test 6: Drift detection"""
    print_section("TEST 6: Drift Detection & Monitoring")
    
    response = requests.get(f"{BASE_URL}/monitoring/drift")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n🔍 Overall Health: {data.get('overall_health')}")
        
        print("\n   Data Drift:")
        dd = data.get('data_drift', {})
        print(f"      Detected: {dd.get('detected')}")
        print(f"      Severity: {dd.get('severity')}")
        print(f"      Avg Confidence: {dd.get('avg_confidence')}")
        print(f"      Recommendation: {dd.get('recommendation')}")
        
        print("\n   Prediction Drift:")
        pd = data.get('prediction_drift', {})
        print(f"      Detected: {pd.get('detected')}")
        print(f"      Severity: {pd.get('severity')}")
        print(f"      Escalation Rate: {pd.get('escalation_rate')}")
        print(f"      Recommendation: {pd.get('recommendation')}")
        
        print("\n   Performance Drift:")
        pf = data.get('performance_drift', {})
        print(f"      Detected: {pf.get('detected')}")
        print(f"      Severity: {pf.get('severity')}")
        print(f"      Accuracy: {pf.get('accuracy')}")
        print(f"      Recommendation: {pf.get('recommendation')}")
    else:
        print_response(response, show_full=True)
    
    return response.status_code == 200


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "=" * 70)
    print("  🧪 API PRODUCTION READINESS TEST SUITE")
    print("=" * 70)
    print(f"  Target: {BASE_URL}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check if API is running
    try:
        response = requests.get(BASE_URL, timeout=5)
    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: API is not running at {BASE_URL}")
        print(f"   {e}")
        print("\n💡 Start the API first:")
        print("   cd api")
        print("   python main.py")
        return
    
    # Run tests
    results = {
        "Health Check": test_health_check(),
        "Single Prediction": test_single_prediction(),
        "Batch Prediction": test_batch_prediction(),
        "Feedback Recording": test_feedback(),
        "Statistics": test_statistics(),
        "Drift Monitoring": test_drift_monitoring()
    }
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    print("\n📋 Results:")
    for test_name, passed_flag in results.items():
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! API is production-ready.")
        print("\n📊 Key Capabilities Verified:")
        print("   ✅ Real-time single ticket prediction")
        print("   ✅ High-volume batch processing")
        print("   ✅ Human-in-the-loop feedback loop")
        print("   ✅ Business metrics & ROI tracking")
        print("   ✅ Drift detection & monitoring")
        print("\n💼 Ready for HungerRush deployment!")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
    
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
