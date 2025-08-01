#!/usr/bin/env python3
"""
Railway Server Webhook Tester
Tests the correct payload format for your Railway server
"""

import json
import urllib.request
import urllib.error
import base64
from datetime import datetime

# Railway server endpoint
RAILWAY_URL = "https://building-sensor-platform-production.up.railway.app/webhook/fall-alert"

def test_railway_webhook():
    """Test the Railway server with the correct payload format"""
    
    timestamp = datetime.now().isoformat()
    
    # Required fields payload (minimal test)
    minimal_payload = {
        "alert_type": "fall",
        "camera_name": "Camera_01_Test",
        "person_id": "person_123",
        "location": "Test Location"
    }
    
    # Complete payload (full test)
    complete_payload = {
        "alert_type": "fall",
        "camera_name": "Camera_01_Lobby",
        "person_id": "person_123",
        "confidence": 0.92,
        "location": "Building Lobby - East Entrance",
        "severity": "critical",
        "title": "Fall Alert Detected",
        "message": "Fall detected with high confidence",
        "requires_immediate_attention": True,
        "notification_type": "fall_detection",
        "timestamp": timestamp,
        "metadata": {"test": "from_python_script"},
        "actions": [
            {"action": "view_camera", "title": "View Camera"},
            {"action": "acknowledge", "title": "Acknowledge"}
        ]
    }
    
    print("🧪 Testing Railway Server Webhook")
    print("=" * 60)
    print(f"🎯 Target: {RAILWAY_URL}")
    print()
    
    # Test 1: Minimal payload (required fields only)
    print("🔍 Test 1: Minimal Payload (Required Fields Only)")
    print(f"📝 Payload: {json.dumps(minimal_payload, indent=2)}")
    success = send_webhook_payload(minimal_payload, "Minimal")
    
    if success:
        print("🎉 SUCCESS! Minimal payload works!")
    else:
        print("❌ Minimal payload failed")
    
    print("-" * 60)
    
    # Test 2: Complete payload
    print("🔍 Test 2: Complete Payload (All Fields)")
    print(f"📝 Payload size: {len(json.dumps(complete_payload))} characters")
    success = send_webhook_payload(complete_payload, "Complete")
    
    if success:
        print("🎉 SUCCESS! Complete payload works!")
    else:
        print("❌ Complete payload failed")
    
    print("\n" + "=" * 60)

def send_webhook_payload(payload_dict, test_name=""):
    """Send a payload to the Railway server"""
    try:
        # Convert payload to JSON
        json_data = json.dumps(payload_dict)
        json_bytes = json_data.encode('utf-8')
        
        # Create request
        req = urllib.request.Request(
            RAILWAY_URL,
            data=json_bytes,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'FallDetectionTest/1.0',
                'X-Alert-Type': 'fall',
                'X-Sensor-Type': 'fall_detection'
            },
            method='POST'
        )
        
        # Send request
        with urllib.request.urlopen(req, timeout=10) as response:
            response_data = response.read().decode('utf-8')
            print(f"✅ {test_name} SUCCESS!")
            print(f"   Status: {response.status}")
            print(f"   Response: {response_data}")
            return True
            
    except urllib.error.HTTPError as e:
        error_response = e.read().decode('utf-8')
        print(f"❌ {test_name} FAILED")
        print(f"   Status: {e.code}")
        print(f"   Response: {error_response}")
        
        # Try to parse error for specific details
        try:
            error_json = json.loads(error_response)
            if "missing_fields" in error_json:
                print(f"   Missing fields: {error_json['missing_fields']}")
            if "error" in error_json:
                print(f"   Error: {error_json['error']}")
        except:
            pass
            
        return False
    except urllib.error.URLError as e:
        print(f"❌ {test_name} CONNECTION ERROR: {e}")
        return False
    except Exception as e:
        print(f"❌ {test_name} ERROR: {e}")
        return False

if __name__ == "__main__":
    test_railway_webhook()
