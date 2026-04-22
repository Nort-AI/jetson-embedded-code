"""
test_heartbeat.py — Quick smoke test for the device heartbeat endpoint.
Run this BEFORE starting the full tracking system to confirm:
  1. device.json is correctly filled in
  2. API_URL is reachable
  3. DEVICE_KEY matches what's in Cloud Run env vars

Usage:
    cd Jetson-Embedded-code
    python test_heartbeat.py
"""
import json
import sys
import requests

with open("device.json") as f:
    d = json.load(f)

api_url    = d.get("api_url", "")
device_key = d.get("device_key", "")

if not api_url:
    print("ERROR: api_url not set in device.json")
    sys.exit(1)

print(f"Testing heartbeat: {api_url}")
print(f"  client_id = {d['client_id']}")
print(f"  store_id  = {d['store_id']}")
print(f"  device_id = {d['device_id']}")
print()

try:
    r = requests.post(
        f"{api_url}/api/v1/devices/heartbeat",
        json={
            "client_id":      d["client_id"],
            "store_id":       d["store_id"],
            "device_id":      d["device_id"],
            "cameras_active": 0,
            "fps_avg":        0.0,
            "disk_usage_pct": 5.0,
            "uptime_seconds": 10,
        },
        headers={"Authorization": f"Bearer {device_key}"},
        timeout=10,
    )
    print(f"HTTP {r.status_code}")
    print(r.json())

    if r.status_code == 200:
        print("\n✅ Heartbeat OK — device.json and API are correctly configured.")
    elif r.status_code == 401:
        print("\n❌ 401 Unauthorized — device_key in device.json doesn't match DEVICE_API_KEY on the API.")
    else:
        print(f"\n⚠️  Unexpected status {r.status_code}")

except requests.exceptions.ConnectionError:
    print(f"\n❌ Cannot reach {api_url}")
    print("   Check: is the API deployed? Is the URL correct in device.json?")
except Exception as e:
    print(f"\n❌ Error: {e}")
