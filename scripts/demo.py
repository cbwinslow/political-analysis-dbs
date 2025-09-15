#!/usr/bin/env python3
"""
Quick demo of the Political Analysis Database API
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from fastapi.testclient import TestClient
from app.main import app

def demo_api():
    """Demo the API endpoints"""
    print("🚀 Political Analysis Database Demo")
    print("=" * 50)
    
    # Create test client
    client = TestClient(app)
    
    print("1. Testing health endpoint...")
    response = client.get("/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    print("\n2. Testing root endpoint...")
    response = client.get("/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    print("\n3. Testing legislators endpoint...")
    response = client.get("/api/v1/legislators/")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Found {len(response.json())} legislators")
    else:
        print(f"   Error: {response.text}")
    
    print("\n4. Testing bills endpoint...")
    response = client.get("/api/v1/bills/")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Found {len(response.json())} bills")
    else:
        print(f"   Error: {response.text}")
    
    print("\n5. Creating a sample legislator...")
    sample_legislator = {
        "name": "Demo Legislator",
        "party": "Demo Party",
        "state": "DC",
        "district": "1",
        "chamber": "house",
        "bio_text": "A demo legislator for testing purposes"
    }
    
    # This will likely fail without a database connection, but shows the endpoint works
    response = client.post("/api/v1/legislators/", json=sample_legislator)
    print(f"   Status: {response.status_code}")
    if response.status_code != 200:
        print(f"   Expected - database not connected in demo mode")
    
    print("\n✅ Demo completed!")
    print("\n📋 Next Steps:")
    print("   1. Run './setup.sh' to start the full stack")
    print("   2. Import sample data with 'python scripts/import_sample_data.py'")
    print("   3. Access services at their respective URLs")
    print("   4. Explore the Jupyter notebooks in /notebooks")

if __name__ == "__main__":
    demo_api()