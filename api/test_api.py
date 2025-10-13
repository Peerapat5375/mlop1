"""
API Test Suite for Cyberbullying Tweet Classifier
-------------------------------------------------
ใช้ทดสอบ Flask API ที่รันจาก app.py
"""

import requests
import json

API_URL = "http://localhost:5002"


# ------------------------------------------------------------
# 🩺 Health Check
# ------------------------------------------------------------
def test_health():
    """Test health endpoint"""
    print("🔍 Testing /health endpoint ...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


# ------------------------------------------------------------
# 🧠 Single Prediction
# ------------------------------------------------------------
def test_single_prediction():
    """Test /predict endpoint"""
    print("🧠 Testing single tweet prediction ...")

    tweet = {
        "tweet_text": "You are such a loser!"
    }

    response = requests.post(
        f"{API_URL}/predict",
        json=tweet,
        headers={"Content-Type": "application/json"}
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


# ------------------------------------------------------------
# 📦 Batch Prediction
# ------------------------------------------------------------
def test_batch_prediction():
    """Test /predict/batch endpoint"""
    print("📦 Testing batch tweet prediction ...")

    tweets = {
        "instances": [
            "I hate you because of your religion",
            "Old people should not use the internet",
            "You're a strong and kind person",
            "Stop being such a baby",
            "I love helping others online"
        ]
    }

    response = requests.post(
        f"{API_URL}/predict/batch",
        json=tweets,
        headers={"Content-Type": "application/json"}
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


# ------------------------------------------------------------
# 🧾 Model Info
# ------------------------------------------------------------
def test_model_info():
    """Test /model/info endpoint"""
    print("ℹ️ Testing /model/info endpoint ...")
    response = requests.get(f"{API_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


# ------------------------------------------------------------
# 🧩 Realistic Example Set
# ------------------------------------------------------------
def test_realistic_examples():
    """Run multiple real example predictions"""
    print("🧩 Testing multiple real-world examples ...")

    examples = [
        "I hate you because of your religion",
        "You are such an idiot!",
        "Women are weak and can't lead",
        "My grandma is so bad with technology",
        "I love my friends from all countries",
        "Old people should stop using phones",
        "You are amazing!",
    ]

    for i, tweet in enumerate(examples, 1):
        response = requests.post(f"{API_URL}/predict", json={"tweet_text": tweet})
        print(f"Example {i}: {tweet}")
        print(f"Response: {json.dumps(response.json(), indent=2)}\n")


# ------------------------------------------------------------
# 🚀 Main Runner
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 CYBERBULLYING CLASSIFIER API TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_health()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        test_realistic_examples()

        print("=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the Flask server is running!")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
