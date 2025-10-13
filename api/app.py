"""
Cyberbullying Tweet Classifier API
----------------------------------
✅ Flask API serving an MLflow-registered text classification model.
"""

from flask import Flask, request, jsonify
import mlflow.pyfunc
import os
import pandas as pd
import traceback

# ------------------------------------------------------------
# 🔧 Configuration
# ------------------------------------------------------------
MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/cyberbullying-tweet-classifier@staging"  # default alias
)

print(f"📦 Loading model from: {MODEL_URI}")
model = mlflow.pyfunc.load_model(MODEL_URI)
print("✅ Model loaded successfully.")

# ------------------------------------------------------------
# 🧠 Flask App Setup
# ------------------------------------------------------------
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_uri": MODEL_URI}), 200

# ------------------------------------------------------------
# 🧩 Predict Single Tweet
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict cyberbullying type for a single tweet.
    Expected JSON: {"tweet_text": "some tweet content"}
    """
    try:
        data = request.get_json()

        if not data or "tweet_text" not in data:
            return jsonify({"error": "Missing key 'tweet_text' in request."}), 400

        tweet_text = data["tweet_text"]
        df = pd.DataFrame({"clean_tweets": [tweet_text]})

        prediction = model.predict(df)[0]

        label_map = {
            0: "not_cyberbullying",
            1: "gender",
            2: "religion",
            3: "age",
            4: "ethnicity"
        }

        return jsonify({
            "input_text": tweet_text,
            "predicted_label": int(prediction),
            "label_name": label_map.get(int(prediction), "unknown")
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------
# 📦 Predict Batch of Tweets
# ------------------------------------------------------------
@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction endpoint.
    Expected JSON: {"instances": ["tweet 1", "tweet 2", ...]}
    """
    try:
        data = request.get_json()

        if "instances" not in data:
            return jsonify({"error": "Missing 'instances' key in request."}), 400

        tweets = data["instances"]
        if not isinstance(tweets, list):
            return jsonify({"error": "'instances' must be a list of texts."}), 400

        df = pd.DataFrame({"clean_tweets": tweets})
        preds = model.predict(df)

        label_map = {
            0: "not_cyberbullying",
            1: "gender",
            2: "religion",
            3: "age",
            4: "ethnicity"
        }

        results = [
            {"tweet": t, "predicted_label": int(p), "label_name": label_map.get(int(p), "unknown")}
            for t, p in zip(tweets, preds)
        ]

        return jsonify({"predictions": results}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------
# 🧾 Model Info
# ------------------------------------------------------------
@app.route("/model/info", methods=["GET"])
def model_info():
    """Return metadata for the loaded model."""
    return jsonify({
        "model_name": "Cyberbullying Tweet Classifier",
        "version_alias": MODEL_URI.split("@")[-1],
        "model_uri": MODEL_URI,
        "description": "Classifies tweets into 5 categories of cyberbullying.",
        "labels": {
            "0": "not_cyberbullying",
            "1": "gender",
            "2": "religion",
            "3": "age",
            "4": "ethnicity"
        }
    }), 200

# ------------------------------------------------------------
# 🚀 Run the API
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
