import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
EXPERIMENT_NAME = "Cyberbullying Tweets - Model Training"
MODEL_NAME = "cyberbullying-tweet-classifier"
RUN_NAME = "rf_tfidf_training"
TARGET_COLUMN = "label"
TEXT_COLUMN = "clean_tweets"
F1_THRESHOLD = 0.85

# Set MLflow tracking URI (local)aaaa
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)


def train_and_register(preprocessing_run_id: str):
    """
    Load processed data from MLflow artifacts using run_id,
    train RandomForest + TF-IDF model,
    evaluate, log metrics, and register model if performance is good.
    """
    print(f"📥 Loading preprocessed data from MLflow Run ID: {preprocessing_run_id}")
    local_path = download_artifacts(run_id=preprocessing_run_id, artifact_path="processed_data")

    train_df = pd.read_csv(os.path.join(local_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(local_path, "test.csv"))

    X_train = train_df[TEXT_COLUMN].astype(str)
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_test = test_df[TEXT_COLUMN].astype(str)
    y_test = test_df[TARGET_COLUMN].astype(int)

    print(f"✅ Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # ------------------------------------------------------------
    # Create Pipeline
    # ------------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            max_df=0.95,
            strip_accents='unicode'
        )),
        ("model", model)
    ])

    # ------------------------------------------------------------
    # MLflow run
    # ------------------------------------------------------------
    with mlflow.start_run(run_name=RUN_NAME) as run:
        print(f"\n🚀 Training model: RandomForest")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        print("\n📊 Performance Metrics:")
        print(f"   Accuracy : {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall   : {recall:.4f}")
        print(f"   F1 Score : {f1:.4f}")

        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1
        })

        # Log parameters
        mlflow.log_params({
            "model_type": "RandomForest",
            "n_estimators": 300,
            "tfidf_max_features": 20000,
            "f1_threshold": F1_THRESHOLD
        })

        # Save and log model
        signature = infer_signature(X_test.tolist()[:10], y_pred[:10])
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=X_test.tolist()[:3]
        )

        # ------------------------------------------------------------
        # Register best model and set alias
        # ------------------------------------------------------------
        if f1 >= F1_THRESHOLD:
            print(f"\n🎯 F1 Score {f1:.4f} >= {F1_THRESHOLD} ✅ Eligible for registration")

            client = MlflowClient()

            # Register model version
            model_uri = model_info.model_uri
            result = mlflow.register_model(
                model_uri=model_uri,
                name=MODEL_NAME
            )

            version = result.version
            print(f"📦 Model registered as version {version}")

            # Set alias staging
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias="staging",
                version=version
            )
            print(f"✅ Alias '@staging' now points to version {version}")
        else:
            print(f"⚠️ F1 Score {f1:.4f} < {F1_THRESHOLD} ❌ Model not registered")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/02_train.py <preprocessing_run_id>")
        sys.exit(1)

    preprocessing_run_id = sys.argv[1]
    train_and_register(preprocessing_run_id)


