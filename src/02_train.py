import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts
from mlflow.models import infer_signature

# ------------------------------------------------------------
# Train Script — Cyberbullying Tweet Classifier
# ------------------------------------------------------------

def train_models(preprocessing_run_id: str):
    """
    Load preprocessed data from MLflow artifacts, train multiple models,
    evaluate each, and register the best one.
    """
    mlflow.set_experiment("Cyberbullying Tweets - Model Training")
    ACCURACY_THRESHOLD = 0.75

    # ------------------------------------------------------------
    # 1️⃣ Load preprocessed data from MLflow artifacts
    # ------------------------------------------------------------
    print(f"🚀 Loading data from preprocessing run: {preprocessing_run_id}")
    local_path = download_artifacts(run_id=preprocessing_run_id, artifact_path="processed_data")

    train_df = pd.read_csv(os.path.join(local_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(local_path, "test.csv"))

    X_train = train_df["clean_tweets"].astype(str)
    y_train = train_df["label"].astype(int)
    X_test = test_df["clean_tweets"].astype(str)
    y_test = test_df["label"].astype(int)

    print(f"✅ Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # ------------------------------------------------------------
    # 2️⃣ Define models to compare
    # ------------------------------------------------------------
    models = {
        "LogisticRegression": LogisticRegression(
            C=5.0, max_iter=2000, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=42), cv=3
        ),
    }

    # TF-IDF vectorizer shared by all models
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
        strip_accents='unicode'
    )

    best_model = None
    best_model_name = ""
    best_accuracy = 0.0
    best_model_info = None

    # ------------------------------------------------------------
    # 3️⃣ Train and evaluate each model
    # ------------------------------------------------------------
    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name}_run"):
            print(f"\n🧠 Training model: {name}")

            pipeline = Pipeline([
                ("tfidf", tfidf),
                ("model", model)
            ])
            pipeline.fit(X_train, y_train)

            # Predictions
            y_pred = pipeline.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision_macro", precision)
            mlflow.log_metric("recall_macro", recall)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_param("model_type", name)

            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1 Score: {f1:.4f}")

            # Save classification report
            report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
            report_path = f"classification_report_{name}.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(report_path)

            # Log model
            signature = infer_signature(X_test.tolist()[:10], y_pred[:10].tolist())
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name="cyberbullying-tweet-classifier",
                signature=signature,
                input_example=X_test.tolist()[:5]
            )

            # Track best model
            if accuracy > best_accuracy:
                best_model = pipeline
                best_model_name = name
                best_accuracy = accuracy
                best_model_info = model_info

    # ------------------------------------------------------------
    # 4️⃣ Register best model (if accuracy >= threshold)
    # ------------------------------------------------------------
    if best_model is not None and best_accuracy >= ACCURACY_THRESHOLD:
        print(f"\n🏆 Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        print(f"📦 Registering model to MLflow Registry...")

        registered_model = mlflow.register_model(
            model_uri=best_model_info.model_uri,
            name="cyberbullying-tweet-classifier"
        )

        print(f"✅ Model registered: {registered_model.name} (version {registered_model.version})")
    else:
        print(f"⚠️ No model met accuracy threshold ({ACCURACY_THRESHOLD:.2f}). Skipping registration.")

# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/train.py <preprocessing_run_id>")
        sys.exit(1)

    run_id = sys.argv[1]
    train_models(run_id)
