"""
Model Evaluation for Cyberbullying Tweet Classifier
Evaluates a registered MLflow model (e.g. @staging) with full metrics and visualizations
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import mlflow
import mlflow.pyfunc
from mlflow.artifacts import download_artifacts

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (9, 6)

class CyberbullyingEvaluator:
    def __init__(self, model_uri="models:/cyberbullying-tweet-classifier@staging", run_id=None):
        """
        Args:
            model_uri: MLflow model URI (can include alias like @staging)
            run_id: Optional, if evaluating from preprocessing artifacts
        """
        self.model_uri = model_uri
        self.model = None
        self.run_id = run_id
        self.results = {}

    def load_model(self):
        """Load model from MLflow"""
        print(f"📦 Loading model from {self.model_uri} ...")
        self.model = mlflow.pyfunc.load_model(self.model_uri)
        print("✅ Model loaded successfully!")

    def load_data(self):
        """Load processed test data"""
        if self.run_id:
            print(f"📂 Downloading test data from preprocessing run: {self.run_id}")
            local_path = download_artifacts(run_id=self.run_id, artifact_path="processed_data")
            test_path = os.path.join(local_path, "test.csv")
        else:
            test_path = "processed_data/test.csv"

        if not os.path.exists(test_path):
            raise FileNotFoundError(f"❌ Cannot find test.csv at {test_path}")

        df = pd.read_csv(test_path)
        print(f"✅ Loaded test data: {df.shape[0]} samples")
        return df["clean_tweets"].astype(str), df["label"].astype(int)

    def evaluate(self, X_test, y_test):
        """Compute core metrics"""
        print("🔎 Making predictions...")
        # Defensive: ensure all inputs are strings. Some datasets may contain
        # numeric values or unexpected types which break text preprocessing
        # (e.g. calling .lower()). Coerce every input to str to avoid that.
        X_list = [str(x) for x in X_test.tolist()]
        # Convert to numpy array with object dtype to ensure sklearn's
        # text transformers receive proper Python strings (not e.g. ints).
        X_arr = np.array(X_list, dtype=object)
        # Sanitize inputs: replace items that contain no alphabetic characters
        # with an empty string to avoid .lower() on non-text types inside
        # sklearn's text preprocessing.
        non_alpha_idx = [i for i, v in enumerate(X_arr) if not any(c.isalpha() for c in str(v))]
        if non_alpha_idx:
            print(f"⚠️ Replacing {len(non_alpha_idx)} non-alphabetic inputs with empty strings (sample indexes: {non_alpha_idx[:10]})")
            for i in non_alpha_idx:
                X_arr[i] = ""
        # Try a sequence of input formats to accommodate model input expectations
        last_exc = None
        try:
            y_pred = self.model.predict(X_arr)
        except Exception as e1:
            print("Error while predicting with numpy array (object dtype):", str(e1))
            last_exc = e1

            # Diagnostic: show types/representations for first 20 items
            sample = list(enumerate(X_arr[:20]))
            print("Input sample types/values (first 20):")
            for i, v in sample:
                print(f"  [{i}] type={type(v).__name__} repr={repr(v)[:200]}")

            # Try pandas Series
            try:
                X_ser = pd.Series(X_list)
                y_pred = self.model.predict(X_ser)
                print("Prediction succeeded with pandas.Series input.")
            except Exception as e2:
                print("Prediction with pandas.Series failed:", str(e2))
                last_exc = e2

                # Try pandas DataFrame with a common column name used in the pipeline
                try:
                    X_df = pd.DataFrame({"clean_tweets": X_list})
                    y_pred = self.model.predict(X_df)
                    print("Prediction succeeded with pandas.DataFrame input (clean_tweets column).")
                except Exception as e3:
                    print("Prediction with pandas.DataFrame failed:", str(e3))
                    last_exc = e3
                    # Give up and re-raise the last exception with context
                    raise last_exc
        # If prediction length doesn't match inputs, attempt per-sample predict
        try:
            pred_len = len(y_pred)
        except TypeError:
            pred_len = 1

        if pred_len != len(X_test):
            print(f"⚠️ Prediction length mismatch: got {pred_len} predictions for {len(X_test)} samples. Falling back to per-sample prediction.")
            per_preds = []
            for i, s in enumerate(X_list):
                try:
                    single = self.model.predict([s])
                    # single may be array-like or scalar
                    if hasattr(single, "__len__"):
                        per_preds.append(single[0])
                    else:
                        per_preds.append(single)
                except Exception as e:
                    print(f"Error predicting sample {i}: {e} -- inserting fallback label 0")
                    per_preds.append(0)

            y_pred = np.array(per_preds, dtype=int)
        else:
            y_pred = np.array(y_pred, dtype=int)

        print("📊 Calculating metrics...")
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }

        print("\n=== Model Performance ===")
        for k, v in metrics.items():
            print(f"{k:20s}: {v:.4f}")

        self.results["metrics"] = metrics
        self.results["y_pred"] = y_pred
        return metrics, y_pred

    def plot_confusion_matrix(self, y_true, y_pred, save_path="models/evaluation/confusion_matrix.png"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix", fontsize=14)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📈 Confusion matrix saved to {save_path}")
        return cm

    def generate_classification_report(self, y_true, y_pred, save_path="models/evaluation/classification_report.txt"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        report = classification_report(y_true, y_pred, digits=4)
        with open(save_path, "w") as f:
            f.write("CYBERBULLYING CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        print("📄 Classification report saved.")
        return report

    def log_to_mlflow(self, metrics, artifacts_dir="models/evaluation"):
        print("📤 Logging to MLflow...")
        mlflow.set_experiment("Cyberbullying Tweets - Model Evaluation")
        with mlflow.start_run(run_name="evaluation_run"):
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(artifacts_dir)
            mlflow.log_param("model_uri", self.model_uri)
            print(f"✅ Logged to MLflow (Run ID: {mlflow.active_run().info.run_id})")

    def full_evaluation(self):
        """Run full evaluation pipeline"""
        self.load_model()
        X_test, y_test = self.load_data()
        metrics, y_pred = self.evaluate(X_test, y_test)

        os.makedirs("models/evaluation", exist_ok=True)
        cm = self.plot_confusion_matrix(y_test, y_pred)
        self.generate_classification_report(y_test, y_pred)

        summary = {"metrics": metrics, "confusion_matrix": cm.tolist()}
        with open("models/evaluation/summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("🧾 Summary saved to models/evaluation/summary.json")

        self.log_to_mlflow(metrics)
        print("\n✅ Evaluation Completed Successfully!")


# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Cyberbullying Tweet Classifier")
    parser.add_argument("--model_uri", type=str,
                        default="models:/cyberbullying-tweet-classifier@staging",
                        help="MLflow model URI or local path")
    parser.add_argument("--run_id", type=str, default=None, help="Optional preprocessing run ID")
    args = parser.parse_args()

    evaluator = CyberbullyingEvaluator(model_uri=args.model_uri, run_id=args.run_id)
    evaluator.full_evaluation()
