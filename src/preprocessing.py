import os
import json
import pandas as pd
import nltk
from string import punctuation
from sklearn.model_selection import train_test_split
import mlflow
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.data import find as nltk_find

# -------------------------------------------------------
# NLTK Setup
# -------------------------------------------------------
def _ensure_nltk():
    resources = {
        "corpora/wordnet": "wordnet",
        "corpora/stopwords": "stopwords",
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, package in resources.items():
        try:
            nltk_find(path)
        except LookupError:
            nltk.download(package, quiet=True)

_ensure_nltk()

# -------------------------------------------------------
# Global text processing tools
# -------------------------------------------------------
lemmatizer = WordNetLemmatizer()
stopwords_list = stopwords.words("english")
stopwords_list.extend(["rt", "mkr", "im", "u", "ur", "lol", "btw", "amp", "n"])
STOPWORDS = set(stopwords_list)

# -------------------------------------------------------
# Text cleaning helpers
# -------------------------------------------------------
def clean_text(text):
    """Clean tweet text: remove usernames, hashtags, links, punctuation, lowercase, tokenize, remove stopwords."""
    if text is None:
        return ''
        
    text = str(text)
    if not text.strip():
        return ''
        
    # Remove hashtags but keep the text after #
    words = text.split()
    words = [w.lstrip('#@') for w in words if not w.lower().startswith('http')]
    text = " ".join(words)
    
    # Remove punctuation
    for mark in punctuation:
        text = text.replace(mark, " ")
    
    # Tokenize and clean
    text = text.lower()
    tokens = word_tokenize(text)
    
    # Only lemmatize single words, keep phrases intact
    cleaned_tokens = []
    for token in tokens:
        if token not in STOPWORDS:
            if token.isalpha():
                lemma = lemmatizer.lemmatize(token)
                if (
                    token.endswith("s")
                    and not token.endswith("ss")
                    and len(token) > 3
                    and lemma == token[:-1]
                ):
                    cleaned_tokens.append(token)
                else:
                    cleaned_tokens.append(lemma)
            elif any(c.isalpha() for c in token):
                cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)

# -------------------------------------------------------
# Label encoding
# -------------------------------------------------------
ENCODE_MAP = {
    "not_cyberbullying": 0,
    "gender": 1,
    "religion": 2,
    "age": 3,
    "ethnicity": 4,
}

# -------------------------------------------------------
# Main preprocessing pipeline
# -------------------------------------------------------
def preprocess_cyberbullying_data(
    data_path=None,
    test_size=0.2,
    random_state=42
):
    mlflow.set_experiment("Cyberbullying Tweets - Data Preprocessing")

    with mlflow.start_run(run_name="data_preprocessing") as run:
        run_id = run.info.run_id
        print(f"🚀 Starting preprocessing run: {run_id}")

        # Resolve dataset path: use DATA_PATH env var if provided, otherwise
        # prefer a repository-relative file `data/cyberbullying_tweets.csv`.
        if data_path is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            data_path = os.getenv("DATA_PATH", os.path.join(repo_root, "data", "cyberbullying_tweets.csv"))

        # 1️⃣ Load dataset
        if not os.path.exists(data_path):
            cwd = os.getcwd()
            raise FileNotFoundError(
                f"❌ Dataset not found at {data_path}\n" \
                f"Current working directory: {cwd}\n" \
                "Hint: set DATA_PATH env var or place the CSV at data/cyberbullying_tweets.csv"
            )

        df = pd.read_csv(data_path)
        print(f"✅ Loaded dataset: {df.shape}")

        # 2️⃣ Clean tweets
        df["tweet_text"] = df["tweet_text"].astype(str)
        df["clean_tweets"] = df["tweet_text"].apply(clean_text)
        # Ensure clean_tweets are strings and drop rows that become empty after cleaning
        df["clean_tweets"] = df["clean_tweets"].astype(str)
        df = df[df["clean_tweets"].str.strip() != ""].copy()

        # Drop rows that are numeric-only or contain no alphabetic characters
        # (helps avoid cases where non-text values sneak into the dataset)
        df = df[df["clean_tweets"].str.contains(r"[A-Za-z]", regex=True)]

        df.drop_duplicates("clean_tweets", inplace=True)

        # 3️⃣ Encode labels
        df = df[df["cyberbullying_type"].isin(ENCODE_MAP.keys())].copy()
        df["label"] = df["cyberbullying_type"].map(ENCODE_MAP)

        # 4️⃣ Split data
        X = df["clean_tweets"]
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 5️⃣ Save processed data
        os.makedirs("processed_data", exist_ok=True)
        train_df = pd.DataFrame({"clean_tweets": X_train, "label": y_train})
        test_df = pd.DataFrame({"clean_tweets": X_test, "label": y_test})

        train_df.to_csv("processed_data/train.csv", index=False)
        test_df.to_csv("processed_data/test.csv", index=False)

        # 6️⃣ Log MLflow metrics + artifacts
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("encode_map", json.dumps(ENCODE_MAP))
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.log_artifacts("processed_data", artifact_path="processed_data")

        print("✅ Data preprocessing completed successfully.")
        print(f"📦 Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"📂 Artifacts logged under run_id: {run_id}")

        return run_id

# -------------------------------------------------------
# Run standalone
# -------------------------------------------------------
if __name__ == "__main__":
    run_id = preprocess_cyberbullying_data()
    print(f"🧩 Preprocessing Run ID: {run_id}")
