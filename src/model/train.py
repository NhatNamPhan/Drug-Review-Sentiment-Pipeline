import pandas as pd
from pathlib import Path
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# ========================
# Setup
# ========================
def ensure_nltk_resources():
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }
    for resource_path, resource_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)

ensure_nltk_resources()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 👉 GIỮ lại từ phủ định
STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "nor", "never"}
LEMMATIZER = WordNetLemmatizer()

# ========================
# Labeling
# ========================
def make_label(rating):
    if rating >= 7:
        return "Positive"
    elif 4 <= rating <= 6:
        return "Neutral"
    else:
        return "Negative"

# ========================
# Preprocess (IMPROVED)
# ========================
def preprocess(text):
    if pd.isna(text):
        return ""

    text = text.lower()

    # 👉 xử lý negation (rất quan trọng)
    text = re.sub(r"\b(not|no|never)\s+([a-z]+)\b", r"\1_\2", text)

    # remove ký tự đặc biệt
    text = re.sub(r"[^a-z_\s]", " ", text)

    tokens = text.split()

    tokens = [
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 1
    ]

    return " ".join(tokens)

# ========================
# Load data
# ========================
df_train = pd.read_csv(BASE_DIR / "data/cleaned/drugsComTrain_cleaned.csv")
df_test  = pd.read_csv(BASE_DIR / "data/cleaned/drugsComTest_cleaned.csv")

df_train["sentiment"] = df_train["rating"].map(make_label)
df_test["sentiment"]  = df_test["rating"].map(make_label)

df_train["review_clean"] = df_train["review"].map(preprocess)
df_test["review_clean"]  = df_test["review"].map(preprocess)

X_train = df_train["review_clean"]
Y_train = df_train["sentiment"]
X_test  = df_test["review_clean"]
Y_test  = df_test["sentiment"]

# ========================
# Pipeline (BEST PRACTICE)
# ========================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=60000,
        ngram_range=(1, 3),   # 🔥 upgrade từ (1,2)
        sublinear_tf=True,
        min_df=5
    )),
    ("clf", LinearSVC(class_weight="balanced", dual="auto"))
])

# ========================
# Hyperparameter tuning
# ========================
param_grid = {
    "clf__C": [0.5, 1.0, 2.0]
}

gs = GridSearchCV(
    pipeline,
    param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)

# ========================
# Train
# ========================
gs.fit(X_train, Y_train)

print("Best params:", gs.best_params_)
print("Best CV macro F1:", gs.best_score_)

# ========================
# Evaluate
# ========================
best_model = gs.best_estimator_

Y_pred = best_model.predict(X_test)

print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
print(classification_report(Y_test, Y_pred))

# ========================
# Save model
# ========================
model_dir = BASE_DIR / "models"
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, model_dir / "sentiment_pipeline.pkl")

print("✅ Model saved!")