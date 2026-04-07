from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from text_utils import preprocess

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ========================
# Labeling
# ========================
def make_label(rating):
    """Convert a numeric drug rating (1-10) into a sentiment label."""
    if rating >= 7:
        return "Positive"
    elif 4 <= rating <= 6:
        return "Neutral"
    else:
        return "Negative"


# ========================
# Load data
# ========================
df_train = pd.read_csv(BASE_DIR / "data/cleaned/drugsComTrain_cleaned.csv")
df_test = pd.read_csv(BASE_DIR / "data/cleaned/drugsComTest_cleaned.csv")

df_train["sentiment"] = df_train["rating"].map(make_label)
df_test["sentiment"] = df_test["rating"].map(make_label)

df_train["review_clean"] = df_train["review"].map(preprocess)
df_test["review_clean"] = df_test["review"].map(preprocess)

X_train = df_train["review_clean"]
Y_train = df_train["sentiment"]
X_test = df_test["review_clean"]
Y_test = df_test["sentiment"]

# ========================
# Pipeline
# ========================
pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=60000,
                ngram_range=(1, 3),  # include unigrams, bigrams, and trigrams
                sublinear_tf=True,  # apply log normalization to term frequencies
                min_df=5,  # ignore terms that appear in fewer than 5 documents
            ),
        ),
        (
            "clf",
            LinearSVC(class_weight="balanced", dual="auto"),
        ),  # handle class imbalance automatically
    ]
)

# ========================
# Hyperparameter tuning
# ========================
param_grid = {"clf__C": [0.5, 1.0, 2.0]}

gs = GridSearchCV(
    pipeline,
    param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
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
