import pandas as pd
from pathlib import Path
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import re
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def make_label(rating):
    if rating >= 7:
        return "Positive"
    elif 4 <= rating <= 6:
        return "Neutral"
    else:
        return "Negative"


def preprocess(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
)

df_train = pd.read_csv(Path(BASE_DIR) / "./data/cleaned/drugsComTrain_cleaned.csv")
df_train["sentiment"] = df_train["rating"].map(make_label)
df_train["review_clean"] = df_train["review"].map(preprocess)

df_test = pd.read_csv(Path(BASE_DIR) / "./data/cleaned/drugsComTest_cleaned.csv")
df_test["sentiment"] = df_test["rating"].map(make_label)
df_test["review_clean"] = df_test["review"].map(preprocess)

X_train = df_train["review_clean"]
Y_train = df_train["sentiment"]
X_test  = df_test["review_clean"]
Y_test  = df_test["sentiment"]

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

param_grid = {
    "C": [0.1, 1.0, 10.0],
    "solver": ["saga"],
    "penalty": ["l1", "l2"],  # fix: tách thành 2 string riêng biệt
}

base_clf = LogisticRegression(
    multi_class="multinomial",
    max_iter=3000,
    n_jobs=-1,
    class_weight="balanced",
)

gs = GridSearchCV(
    base_clf,
    param_grid,
    cv=StratifiedKFold(n_splits=3),
    scoring="f1_macro",  # fix: macro không phải marco
    n_jobs=-1,
    verbose=1,
)

gs.fit(X_train_vec, Y_train)
print("Best params:", gs.best_params_)
print("Best CV macro F1:", gs.best_score_)

clf = gs.best_estimator_  # đã được fit sẵn, không cần fit lại

Y_pred = clf.predict(X_test_vec)

print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
print(classification_report(Y_test, Y_pred))

model_dir = Path(BASE_DIR) / "models"
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.pkl")
joblib.dump(clf, model_dir / "lr_sentiment_model.pkl")
print("Done")