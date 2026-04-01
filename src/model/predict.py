import argparse
import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_pipeline.pkl"
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "reviews_to_predict.txt"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "reviews_predicted.csv"


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
STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "nor", "never"}
LEMMATIZER = WordNetLemmatizer()


def preprocess(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"\b(not|no|never)\s+([a-z]+)\b", r"\1_\2", text)
    text = re.sub(r"[^a-z_\s]", " ", text)
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def softmax(scores):
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def predict_with_confidence(model, texts):
    clean_texts = [preprocess(t) for t in texts]
    preds = model.predict(clean_texts)
    classes = list(model.classes_)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(clean_texts)
    else:
        decision_scores = model.decision_function(clean_texts)
        if decision_scores.ndim == 1:
            decision_scores = np.vstack([-decision_scores, decision_scores]).T
        probs = softmax(decision_scores)

    return preds, classes, probs


def run_batch(model, input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        reviews = [line.strip() for line in f if line.strip()]

    preds, classes, probs = predict_with_confidence(model, reviews)
    result_df = pd.DataFrame({"review": reviews, "sentiment": preds})

    for idx, class_name in enumerate(classes):
        result_df[f"prob_{class_name.lower()}"] = probs[:, idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved predictions to: {output_path}")
    print(f"Total reviews predicted: {len(result_df)}")


def run_interactive(model):
    print("\n--- SENTIMENT PREDICTION ---")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("\nNhập review của bạn (Tiếng Anh): ")
        if user_input.strip().lower() in ["exit", "quit", ""]:
            break

        preds, classes, probs = predict_with_confidence(model, [user_input])
        prob_map = dict(zip(classes, probs[0]))

        print(f"👉 Sentiment : {preds[0]}")
        for class_name in ["Positive", "Neutral", "Negative"]:
            if class_name in prob_map:
                print(f"   ({class_name:<8}: {prob_map[class_name]:.2%})")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict sentiment from reviews.")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch prediction from a text file and save to CSV.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input text file path (one review per line).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path for batch prediction.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    sentiment_model = joblib.load(MODEL_PATH)
    args = parse_args()

    if args.batch:
        run_batch(sentiment_model, args.input, args.output)
    else:
        run_interactive(sentiment_model)
