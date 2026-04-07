import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_resources():
    """Download required NLTK resources if they are not already available."""
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

# Retain negation words so the model can distinguish "good" from "not good"
STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "nor", "never"}
LEMMATIZER = WordNetLemmatizer()


def preprocess(text):
    """
    Clean and normalize a review text for ML inference.
    Steps:
        1. Lowercase the input.
        2. Join negation phrases with underscore (e.g. "not good" -> "not_good")
           so the vectorizer treats them as a single meaningful token.
        3. Remove all characters except lowercase letters, underscores, and spaces.
        4. Tokenize, lemmatize, and drop stopwords / single-character tokens.
    Args:
        text: Raw review string (may be NaN).
    Returns:
        Preprocessed string ready for TF-IDF vectorization.
    """
    if pd.isna(text):
        return ""

    text = text.lower()

    # Merge negation + following word into one token (critical for sentiment accuracy)
    text = re.sub(r"\b(not|no|never)\s+([a-z]+)\b", r"\1_\2", text)

    # Strip special characters while preserving underscores from negation handling
    text = re.sub(r"[^a-z_\s]", " ", text)

    tokens = text.split()
    tokens = [
        LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 1
    ]

    return " ".join(tokens)
