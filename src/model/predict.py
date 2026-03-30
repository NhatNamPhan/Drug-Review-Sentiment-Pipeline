import pandas as pd
import joblib
import re
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def preprocess(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens 
              if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

def predict_sentiment(text):
    clean_text = preprocess(text)
    
    vec_text = vectorizer.transform([clean_text])
    
    prediction = clf.predict(vec_text)[0]
    
    probs = clf.predict_proba(vec_text)[0]
    classes = clf.classes_
    prob_dict = dict(zip(classes, probs))
    
    return {
        "sentiment": prediction,
        "probabilities": prob_dict
    }


model_dir = Path(BASE_DIR) / "models"
vectorizer = joblib.load(model_dir / "tfidf_vectorizer.pkl")
clf = joblib.load(model_dir / "lr_sentiment_model.pkl")


if __name__ == "__main__":
    print("\n--- TEST MACHINE LEARNING ---")
    print("Gõ 'exit' hoặc 'quit' để thoát.")
    
    while True:
        user_input = input("\nNhập review của bạn (Tiếng Anh): ")
        
        # Nếu muốn thoát chương trình
        if user_input.strip().lower() in ["exit", "quit", ""]:
            break
            
        # Đưa vào model đọc bệnh
        result = predict_sentiment(user_input)
        
        # In ra kết quả
        print(f"👉 Sentiment : {result['sentiment']}")
        print(f"   (Positive:  {result['probabilities']['Positive']:.2%})")
        print(f"   (Neutral :  {result['probabilities']['Neutral']:.2%})")
        print(f"   (Negative:  {result['probabilities']['Negative']:.2%})")

