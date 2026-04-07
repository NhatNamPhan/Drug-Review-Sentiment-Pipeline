import html
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def clean_text(text):
    if pd.isna(text):
        return text
    text = html.unescape(str(text))
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    if "users found" in text:
        text = "unknown"
    return text


def clean_df(input_path, output_path):
    df = pd.read_csv(input_path)
    df["review"] = df["review"].map(clean_text)
    df["condition"] = df["condition"].map(clean_text)
    df["condition"] = df["condition"].fillna("unknown")
    df = df.drop_duplicates()
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%y")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


clean_df(
    Path(BASE_DIR) / "./data/raw/drugsComTrain_raw.csv",
    Path(BASE_DIR) / "./data/cleaned/drugsComTrain_cleaned.csv",
)

clean_df(
    Path(BASE_DIR) / "./data/raw/drugsComTest_raw.csv",
    Path(BASE_DIR) / "./data/cleaned/drugsComTest_cleaned.csv",
)
