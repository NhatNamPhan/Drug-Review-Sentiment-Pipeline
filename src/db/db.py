import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

host = os.getenv("POSTGRES_HOST")
port = os.getenv("POSTGRES_PORT")
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
db = os.getenv("POSTGRES_DB")

DATABASE_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

engine = create_engine(DATABASE_URL)


def create_table():
    with engine.connect() as conn:
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            review TEXT,
            sentiment VARCHAR(10),
            prob_positive FLOAT,
            prob_neutral FLOAT,
            prob_negative FLOAT,
            predicted_at TIMESTAMP DEFAULT NOW()
        );
        """)
        )
        conn.commit()
    print("Table 'predictions' is ready.")


def insert_predictions(df):
    df.to_sql(name="predictions", con=engine, if_exists="append", index=False)
    print(f"Inserted {len(df)} rows into 'predictions'.")
