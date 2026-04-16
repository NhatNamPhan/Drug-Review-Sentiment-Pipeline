from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {"owner": "airflow", "retries": 2, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="drug_review_pipeline",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    def task_ingest():
        import os
        from pathlib import Path
        from dotenv import load_dotenv
        import kaggle

        BASE_DIR = Path("/opt/airflow")

        load_dotenv()

        kaggle.api.authenticate()
        out_dir = BASE_DIR / "data" / "raw"
        out_dir.mkdir(parents=True, exist_ok=True)

        kaggle.api.dataset_download_files(
            dataset="jessicali9530/kuc-hackathon-winter-2018",
            path=str(out_dir),
            unzip=True,
        )

    def task_clean():
        import sys
        from pathlib import Path

        BASE_DIR = Path("/opt/airflow")
        sys.path.insert(0, str(BASE_DIR))

        from src.cleaning.cleaning import clean_df

        clean_df(
            BASE_DIR / "data/raw/drugsComTrain_raw.csv",
            BASE_DIR / "data/cleaned/drugsComTrain_cleaned.csv",
        )

        clean_df(
            BASE_DIR / "data/raw/drugsComTest_raw.csv",
            BASE_DIR / "data/cleaned/drugsComTest_cleaned.csv",
        )

    def task_store_db():
        import sys
        from pathlib import Path

        BASE_DIR = Path("/opt/airflow")
        sys.path.insert(0, str(BASE_DIR))

        from src.db.db import create_table, insert_predictions
        import pandas as pd

        create_table()
        df = pd.read_csv(BASE_DIR / "data/cleaned/drugsComTrain_cleaned.csv")
        insert_predictions(df.head(1000))
        
    def task_predict():
        import sys
        import pandas as pd
        import joblib 
        from pathlib import Path
        
        BASE_DIR = Path("/opt/airflow")
        sys.path.insert(0, str(BASE_DIR))

        from src.db.db import create_table, insert_predictions
        
        MODEL_PATH = BASE_DIR / "models/sentiment_pipeline.pkl"
        TEST_DATA_PATH = BASE_DIR / "data/cleaned/drugsComTest_cleaned.csv"

        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(TEST_DATA_PATH)
        
        reviews = df['review'].dropna().tolist()
        
        reviews = reviews[:500]
        
        preds = model.predict(reviews)
        
        results_df = pd.DataFrame({
            "review": reviews,
            "sentiment": preds,
            "prob_positive": None,
            "prob_neutral": None,
            "prob_negative": None
        })
        
        create_table()
        insert_predictions(results_df)

    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=task_ingest
    )
    
    clean = PythonOperator(
        task_id="clean_data",
        python_callable=task_clean
    )
    
    store = PythonOperator(
        task_id="store_to_db",
        python_callable=task_store_db
    )        
        
    predict = PythonOperator(
        task_id="run_prediction",
        python_callable=task_predict
    )
    
    ingest >> clean >> store >> predict