import os
from pathlib import Path
import kaggle
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv()

kaggle.api.authenticate()

out_dir = BASE_DIR / "data" / "raw"
out_dir.mkdir(parents=True, exist_ok=True)

kaggle.api.dataset_download_files(
    dataset="jessicali9530/kuc-hackathon-winter-2018",
    path=str(out_dir),
    unzip=True
)