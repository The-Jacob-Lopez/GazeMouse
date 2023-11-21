"""This is a script which downloads the Webpage Fixations dataset"""

import gdown
from pathlib import Path
import zipfile
import os
import threading

def download_and_extract(drive_url, dataset_dir):
    print(f"Downloading data: {drive_url}")
    temp_output = str(dataset_dir / f"temp{threading.get_ident()}.zip")
    gdown.download(drive_url, temp_output, quiet=False)

    print(f"Extracting data: {drive_url}")
    with zipfile.ZipFile(temp_output, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    os.remove(temp_output)

webpage_fixations_dir = Path("GazeMouse/data/raw/webpage_fixations")
folder_url = "https://drive.google.com/uc?id=1famhPJDXE14lPD35Oy7C1V2-PWmaMUlA"

# Insure parent directory exists
webpage_fixations_dir.mkdir(parents=True, exist_ok=True)
download_and_extract(folder_url, webpage_fixations_dir)
