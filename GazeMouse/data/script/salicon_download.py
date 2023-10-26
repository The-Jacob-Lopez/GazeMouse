"""This is a script which downloads the SALICON dataset"""

import gdown
from pathlib import Path
import zipfile
import os
import threading
from threading import Thread

def download_and_extract(drive_url, dataset_dir):
    print(f"Downloading data: {drive_url}")
    temp_output = str(dataset_dir / f"temp{threading.get_ident()}.zip")
    gdown.download(drive_url, temp_output, quiet=False)

    print(f"Extracting data: {drive_url}")
    with zipfile.ZipFile(temp_output, "r") as zip_ref:
        zip_ref.extractall(dir)

    os.remove(temp_output)

salicon_dir = Path("GazeMouse/data/raw/salicon")
images_url = "https://drive.google.com/uc?id=1g8j-hTT-51IG1UFwP0xTGhLdgIUCW5e5"
fixations_url = "https://drive.google.com/uc?id=1P-jeZXCsjoKO79OhFUgnj6FGcyvmLDPj"
fixation_maps_url = "https://drive.google.com/uc?id=1PnO7szbdub1559LfjYHMy65EDC4VhJC8"

# Insure parent directory exists
salicon_dir.mkdir(parents=True, exist_ok=True)

# Download data in parallel
for url in [images_url, fixations_url, fixation_maps_url]:
    Thread(target = download_and_extract, args = (url, salicon_dir)).start()

