"""This is a script which downloads a subset of the GazeCapture dataset. 
It contains 48,000 training samples and 5,000 validation samples and is 
approximately 2 Gigabytes in size. More information can be found on 
https://github.com/hugochan/Eye-Tracker."""

from urllib.request import urlretrieve
from pathlib import Path

url = "http://hugochan.net/download/eye_tracker_train_and_val.npz"
gazecapture_dir = Path("GazeMouse/data/raw/gazecapture_subset")
file_location = gazecapture_dir / "eye_tracker_train_and_val.npz"

# insure that dataset directory is created
gazecapture_dir.mkdir(parents=True, exist_ok=True)

path, headers = urlretrieve(url, str(file_location))
for name, value in headers.items():
    print(name, value)
