from urllib.request import urlretrieve
from pathlib import Path

url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
checkpoint_dir = Path("GazeMouse/data/pytorch_checkpoints")
file_location = checkpoint_dir / "face_landmarker_v2_with_blendshapes.task"

path, headers = urlretrieve(url, str(file_location))
for name, value in headers.items():
    print(name, value)