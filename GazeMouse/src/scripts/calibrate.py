from src.model.EyeTracker import EyeTracker
from pathlib import Path
import cv2
from PIL import Image
import numpy as np

vid = cv2.VideoCapture(0) 
itracker_checkpoint = str(Path('GazeMouse/data/uploadable_checkpoints/best_gazecapture_model.pth'))
torch_device = 'cuda:0'
normalizer_file = str(Path('GazeMouse/data/numpy/normalize_mean.npy'))
detector_checkpoint = str(Path('GazeMouse/data/uploadable_checkpoints/face_landmarker_v2_with_blendshapes.task'))
tracker = EyeTracker(itracker_checkpoint, torch_device, normalizer_file, detector_checkpoint)

def collect():
    ret, frame = vid.read() 
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return tracker.e2e_gaze_prediction(opencv_image)[0]

def collect_samples(num_samples):
    samples = []
    for i in range(num_samples):
        samples.append(collect())
    return np.array(samples)

num_samples = 100

points = []
input("Look at the top left and press enter")
points.append(collect_samples(num_samples))

input("Look at the top right and press enter")
points.append(collect_samples(num_samples))

input("Look at the bottom left and press enter")
points.append(collect_samples(num_samples))

input("Look at the bottom right and press enter")
points.append(collect_samples(num_samples))

points = np.array(points)
mean_corners = np.mean(points, axis=1)
print(mean_corners)

with open(str(Path('GazeMouse/data/numpy/calibrate.npy')), 'wb') as f:
    np.save(f, mean_corners)