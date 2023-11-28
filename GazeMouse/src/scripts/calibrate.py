from src.model.EyeTracker import EyeTracker
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import mouse

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

def collect_samples(num_samples, pressable = False):
    pred_samples = []
    pixel_samples = []
    for i in range(num_samples):
        if pressable:
            out = input('Press Enter')
            if out == 'q':
                break
        print(i)
        pred_samples.append(collect())
        pixel_samples.append(mouse.get_position())
    return np.array([pred_samples, pixel_samples])

num_samples = 200
points = []
input("Scroll your mouse and track it with your eyes")
data = np.array(collect_samples(num_samples, pressable=True))
print(data)
print(data.shape)

with open(str(Path('GazeMouse/data/numpy/calibrate.npy')), 'wb') as f:
    np.save(f, data)