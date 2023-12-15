from src.model.EyeTracker import EyeTracker
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import mouse
import time

vid = cv2.VideoCapture(0) 

def collect():
    ret, frame = vid.read() 
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return opencv_image

def warmup():
    collect()
    mouse.get_position()

def collect_samples(num_samples, num_rounds, delay):
    pixel_samples = []
    image_samples = []
    for j in range(num_rounds):
        input(f"Start round {j+1}")
        for i in range(num_samples):
            image_samples.append(collect())
            pixel_samples.append(mouse.get_position())
            time.sleep(delay)
            print(i)
    return pixel_samples, image_samples

num_samples = 500
num_rounds = 3
delay = 0.02

warmup()
name = input('Enter participant\'s name: ')
print('Scroll your mouse and track it with your eyes')
pixel_samples, image_samples = collect_samples(num_samples, num_rounds, delay)
pixel_samples, image_samples = np.array(pixel_samples), np.array(image_samples)
print(f'Pixel shape:{pixel_samples.shape}')
print(f'Image shape:{image_samples.shape}')

with open(str(Path(f'GazeMouse/data/finetuning/calibrate_{name}.npy')), 'wb') as f:
    np.save(f, pixel_samples)
    np.save(f, image_samples)