from multiprocessing import Queue

from PIL import Image
import mss
import cv2
import numpy as np

def _resize_image_to_window(image, width:int, height:int):
    """
    Scales an image to fit within the height and width 
    """
    scaling_factor = min(width / image.width, height / image.height)
    new_width = int(image.width * scaling_factor)
    new_height = int(image.height * scaling_factor)
    return image.resize((new_width, new_height), Image.NEAREST)

class screen_recorder:
    sct = mss.mss()

    def __init__(self, queue:Queue, width:int = 800, height:int = 600):
        self.queue = queue
        self.monitor = self.sct.monitors[1]
        self.width = width
        self.height = height
        self.is_recording = False

    def capture(self):
        image_bytes = np.asarray(self.sct.grab(self.monitor))
        image_bytes = cv2.cvtColor(image_bytes, cv2.COLOR_BGR2RGB)

        captured_image = Image.fromarray(image_bytes)
        captured_image = _resize_image_to_window(captured_image, self.width, self.height)
        return captured_image
    
    def start_recording(self):
        self.is_recording = True
        while self.is_recording:
            self.queue.put(self.capture())
            
    def stop_recording(self):
        self.is_recording = False
            