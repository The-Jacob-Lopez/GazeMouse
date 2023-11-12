from multiprocessing import Queue

from PIL import Image
import cv2

class webcam_recorder:
    vid = cv2.VideoCapture(0)
    
    def __init__(self, queue:Queue, width:int = 800, height:int = 600):
        self.queue = queue
        self.width, self.height = width, height
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.is_recording = False

    def capture(self):
        _, frame = self.vid.read()
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        return captured_image

    def start_recording(self):
        self.is_recording = True
        while self.is_recording:
            self.queue.put(self.capture()) 
            
    def stop_recording(self):
        self.is_recording = False
            