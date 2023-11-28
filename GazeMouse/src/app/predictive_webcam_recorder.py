from multiprocessing import Queue

from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
from src.app.mediapipe_webcam_recorder import detector
import multiprocessing
from src.model.EyeTracker import EyeTracker
from pathlib import Path

itracker_checkpoint = str(Path('GazeMouse/data/uploadable_checkpoints/best_gazecapture_model.pth'))
torch_device = 'cpu'
normalizer_file = str(Path('GazeMouse/data/numpy/normalize_mean.npy'))
detector_checkpoint = str(Path('GazeMouse/data/uploadable_checkpoints/face_landmarker_v2_with_blendshapes.task'))
tracker = EyeTracker(itracker_checkpoint, torch_device, normalizer_file, detector_checkpoint)

class periodic_worker:
    def __init__(self, output_queue:Queue):
        self.output_queue = output_queue
        self.is_processing = False
 
    def process(self):
        pass

    def start_processing(self):
        self.is_processing = True
        while self.is_processing:
            self.output_queue.put(self.process())

    def stop_processing(self):
        self.is_processing = False

class responsive_worker:
    def __init__(self, input_queue:Queue, output_queue:Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
    
    def process(self, input):
        pass

    def start_processing(self):
        while True:
            input = self.input_queue.get()
            self.output_queue.put(self.process(input))
        
class responsive_tracker(responsive_worker):
    def __init__(self, input_queue:Queue, output_queue:Queue):
        super().__init__(input_queue, output_queue)
        
    def process(self, input):
        #Note, this will break when a face is not detected
        return tracker.e2e_gaze_prediction(input)

class responsive_detector(responsive_worker):
    def __init__(self, input_queue:Queue, output_queue:Queue):
        super().__init__(input_queue, output_queue)
        
    def process(self, input):
        captured_image = np.asarray(input)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=captured_image)
        detection_result = detector.detect(mp_image)
        return detection_result

class predictive_webcam_recorder(periodic_worker):
    vid = cv2.VideoCapture(0)
    
    def __init__(self, cap_output_queue:Queue, 
                 tracker_output_queue:Queue, 
                 detector_output_queue:Queue, 
                 cap_width:int = 800, 
                 cap_height:int = 600, 
                 enable_detection = True, 
                 enable_tracking = True,
                 intermediate_queue_max_size = 2):

        self.width, self.height = cap_width, cap_height
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.detector_input_queue = Queue(intermediate_queue_max_size)
        self.tracker_input_queue = Queue(intermediate_queue_max_size)

        self.tracker = responsive_tracker(self.tracker_input_queue, tracker_output_queue)
        self.detector = responsive_detector(self.detector_input_queue, detector_output_queue)

        self.enable_detection = enable_detection
        self.enable_tracking = enable_tracking

        super().__init__(cap_output_queue)

    def process(self):
        _, frame = self.vid.read()
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        captured_image = Image.fromarray(opencv_image)
        #send to other workers
        self._send_to_workers(opencv_image)
        return captured_image
    
    def start_processing(self):
        multiprocessing.Process(target=self.tracker.start_processing).start()
        multiprocessing.Process(target=self.detector.start_processing).start()
        super().start_processing()
    
    def _send_to_workers(self, image):
        if self.enable_tracking:
            self.tracker_input_queue.put(image)
        if self.enable_detection:
            self.detector_input_queue.put(image)