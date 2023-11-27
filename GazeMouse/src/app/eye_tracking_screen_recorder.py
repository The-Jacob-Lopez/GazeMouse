from src.app.screen_recorder import screen_recorder
from src.app.screen_recorder import _resize_image_to_window
import numpy as np
import cv2
from PIL import Image
from src.model.EyeTracker import EyeTracker
import mediapipe as mp

class eye_tracking_screen_recorder(screen_recorder):
    itracker_checkpoint = 'GazeMouse\\data\\uploadable_checkpoints\\best_gazecapture_model.pth'
    torch_device = 'cuda:0'
    normalizer_file = 'GazeMouse\\data\\numpy\\normalize_mean.npy'
    detector_checkpoint = 'GazeMouse\\data\\uploadable_checkpoints\\face_landmarker_v2_with_blendshapes.task'
    tracker = EyeTracker(itracker_checkpoint, torch_device, normalizer_file, detector_checkpoint)

    def capture(self):
        image_bytes = np.asarray(self.sct.grab(self.monitor))
        image_bytes = cv2.cvtColor(image_bytes, cv2.COLOR_BGR2RGB)
        captured_image = Image.fromarray(image_bytes)
        captured_image = _resize_image_to_window(captured_image, self.width, self.height)
        
        # model_inference
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_bytes)
        gaze_pred = eye_tracking_screen_recorder.tracker.e2e_gaze_prediction(img)
        
        return [captured_image, gaze_pred]