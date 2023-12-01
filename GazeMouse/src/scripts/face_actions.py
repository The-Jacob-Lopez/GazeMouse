from src.app.app_settings import app_settings
from src.app.predictive_webcam_recorder import predictive_webcam_recorder
from src.app.app_settings import app_settings
from multiprocessing import Queue, freeze_support
import multiprocessing
import numpy as np
from pathlib import Path

relevant_categories = ['eyeBlinkLeft', 'eyeBlinkRight']

def run():
    settings = app_settings()
    actions = np.array(settings.actions)

    webcam_capture_queue = Queue(maxsize=1)
    tracker_pred_queue = Queue(maxsize=1)
    detector_pred_queue = Queue(maxsize=1)

    webcam_capture = predictive_webcam_recorder(webcam_capture_queue, tracker_pred_queue, detector_pred_queue)
    multiprocessing.Process(target=webcam_capture.start_processing).start()

    def clear():
        captured_image = webcam_capture_queue.get()
        tracker_pred = tracker_pred_queue.get()
        detector_pred = detector_pred_queue.get()

    preds = []
    clear()
    for action in actions:
        input(f'Give facial expression for action: {action}')
        clear()
        history = []
        for i in range(100):
            captured_image = webcam_capture_queue.get()
            tracker_pred = tracker_pred_queue.get()
            detector_pred = detector_pred_queue.get().face_blendshapes[0]
            detector_pred = [category for category in detector_pred if category.category_name in relevant_categories]
            detector_pred = [category.score for category in detector_pred]
            print(detector_pred)
            history.append(detector_pred)
        preds.append(np.mean(history, axis=0))

    with open(str(Path('GazeMouse/data/numpy/expressions.npy')), 'wb') as f:
        np.save(f, np.array(preds))    
    
    print('Done')

if __name__ == "__main__":
    freeze_support()
    run()

    