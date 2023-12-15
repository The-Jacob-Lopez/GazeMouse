from src.app.app_settings import app_settings
from src.app.predictive_webcam_recorder import predictive_webcam_recorder
from src.app.app_settings import app_settings
from multiprocessing import Queue, freeze_support
import multiprocessing
import numpy as np
from src.app.expression_metric import expression_metric
from pathlib import Path

relevant_categories = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']
def convert_detection_to_user_action(detection, history, metric):
    detector_pred = detection.face_blendshapes[0]
    detector_pred = [category for category in detector_pred if category.category_name in relevant_categories]
    detector_pred = [category.score for category in detector_pred]
    history.pop(0)
    history.append(detector_pred)
    #detector_pred = np.mean(history, axis=0)
    closest_action = metric.get_most_common_user_action(history)
    return closest_action

def run():
    settings = app_settings()
    actions = np.array(settings.actions)

    webcam_capture_queue = Queue(maxsize=1)
    tracker_pred_queue = Queue(maxsize=1)
    detector_pred_queue = Queue(maxsize=1)

    webcam_capture = predictive_webcam_recorder(webcam_capture_queue, tracker_pred_queue, detector_pred_queue)
    multiprocessing.Process(target=webcam_capture.start_processing).start()

    with open(str(Path('GazeMouse/data/numpy/expressions.npy')), 'rb') as f:
        action_expressions = np.load(f)
    
    metric = expression_metric(actions, action_expressions)
    print(metric.action_expressions)
    history = [[0]*len(relevant_categories) for i in range(10)]
    while True:
        captured_image = webcam_capture_queue.get()
        tracker_pred = tracker_pred_queue.get()
        detector_pred = detector_pred_queue.get()
        closest_action = convert_detection_to_user_action(detector_pred, history, metric)
        print(closest_action)

if __name__ == "__main__":
    freeze_support()
    run()