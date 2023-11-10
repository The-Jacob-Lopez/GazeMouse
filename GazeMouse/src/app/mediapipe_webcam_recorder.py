from multiprocessing import Queue

from PIL import Image
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt

from src.app.webcam_recorder import webcam_recorder
from pathlib import Path

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for face_landmarks in face_landmarks_list:
        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image

base_options = python.BaseOptions(
    model_asset_path=str(Path('GazeMouse/data/pytorch_checkpoints/face_landmarker_v2_with_blendshapes.task')))
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# TODO: adopt this code: https://mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html#:~:text=For%20webcam%20input%3A-,drawing_spec%20%3D%20mp_drawing.DrawingSpec(thickness%3D1%2C%20circle_radius%3D1),-cap%20%3D%20cv2
class mediapipa_webcam_recorder(webcam_recorder):
    def capture(self):
        _, frame = self.vid.read()
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        captured_image = np.asarray(opencv_image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=captured_image)
        detection_result = detector.detect(mp_image)
        annotated_image = draw_landmarks_on_image(
            mp_image.numpy_view(), detection_result)
        return Image.fromarray(annotated_image)
