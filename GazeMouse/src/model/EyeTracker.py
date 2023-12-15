from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.model.ITrackerModel import ITrackerModel
import mediapipe as mp
import numpy
import numpy as np
import math
import cv2
import torch
from torchvision import transforms

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int):
  """Converts normalized value pair to pixel coordinates."""
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def item_crop(image, detection_result, landmark_indeces):
    landmarks = [detection_result.face_landmarks[0][i] for i in landmark_indeces]
    xy = [_normalized_to_pixel_coordinates(x.x, x.y, image.shape[1], image.shape[0]) for x in landmarks]
    minx = min([x[0] for x in xy])
    miny = min([x[1] for x in xy])
    maxx = max([x[0] for x in xy])
    maxy = max([x[1] for x in xy])
    return image[miny:maxy, minx: maxx]

def item_mask(image, detection_result, landmark_indeces):
    landmarks = [detection_result.face_landmarks[0][i] for i in landmark_indeces]
    xy = [_normalized_to_pixel_coordinates(x.x, x.y, image.shape[1], image.shape[0]) for x in landmarks]
    minx = min([x[0] for x in xy])
    miny = min([x[1] for x in xy])
    maxx = max([x[0] for x in xy])
    maxy = max([x[1] for x in xy])
    row_mask = numpy.ma.getmask(numpy.ma.masked_inside(range(image.shape[1]), minx, maxx))
    column_mask = numpy.ma.getmask(numpy.ma.masked_inside(range(image.shape[0]), miny, maxy))
    return numpy.outer(column_mask, row_mask)

def get_static_normalize(normalizer_file):
    with open(normalizer_file, 'rb') as f:
        mean = np.load(f)
    def static_normalize(data):
        shape = data.shape
        data = np.reshape(data, (-1))
        data = data / 255. # scaling
        data = data - mean # normalizing
        return np.reshape(data, shape)
    return static_normalize

def prepare_model(torch_device, itracker_checkpoint):
    model = ITrackerModel()
    model.load_state_dict(torch.load(itracker_checkpoint, map_location=torch_device))
    model = model.to(torch_device)
    model.eval()
    return model

def prepare_detector(detector_checkpoint):
    base_options = python.BaseOptions(model_asset_path=detector_checkpoint)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

class EyeTracker:
    imsize = 224
    left_eye_landmark_indeces = [343, 283]
    right_eye_landmark_indeces = [53, 114]
    face_landmark_indeces = {item for sublist in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for item in sublist}
    
    def __init__(self, itracker_checkpoint, torch_device, normalizer_file, detector_checkpoint):
        self.itracker_checkpoint = itracker_checkpoint
        self.torch_device = torch_device
        self.normalizer = get_static_normalize(normalizer_file)
        self.model = prepare_model(torch_device, itracker_checkpoint)
        self.detector = prepare_detector(detector_checkpoint)

    def prepare_data(self, face, eye_left, eye_right, face_mask):
        # resize inputs to be 64 x 64 images, consistent with training data
        face, eye_left, eye_right, face_mask = \
            cv2.resize(face, (64, 64), cv2.INTER_NEAREST), \
            cv2.resize(eye_left, (64, 64), cv2.INTER_NEAREST), \
            cv2.resize(eye_right, (64, 64), cv2.INTER_NEAREST), \
            cv2.resize(np.float32(face_mask), (25, 25), cv2.INTER_NEAREST), \
                
        eye_left = self.normalizer(eye_left)
        eye_right = self.normalizer(eye_right)
        face = self.normalizer(face)
        face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
        # Make into torch tensors
        face, eye_left, eye_right, face_mask = \
            torch.FloatTensor(face),\
            torch.FloatTensor(eye_left),\
            torch.FloatTensor(eye_right),\
            torch.FloatTensor(face_mask)

        # Permute images
        face, eye_left, eye_right = \
        face.permute(2,0,1),\
            eye_left.permute(2,0,1),\
            eye_right.permute(2,0,1)

        # Resize images
        resize = transforms.Resize(EyeTracker.imsize)
        face, eye_left, eye_right = \
            resize(face),\
            resize(eye_left),\
            resize(eye_right)
        
        # To device
        face, eye_left, eye_right, face_mask = face.to(self.torch_device), eye_left.to(self.torch_device), eye_right.to(self.torch_device), face_mask.to(self.torch_device)
        
        #add batch dimension to make it compatible with model
        face, eye_left, eye_right, face_mask = face.unsqueeze(0), eye_left.unsqueeze(0), eye_right.unsqueeze(0), face_mask.unsqueeze(0)
        return face, eye_left, eye_right, face_mask

    # May return None in the case that the detector does not detect a face
    def get_item_crops(self, image):
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(frame)
        if len(detection_result.face_landmarks) == 0:
            return None

        frame = frame.numpy_view()

        left_eye_crop = item_crop(frame, detection_result, EyeTracker.left_eye_landmark_indeces)
        right_eye_crop = item_crop(frame, detection_result, EyeTracker.right_eye_landmark_indeces)
        face_crop = item_crop(frame, detection_result, EyeTracker.face_landmark_indeces)
        face_mask = item_mask(frame, detection_result, EyeTracker.face_landmark_indeces)
        for crop in [left_eye_crop, right_eye_crop, face_crop, face_mask]:
            if len(crop) == 0:
                return None
        return face_crop, left_eye_crop, right_eye_crop, face_mask
    
    # May return None in the case that the detector does not detect a face
    def e2e_gaze_prediction(self, image):
        crops = self.get_item_crops(image)
        if crops is None:
            return None
        try:
            return self.gaze_prediction(*crops)
        except Exception:
            return None
    
    # May return None in the case that the detector does not detect a face
    def gaze_prediction(self, face_crop, left_eye_crop, right_eye_crop, face_mask):
        face, eyes_left, eyes_right, mask = self.prepare_data(face_crop, left_eye_crop, right_eye_crop, face_mask)
        #face, eyes_left, eyes_right = [torch.permute(x, (0,2,3,1)) for x in [face, eyes_left, eyes_right]]
        return self.model(face, eyes_left, eyes_right, mask)[0].cpu().detach().numpy()
