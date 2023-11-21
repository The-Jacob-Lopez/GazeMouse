import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np

def load_data(file):
    npzfile = np.load(file)
    train_eye_left = npzfile['train_eye_left'].astype('float16')
    train_eye_right = npzfile['train_eye_right'].astype('float16')
    train_face = npzfile['train_face'].astype('float16')
    train_face_mask = npzfile['train_face_mask'].astype('float16')
    train_y = npzfile['train_y'].astype('float16')
    val_eye_left = npzfile['val_eye_left']
    val_eye_right = npzfile['val_eye_right'].astype('float16')
    val_face = npzfile['val_face'].astype('float16')
    val_face_mask = npzfile['val_face_mask'].astype('float16')
    val_y = npzfile['val_y'].astype('float16')
    return [train_face, train_eye_left, train_eye_right, train_face_mask, train_y], [val_face, val_eye_left, val_eye_right, val_face_mask, val_y]

def normalize(data):
    shape = data.shape
    data = np.reshape(data, (shape[0], -1))
    data = data / 255. # scaling
    data = data - np.mean(data, axis=0) # normalizing
    return np.reshape(data, shape)

def prepare_data(data):
    face, eye_left, eye_right, face_mask, y = data
    eye_left = normalize(eye_left)
    eye_right = normalize(eye_right)
    face = normalize(face)
    face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
    return [face, eye_left, eye_right, face_mask, y]


class ITrackerData(torch.utils.data.Dataset):
    def __init__(self, data, im_shape=(224,224), grid_shape=(25, 25)):
        self.face, self.eye_left, self.eye_right, self.face_mask, self.y = data

        self.imSize = im_shape
        self.gridSize = grid_shape

    def __getitem__(self, index):

        face, eye_left, eye_right, face_mask, y = \
             self.face[index], self.eye_left[index], \
             self.eye_right[index], self.face_mask[index], \
             self.y[index]

        # Make into torch tensors
        face, eye_left, eye_right, face_mask, y = \
            torch.FloatTensor(face),\
            torch.FloatTensor(eye_left),\
            torch.FloatTensor(eye_right),\
            torch.FloatTensor(face_mask),\
            torch.FloatTensor(y)

        # Permute images
        face, eye_left, eye_right = \
           face.permute(2,0,1),\
            eye_left.permute(2,0,1),\
            eye_right.permute(2,0,1)

        # Resize images
        resize = transforms.Resize(self.imSize)
        face, eye_left, eye_right = \
            resize(face),\
            resize(eye_left),\
            resize(eye_right)

        return face, eye_left, eye_right, face_mask, y


    def __len__(self):
        return len(self.y)

class TrainITrackerData(ITrackerData):
    def __init__(self, data_path, im_shape=(224,224), grid_shape=(25, 25)):
        super().__init__(prepare_data(load_data(data_path)[0]), im_shape, grid_shape)

class TestITrackerData(ITrackerData):
    def __init__(self, data_path, im_shape=(224,224), grid_shape=(25, 25)):
        super().__init__(prepare_data(load_data(data_path)[1]), im_shape, grid_shape)