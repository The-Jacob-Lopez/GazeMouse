from scipy.interpolate import RBFInterpolator
from scipy.interpolate import LinearNDInterpolator
import numpy as np
from scipy.spatial import KDTree
import math 

class eye_tracking_filter:
    def __init__(self, calibration_data, screen_resolution, history_size):
        #self.interp = RBFInterpolator(calibration_data[0], calibration_data[1], neighbors=50, smoothing=1, kernel='gaussian', epsilon=1)
        self.interp = LinearNDInterpolator(calibration_data[0], calibration_data[1])
        self.history = [[0,0] for i in range(history_size)]
        self.curr_pos = np.array([0,0])
        self.sensitivity = 0.05
        self.screen_resolution = screen_resolution
        self.region_of_interpolation = [[min(calibration_data[0,:,0]), max(calibration_data[0,:,0])],[min(calibration_data[0,:,1]), max(calibration_data[0,:,1])]]
        self.kdtree = KDTree(calibration_data[0])
        self.calibration_data = calibration_data
        
    def clip_to_region(self, gaze_pred):
        ep = 1*10**-2
        gaze_x, gaze_y = gaze_pred
        nearest_x = min(max(gaze_x + ep, self.region_of_interpolation[0][0] + ep) - ep, self.region_of_interpolation[0][1] - ep)
        nearest_y = min(max(gaze_y + ep, self.region_of_interpolation[1][0] + ep) - ep, self.region_of_interpolation[1][1] - ep)
        return [nearest_x, nearest_y]
    
    def clip_to_screen(self, pos):
        x, y = pos
        nearest_x = min(max(x, 0), self.screen_resolution[0])
        nearest_y = min(max(y, 0), self.screen_resolution[1])
        return [nearest_x, nearest_y]
    
    def __call__(self, gaze_pred):
        filtered_pred = self.interp(gaze_pred)[0]
        if math.isnan(filtered_pred[0]):
            d,i = self.kdtree.query(gaze_pred, k=1)
            filtered_pred = self.interp(self.calibration_data[0][i])[0]
        self.history.pop(0)
        self.history.append(filtered_pred)
        estimated_pos = np.mean(np.array(self.history), axis=0)
        self.curr_pos = self.clip_to_screen(self.curr_pos + self.sensitivity*(estimated_pos - self.curr_pos))
        return self.curr_pos