from scipy.interpolate import RBFInterpolator
import numpy as np

class eye_tracking_filter:
    def __init__(self, calibration_data, screen_resolution, history_size):
        x_val = calibration_data[:,0].tolist()
        y_val = calibration_data[:,1].tolist()
        pixel_corners = [[0,0],[screen_resolution[0],0],[0,screen_resolution[1]], [screen_resolution[0],screen_resolution[1]]]
        self.region_of_interpolation = [[min(x_val), max(x_val)],[min(y_val), max(y_val)]]
        self.interp = RBFInterpolator(calibration_data, pixel_corners)
        self.history = [[0,0] for i in range(history_size)]
        
    def clip_to_region(self, gaze_pred):
        ep = 1*10**-2
        gaze_x, gaze_y = gaze_pred[0]
        nearest_x = min(max(gaze_x + ep, self.region_of_interpolation[0][0] + ep) - ep, self.region_of_interpolation[0][1] - ep)
        nearest_y = min(max(gaze_y + ep, self.region_of_interpolation[1][0] + ep) - ep, self.region_of_interpolation[1][1] - ep)
        return [nearest_x, nearest_y]
    
    def __call__(self, gaze_pred):
        filtered_pred = self.interp(gaze_pred)[0]
        self.history = self.history[1:]
        self.history.append(filtered_pred)
        return np.mean(np.array(self.history), axis=0)