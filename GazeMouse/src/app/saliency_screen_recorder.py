from src.app.screen_recorder import screen_recorder, _resize_image_to_window
from src.model.SaliencyMapper import SaliencyMapper
from PIL import Image
import torchvision
import numpy as np
import time

saliency_mapper = SaliencyMapper(device = 'cuda')

class saliency_screen_recorder(screen_recorder):
    def capture(self):
        screenshot = super().capture()
        
        heatmap = torchvision.transforms.functional.to_pil_image(saliency_mapper.predict(screenshot).cpu().unsqueeze(0)).resize((800,450)).convert("RGBA")
        transparency = 128
        screenshot = screenshot.convert("RGBA")
        screenshot.putalpha(transparency)
        combined = Image.alpha_composite(screenshot, heatmap)
        return _resize_image_to_window(combined, 800, 600)

class saliency_heatmap_producer(screen_recorder):
    def capture(self):
        screenshot = super().capture()
        heatmap = torchvision.transforms.functional.to_pil_image(saliency_mapper.predict(screenshot).cpu().unsqueeze(0)).resize((2560,1440)).convert("RGBA")
        heatmap = np.array(heatmap)[:,:,1] / 255
        heatmap = np.swapaxes(heatmap, 0,1)
        time.sleep(3)
        return heatmap
