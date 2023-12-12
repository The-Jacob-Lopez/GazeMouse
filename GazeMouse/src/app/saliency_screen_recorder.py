from src.app.screen_recorder import screen_recorder, _resize_image_to_window
from src.model.SaliencyMapper import SaliencyMapper
from PIL import Image


class saliency_screen_recorder(screen_recorder):
    saliency_mapper = SaliencyMapper()

    def capture(self):
        screenshot = super().capture()
        heatmap = self.saliency_mapper(screenshot)
        transparency = 128
        combined = Image.alpha_composite(screenshot.convert("RGBA").putalpha(transparency), heatmap)
        return _resize_image_to_window(combined, 800, 600)

