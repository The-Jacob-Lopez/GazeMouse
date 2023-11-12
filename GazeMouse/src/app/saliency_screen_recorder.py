from src.app.screen_recorder import screen_recorder, _resize_image_to_window
from src.models.saliency_model import TranSalNet
from pathlib import Path
from PIL import Image
import torch
import torchvision


model = TranSalNet()
weights = str(Path("GazeMouse/data/raw/weights/best_transalnet_model.pth"))
model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))

TRANSALNET_WIDTH = 384
TRANSALNET_HEIGHT = 288

# Resize and convert to tensor. TranSalNet takes these sizes as inputs.
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((TRANSALNET_HEIGHT, TRANSALNET_WIDTH)),
    torchvision.transforms.ToTensor(),
])

class saliency_screen_recorder(screen_recorder):
    def capture():
        screenshot = super().capture()
        screenshot = transforms(screenshot)
        heatmap = model(screenshot)
        transparency = 128
        combined = Image.alpha_composite(screenshot.convert("RGBA").putalpha(transparency), heatmap)
        return _resize_image_to_window(combined, 800, 600)

