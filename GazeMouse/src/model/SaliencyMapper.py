import torch
import torchvision
from pathlib import Path
from src.model.SaliencyModel import TranSalNet, TRANSALNET_HEIGHT, TRANSALNET_WIDTH

# Download from https://drive.google.com/file/d/1-LC6MdvsYdgisCWJbIvklifr1ZeDjz7q/view?usp=drive_link
checkpoint = 'GazeMouse/data/uploadable_checkpoints/best_transalnet_model.pth'

class SaliencyMapper:
    """
    Wraps the TranSalNet model. 
    """

    def __init__(self, checkpoint=checkpoint, device="cpu"):
        """
        Creates a TranSalNet model, loads weights from the default path, and creates
        the transforms necessary to run inference on an input.
        """
        self.model = TranSalNet()
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(str(Path(checkpoint)), map_location=torch.device(device)))

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((TRANSALNET_HEIGHT, TRANSALNET_WIDTH)),
            torchvision.transforms.ToTensor(),
        ])
    
    def _process_img(self, img):
        # Convert to 3 channels
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # Apply transformations
        img = self.transforms(img)

        # Add bs if necessary
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        
        return img

    def _postprocess_pred(self, pred):
        return pred.squeeze()


    def predict(self, img):
        """
        Predicts the saliency on an RGB Pillow Image.
        """
        img = self._process_img(img)
        pred = self.model.forward(img)
        return self._postprocess_pred(pred)

