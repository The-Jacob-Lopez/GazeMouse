import torch
import torchvision
import Path
from src.model.SaliencyModel import TranSalNet, TRANSALNET_HEIGHT, TRANSALNET_WIDTH


class SaliencyMapper:
    """
    Wraps the TranSalNet model. 
    """

    def __init__(self,
                 weights="GazeMouse/data/raw/weights/best_transalnet_model.pth"):
        """
        Creates a TranSalNet model, loads weights from the default path, and creates
        the transforms necessary to run inference on an input.
        """
        self.model = TranSalNet()
        weights = str(Path(weights))
        self.model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((TRANSALNET_WIDTH, TRANSALNET_HEIGHT)),
            torchvision.transforms.ToTensor(),
        ])

    def predict(self, x):
        """
        Transforms x to input of transform shape.
        """
        x = self.transforms(x)
        x = self.model(x)
        return x

