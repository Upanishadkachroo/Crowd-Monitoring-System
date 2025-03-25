import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

class DensityMap:
    def __init__(self, model_path="csrnet.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model(model_path)
        self.model.eval()

    def load_model(self, model_path):
        # Load the pre-trained CSRNet model
        model = torch.hub.load("leeyeehoo/CSRNet-pytorch", "csrnet", pretrained=True)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model

    def preprocess(self, image):
        # Preprocess the image for CSRNet
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0).to(self.device)
        return image

    def generate(self, frame, centers):
        # Generate a density map using CSRNet
        input_tensor = self.preprocess(frame)
        with torch.no_grad():
            output = self.model(input_tensor)
        density_map = output.squeeze().cpu().numpy()

        # Normalize the density map for visualization
        density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
        density_map = density_map.astype(np.uint8)
        return density_map