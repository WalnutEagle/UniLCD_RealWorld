import torch
from torchvision import transforms
from PIL import Image

class CarlaRunDataset:
    def __init__(self, rgb_image, depth_image):
        self.rgb_image = rgb_image
        self.depth_image = depth_image
        
        # Define the transformations
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),  # Resize to 300x300
            transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
        ])

    def preprocess(self):
        """
        Preprocess RGB and depth images.
        
        Returns:
            torch.Tensor: Concatenated and preprocessed image tensor.
        """
        # Process RGB image
        rgb_pil = Image.fromarray(self.rgb_image)
        rgb_tensor = self.transform(rgb_pil)

        # Process depth image
        depth_pil = Image.fromarray(self.depth_image)
        depth_tensor = self.transform(depth_pil)

        # Concatenate the images along the channel dimension
        combined_image = torch.cat((rgb_tensor, depth_tensor), dim=0)  # Shape: (2*C, H, W)

        return combined_image

def get_single_image_dataloader(rgb_image, depth_image):
    dataset = CarlaRunDataset(rgb_image, depth_image)
    combined_image = dataset.preprocess()
    
    return combined_image
