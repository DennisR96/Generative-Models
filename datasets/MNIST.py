from torch.utils import data
import torchvision.datasets as datasets
from torchvision import transforms
import gzip
import numpy as np 
from PIL import Image
import torch.nn.functional as F

class MNIST(data.Dataset):
    def __init__(self, config):
        self.config = config
        self.images = self.load_images(config.dataset.path_images)
        self.labels = self.load_labels(config.dataset.path_labels)
        
        self.transform =  transforms.Compose([
                transforms.Resize((self.config.dataset.resolution, 
                                   self.config.dataset.resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image, mode='L')
        
        if self.transform:
            image = self.transform(image)
            
        output = {
            "images" : image,
            "labels" : label,
        }
        
        if self.config.dataset.lowres:
            lowres = F.interpolate(image.unsqueeze(0), 
                                  size=(16, 16), 
                                  mode='bicubic', 
                                  align_corners=False).squeeze(0)
            lowres = F.interpolate(lowres.unsqueeze(0), 
                                   size=(64, 64), 
                                   mode='bicubic', 
                                   align_corners=False).squeeze(0)
            output["lowres"] = lowres    
        return output
    
    def load_images(self, path):
        with gzip.open(path, 'rb') as f:
            magic_number = int.from_bytes(f.read(4), byteorder='big')
            num_images = int.from_bytes(f.read(4), byteorder='big')
            rows = int.from_bytes(f.read(4), byteorder='big')
            cols = int.from_bytes(f.read(4), byteorder='big')
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
            return images

    def load_labels(self, path):
        with gzip.open(path, 'rb') as f:
            magic_number = int.from_bytes(f.read(4), byteorder='big')
            num_labels = int.from_bytes(f.read(4), byteorder='big')
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels