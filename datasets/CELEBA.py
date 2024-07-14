import torchvision.datasets as datasets
import os
from torchvision import transforms
from torch.utils import data
from PIL import Image
import torch.nn.functional as F
import torch

class CELEBA(data.Dataset):
    def __init__(self, config):
        super(CELEBA, self).__init__()
        self.config = config
        
        self.files = [i for i in os.listdir(config.dataset.path) if i.endswith(".jpg")]
        self.filepaths = [os.path.join(config.dataset.path, i) for i in self.files]
        self.transform = transforms.Compose([
                transforms.Resize((config.dataset.resolution, config.dataset.resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def lowres_batch(self, tensor_hr):
        tensor_lr = F.interpolate(tensor_hr.unsqueeze(0), 
                                size=(self.config.dataset.lowres_resolution, self.config.dataset.lowres_resolution), 
                                mode='bicubic', 
                                align_corners=False).squeeze(0) 
        return tensor_lr
    
    def __getitem__(self, index):
        path = self.filepaths[index]
        img = Image.open(path).convert('RGB') 
        
        tensor = self.transform(img) 
        tensor_lr = self.lowres_batch(tensor)
        
        output = {"images" : tensor,
                  "lowres" : tensor_lr}
        return output
    
    def __len__(self):
        return len(self.filepaths)
    