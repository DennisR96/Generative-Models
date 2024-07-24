import os
import zipfile

import gdown
from torchvision import transforms
from torch.utils import data
from PIL import Image
import torch.nn.functional as F

class CELEBA(data.Dataset):
    def __init__(self, config):
        super(CELEBA, self).__init__()
        self.config = config
        
        # Check if the directory exists
        if not os.path.isdir("datasets/CELEBA"):
            print("–– Downloading Dataset ––")
            gdown.download(id="1oLjeNuWWOBlcaRUkw2J49BDEoFi3V1-g", output="celebA.zip")
            with zipfile.ZipFile("celebA.zip", 'r') as zip_ref:
                zip_ref.extractall("datasets/")
            os.remove("celebA.zip")
        else:
            print("–– Dataset detected locally ––")
        
        self.files = [i for i in os.listdir("datasets/CELEBA") if i.endswith(".jpg")]
        self.filepaths = [os.path.join("datasets/CELEBA", i) for i in self.files]
        
        # Transformation
        self.transform = transforms.Compose([
                transforms.Resize((config.dataset.resolution, config.dataset.resolution)),
                transforms.ToTensor(),])
    
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
    