from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import os

class DIV2K(data.Dataset):
    def __init__(self, config):
        super().__init__()     
        self.config = config  
        self.files = [i for i in os.listdir(config.dataset.path) if i.endswith(".png")]
        self.filepaths = [os.path.join(config.dataset.path, i) for i in self.files]
        
        self.transform = transforms.Compose([
                transforms.Resize((config.dataset.resolution, config.dataset.resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
    
    
    def lowres_batch(self, tensor_hr):
        tensor_lr = F.interpolate(tensor_hr.unsqueeze(0), 
                                size=(self.config.dataset.lowres, self.config.dataset.lowres), 
                                mode='bicubic', 
                                align_corners=False).squeeze(0)  
        return tensor_lr
    
    
    def __getitem__(self, index):
        path = self.filepaths[index]
        img = Image.open(path).convert('RGB')  

        tensor_hr = self.transform(img)
        tensor_lr = self.lowres_batch(tensor_hr)
        
        out = {"images" : tensor_hr, 
            "lowres" : tensor_lr}
        return out
    
    def __len__(self):
        return len(self.filepaths)
    