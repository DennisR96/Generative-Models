from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import os

class Dataset(data.Dataset):
    def __init__(self, config):
        super().__init__()       
        self.files = [i for i in os.listdir(config.dataset.path) if i.endswith(config.dataset.filetype)]
        self.filepaths = [os.path.join(config.dataset.path, i) for i in self.files]
        
        self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
    
    def __getitem__(self, index):
        path = self.filepaths[index]
        img = Image.open(path).convert('RGB')  

        tensor_hr = self.transform(img)
        tensor_lr = F.interpolate(tensor_hr.unsqueeze(0), 
                                  size=(16, 16), 
                                  mode='bicubic', 
                                  align_corners=False).squeeze(0)        
        
        out = {
            "HR" : tensor_hr, 
            "LR" : tensor_lr
            
        }
        return out
    
    def __len__(self):
        return len(self.filepaths)
    
