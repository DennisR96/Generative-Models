import torchvision.datasets as datasets
import os
from torchvision import transforms
from torch.utils import data
from PIL import Image
import torch.nn.functional as F

class FFHQ(data.Dataset):
    def __init__(self, config):
        super(FFHQ, self).__init__()
        
        self.files = [i for i in os.listdir(config.dataset.path) if i.endswith(".png")]
        self.filepaths = [os.path.join(config.dataset.path, i) for i in self.files]
        
        self.transform = transforms.Compose([
                transforms.Resize((config.dataset.resolution, config.dataset.resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def lowres_batch(self, tensor_hr):
        tensor_lr = F.interpolate(tensor_hr.unsqueeze(0), 
                                size=(16, 16), 
                                mode='bicubic', 
                                align_corners=False).squeeze(0)  
        return tensor_lr
    
    def __getitem__(self, index):
        path = self.filepaths[index]
        img = Image.open(path).convert('RGB') 
        
        tensor = self.transform(img) 
        output = {"images" : tensor}
        return output
    
    def __len__(self):
        return len(self.filepaths)
    
    
        
        
        
      
            



