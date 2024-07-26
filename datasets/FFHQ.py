import torchvision.datasets as datasets
import os
from torchvision import transforms
from torch.utils import data
from PIL import Image
import torch.nn.functional as F
import torch

class FFHQ(data.Dataset):
    def __init__(self, config):
        super(FFHQ, self).__init__()
        self.config = config
        
        self.files = [i for i in os.listdir(config.dataset.path) if i.endswith(".png")]
        self.filepaths = [os.path.join(config.dataset.path, i) for i in self.files]
        
        self.transform = transforms.Compose([
                transforms.Resize((config.dataset.resolution, config.dataset.resolution)),
                transforms.ToTensor(),
                ])
    
    def lowres_batch(self, tensor_hr):

        output = self.diffusion("", image=tensor_hr, strength=0.5)
        tensor_hr = output.images[0]
        return tensor_hr
        
    
    def __getitem__(self, index):
        path = self.filepaths[index]
        pil_img = Image.open(path).convert('RGB') 
        tensor_img = self.transform(pil_img)
        
        output = {"images" : tensor_img,
                  "lowres" : tensor_img}
        return output
    
    def __len__(self):
        return len(self.filepaths)
    
    
        
        
        
      
            



