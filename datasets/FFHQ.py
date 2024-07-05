import torchvision.datasets as datasets
import os
from torchvision import transforms
from torch.utils import data
from PIL import Image
import torch.nn.functional as F
import torch
from diffusers import AutoPipelineForImage2Image
class FFHQ(data.Dataset):
    def __init__(self, config):
        super(FFHQ, self).__init__()
        self.config = config
        
        self.files = [i for i in os.listdir(config.dataset.path) if i.endswith(".png")]
        self.filepaths = [os.path.join(config.dataset.path, i) for i in self.files]
        
        self.transform = transforms.Compose([
                transforms.Resize((config.dataset.resolution, config.dataset.resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        #self.diffusion = AutoPipelineForImage2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
        #                                                torch_dtype=torch.float16).to("mps")
    
    # def lowres_batch(self, tensor_hr):
    #     tensor_lr = F.interpolate(tensor_hr.unsqueeze(0), 
    #                             size=(16, 16), 
    #                             mode='bicubic', 
    #                             align_corners=False).squeeze(0) 
    #     tensor_lr = F.interpolate(tensor_hr.unsqueeze(0), 
    #                             size=(self.config.dataset.lowres, self.config.dataset.lowres), 
    #                             mode='bicubic', 
    #                             align_corners=False).squeeze(0) 
    #     return tensor_lr
    
    def lowres_batch(self, tensor_hr):

        output = self.diffusion("", image=tensor_hr, strength=0.5)
        tensor_hr = output.images[0]
        return tensor_hr
        
    
    def __getitem__(self, index):
        path = self.filepaths[index]
        
        # Convert to PIL
        pil_img = Image.open(path).convert('RGB') 
        
        # Diffusion
        #pil_lr = self.lowres_batch(pil_img)
        
        # Transform
        tensor_img = self.transform(pil_img)
        #tensor_lr = self.transform(pil_lr)
        
        output = {"images" : tensor_img,
                  "lowres" : tensor_img}
        return output
    
    def __len__(self):
        return len(self.filepaths)
    
    
        
        
        
      
            



