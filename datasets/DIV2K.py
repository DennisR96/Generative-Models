from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import os
import lightning as L

class DATASET_DIV2K(data.Dataset):
    def __init__(self, config):
        super().__init__()     
        self.config = config  
        self.files = [i for i in os.listdir("datasets/DIV2K") if i.endswith(".png")]
        self.filepaths = [os.path.join("datasets/DIV2K", i) for i in self.files]
        
        self.transform = transforms.Compose([
                transforms.Resize((config.dataset.resolution, config.dataset.resolution)),
                transforms.ToTensor()])    
    
    def __getitem__(self, index):
        path = self.filepaths[index]
        img = Image.open(path).convert('RGB')  

        tensor_hr = self.transform(img)
        tensor_lr = F.interpolate(tensor_hr.unsqueeze(0), 
                                size=(self.config.dataset.lowres_resolution, self.config.dataset.lowres_resolution), 
                                mode='bicubic', 
                                align_corners=False).squeeze(0)  
        
        out = {"images" : tensor_hr, 
            "lowres" : tensor_lr}
        return out
    
    def __len__(self):
        return len(self.filepaths)

class DIV2K(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
            
    def setup(self, stage=None):
        self.dataset = DATASET_DIV2K(self.config)
        
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.config.dataset.batch_size, num_workers=self.config.dataset.num_workers)