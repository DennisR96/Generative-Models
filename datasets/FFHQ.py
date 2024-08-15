import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import lightning as L


class DATASET_FFHQ(Dataset):
    def __init__(self, config):
        super(DATASET_FFHQ, self).__init__()
        self.config = config
        
        self.files = [i for i in os.listdir("datasets/FFHQ_DIFF") if i.endswith(".png")]
        self.filepaths_diff = [os.path.join("datasets/FFHQ_DIFF", i) for i in self.files]
        self.filepaths = [os.path.join("datasets/FFHQ", i) for i in self.files]
        
        
        self.transform = transforms.Compose([
                transforms.Resize((config.dataset.resolution, config.dataset.resolution)),
                transforms.ToTensor(),
                ])
    
    def lowres_batch(self, tensor_hr):

        output = self.diffusion("", image=tensor_hr, strength=0.5)
        tensor_hr = output.images[0]
        return tensor_hr
        
    
    def __getitem__(self, index):
        # Load Normal Image
        path = self.filepaths[index]
        pil_img = Image.open(path).convert('RGB') 
        tensor_img = self.transform(pil_img)
        
        # Load Diffusion Image
        path_diff = self.filepaths_diff[index]
        pil_diff_img = Image.open(path_diff).convert('RGB') 
        tensor_dif_img = self.transform(pil_diff_img)
        
        output = {"images" : tensor_img,
                  "lowres" : tensor_dif_img
                  }
        return output
    
    def __len__(self):
        return len(self.filepaths)
    
    
class FFHQ(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def setup(self, stage=None):
        self.dataset = DATASET_FFHQ(self.config)
        
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.config.dataset.batch_size, num_workers=self.config.dataset.num_workers)       
        
        
    
        



