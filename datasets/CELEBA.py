import lightning as L
import os
import gdown
import zipfile
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch.nn.functional as F
from PIL import Image


class DATASET_CELEBA(Dataset):
    def __init__(self, config):
        self.config = config
        self.files = [i for i in os.listdir("datasets/CELEBA") if i.endswith(".jpg")]
        self.filepaths = [os.path.join("datasets/CELEBA", i) for i in self.files]
        self.transform = transforms.Compose([
                transforms.Resize((config.dataset.resolution, config.dataset.resolution)),
                transforms.ToTensor(),])
    
    def __getitem__(self, index):
        path = self.filepaths[index]
        img = Image.open(path).convert('RGB') 
        img = self.transform(img)
        lowres = F.interpolate(img.unsqueeze(0), 
                                size=(self.config.dataset.lowres_resolution, self.config.dataset.lowres_resolution), 
                                mode='bicubic', 
                                align_corners=False).squeeze(0) 
        return {"images" : img, 
                "lowres": lowres,}
    
    def __len__(self):
        return len(self.filepaths)


class CELEBA(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def prepare_data(self):
        # Download Dataset
        if not os.path.isdir("datasets/CELEBA"):
            print("Downloading Dataset")
            gdown.download(id="1oLjeNuWWOBlcaRUkw2J49BDEoFi3V1-g", output="celebA.zip")
            with zipfile.ZipFile("celebA.zip", 'r') as zip_ref:
                zip_ref.extractall("datasets/")
            os.remove("celebA.zip")
        else:
            print("Dataset detected")
            
    def setup(self, stage=None):
        self.dataset = DATASET_CELEBA(self.config)
        
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.config.dataset.batch_size, num_workers=self.config.dataset.num_workers)
    



