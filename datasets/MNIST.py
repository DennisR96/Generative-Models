from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import gzip
import numpy as np 
from PIL import Image
import lightning as L


class DATASET_MNIST(Dataset):
    """
    Custom Dataset for MNIST
    """
    def __init__(self, config):
        self.config = config
        
        # Load Images and Labels
        self.images = self.load_images("datasets/MNIST/train-images-idx3-ubyte.gz")
        self.labels = self.load_labels("datasets/MNIST/train-labels-idx1-ubyte.gz")
        
        # Transform
        self.transform =  transforms.Compose([
                transforms.Resize((self.config.dataset.resolution, 
                                   self.config.dataset.resolution)),
                transforms.ToTensor(),])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image, mode='L')
        
        if self.transform:
            image = self.transform(image)
            
        output = {
            "images" : image,
            "labels" : label,
        }
        return output
    
    def load_images(self, path):
        with gzip.open(path, 'rb') as f:
            magic_number = int.from_bytes(f.read(4), byteorder='big')
            num_images = int.from_bytes(f.read(4), byteorder='big')
            rows = int.from_bytes(f.read(4), byteorder='big')
            cols = int.from_bytes(f.read(4), byteorder='big')
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
            return images

    def load_labels(self, path):
        with gzip.open(path, 'rb') as f:
            magic_number = int.from_bytes(f.read(4), byteorder='big')
            num_labels = int.from_bytes(f.read(4), byteorder='big')
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

class MNIST(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def setup(self, stage=None):
        self.dataset = DATASET_MNIST(self.config)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.config.dataset.batch_size, num_workers=self.config.dataset.num_workers)