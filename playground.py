from datasets.FFHQ import FFHQ
from datasets import load_dataset
from utils.utils import dict2namespace
import os, yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image

config_path = "config/ffhq.yaml"
with open(os.path.join(config_path), "r") as f:
    config_yaml = yaml.safe_load(f)

config = dict2namespace(config_yaml)


Dataset = load_dataset(config)
dataloader = DataLoader(Dataset, batch_size=64,shuffle=True)

batch = next(iter(dataloader))