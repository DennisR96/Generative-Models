from models.gan.esrgan import ESRGAN
from datasets.CELEBA import CELEBA
from networks import load_network
from utils.utils import dict2namespace
import lightning as L


import os, yaml
from torch.utils.data import DataLoader

# 1. Load Configuration
config_path = "config/ESRGAN/celeba.yaml"
with open(os.path.join(config_path), "r") as f:
    config_yaml = yaml.safe_load(f)

config = dict2namespace(config_yaml)

# 2. Load Network
network = load_network(config)

model = ESRGAN(config, network)

# Load Dataset
ds = CELEBA(config)
train_dataloader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True,
)
trainer = L.Trainer(accelerator="cuda", fast_dev_run=7, profiler="simple")
trainer.fit(model=model, train_dataloaders=train_dataloader)