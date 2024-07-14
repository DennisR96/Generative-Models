import os
import yaml
from utils.utils import dict2namespace
import torch
import wandb

from datasets import load_dataset
from models import load_model
from networks import load_network

# 1. Load Configuration
config_path = "config/celeba.yaml"
with open(os.path.join(config_path), "r") as f:
    config_yaml = yaml.safe_load(f)

config = dict2namespace(config_yaml)

# 2. Logging
wandb.init(
    project="my-awesome-project",
    config=config_yaml)

# 2. Load Network
network = load_network(config)

# 3. Load Dataset
dataset = load_dataset(config) 

# 4. Load Model
model = load_model(config, dataset,network)
model.train()

