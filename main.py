import os, yaml
from datasets import load_dataset
from utils.utils import dict2namespace

from networks import load_network
from models import load_model

import lightning as L
import lightning.pytorch.loggers as loggers
import lightning.pytorch.callbacks as callbacks

# -- 1. Load Configuration --
config_path = "config/ESRGAN/celeba.yaml"
with open(os.path.join(config_path), "r") as f:
    config_yaml = yaml.safe_load(f)

config = dict2namespace(config_yaml)  

# -- 2. Dataset --
dm = load_dataset(config)
dm.setup()

batch = next(iter(dm.train_dataloader()))

# -- 3. Network -- 
network = load_network(config)

# -- 4. Model -- 
model = load_model(config, network)

# -- 5. Trainer -- 
# Loggers
TensorBoard = loggers.TensorBoardLogger(
    save_dir=f"results/",
    name=config.log.project,
    version=config.log.id,
    )

# Callbacks
checkpoint = callbacks.ModelCheckpoint(
    dirpath=f"results/{config.log.project}/{config.log.id}/Checkpoints",
    save_last=True,
    save_top_k=-1,
    every_n_epochs=1,
    )

lr_scheduler = callbacks.LearningRateMonitor(logging_interval="step")

# Trainer
trainer = L.Trainer(
    default_root_dir="results", 
    callbacks=[checkpoint, lr_scheduler],
    accelerator="auto", 
    log_every_n_steps=10, 
    logger=[TensorBoard],
    max_epochs=10000,
    limit_train_batches=2)

trainer.fit(model, dm, ckpt_path="last")