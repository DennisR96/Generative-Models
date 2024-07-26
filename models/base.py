from torch import nn, optim
import torch
from torch.utils.data import DataLoader

from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from utils.utils import namespace2dict
import wandb
import pandas as pd
import os

class Base():
    """
    General Generative Model Class
    
    """    
    def __init__(self, config, dataset, networks):
        # -- 1. Configuration --
        self.config = config
        self.device = self.get_device()
        
        # -- 2. Dataloader -- 
        self.dataloader = DataLoader(dataset, 
                                     batch_size=self.config.model.dataloader.batch_size, 
                                     shuffle=True, 
                                     num_workers=self.config.model.dataloader.num_workers)
        
        # -- 3. Project Folder -- 
        self.path = f"results/{self.config.log.project}_{self.config.log.id}"
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
            os.makedirs(os.path.join(self.path, "models"))
            os.makedirs(os.path.join(self.path, "inference"))        
        
        # -- 4. Wandb Logging -- 
        if self.config.log.active:
            config_dict = namespace2dict(config)
            wandb.init(
                project=config.log.project,
                config=config_dict,
                id=config.log.id) 
        pass
    
    def get_device(self):
        """
        Setup Available Device
        """
        if self.config.device:
            return self.config.device
        elif torch.cuda.is_available():
            print("-- CUDA Device detected --")
            device = "cuda"
        elif torch.backends.mps.is_available():
            print("-- MPS Device detected --")
            device = "mps"
        else:
            print("-- CPU Device detected --")
            device = "cpu"            
        return device

    def get_optimizer(self, parameters):
        """
        Setup Model Optimizer

        Args:
            parameters (_type_):    PyTorchModel.parameters()
        """
        if self.config.model.optimizer.name == 'ADAM':
            return optim.Adam(parameters, lr=self.config.model.optimizer.lr, weight_decay=self.config.model.optimizer.weight_decay,
                              betas=(self.config.model.optimizer.beta1, 0.999), amsgrad=self.config.model.optimizer.amsgrad)
        elif self.config.model.optimizer.name == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.model.optimizer.lr, weight_decay=self.config.model.optimizer.weight_decay)
        elif self.config.model.optimizer.name == 'SGD':
            return optim.SGD(parameters, lr=self.config.model.optimizer.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.model.optimizer.name))
    
    def get_scheduler(self, optimizer):
        """
        Scheduler Setup

        Returns:
            _type_: _description_
        """
        if self.config.model.scheduler.name == 'MultiStepLR':
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config.model.scheduler.milestones)
        return 0   
        
    def model_load(self, model, path):
        return model.load_state_dict(torch.load(path))
        
    def model_save(self, model, path):
        """
        Save PyTorch Model as .pth
        """
        torch.save(model.state_dict(), path)
        return 0
    
    def train(self):
        for epoch in tqdm(range(self.config.model.train.num_epochs), desc='Epochs'):
            for step, batch in enumerate(tqdm(self.dataloader, desc='Steps', leave=False)):
                self.model_update_parameters(batch)
        return 0
    
    def inference(self):
        return 0
    
        
        
    