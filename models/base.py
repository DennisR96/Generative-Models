from torch import nn, optim
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio
from torchvision import transforms

from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import wandb
import pandas as pd
import os

class Base():
    """
    General Generative Model Class
    
    """    
    def __init__(self, config):
        self.config = config
        self.device = self.get_device()
        
        # Create Run
        self.path = f"results/{self.config.log.project}_{self.config.log.id}"
        
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
            os.makedirs(os.path.join(self.path, "models"))
            os.makedirs(os.path.join(self.path, "inference"))
            os.makedirs(os.path.join(self.path, "logs"))
        
        # Local
        self.df = pd.DataFrame()
        
        # Wandb Logging
        if self.config.log.wandb:
            wandb.init(
                project=config.log.project,
                config=config,
                id=config.log.id) 
            
        pass
    
    def log(self, run, models=None, inferences=None):
        if self.config.log.local:
            self.log_local(run, models, inferences)
        if self.config.log.wandb:
            self.log_wandb(run, models, inferences)
        return 0
        
    def log_local(self, run, models=None, inferences=None):
        # -- 1. Log Data -- 
        if self.config.log.resume:
            df = pd.read_csv(os.path.join(self.path, "logs", "log.csv"))
            df_new = pd.DataFrame([run])
            df = pd.concat([df, df_new],ignore_index=True)
        else:
            df = pd.DataFrame([run])
            self.config.log.resume = True
        df.to_csv(os.path.join(os.path.join(self.path, "logs", "log.csv")), index=False) 
        
        # -- 2. Log Models -- 
        if models:
            for name, model in models.items():
                path = os.path.join(self.path, "models", name)
                torch.save(model, path)
        
        # -- 3. Log Inferences --  
        if inferences:
            for name, value in inferences.items():
                path = os.path.join(self.path, "inference", name)
                save_image(value, path)
        return 0  
    
    def log_wandb(self, run, models=None, inferences=None):
        wandb_run = run
        
        if models:
            for name, model in models.items():
                path = os.path.join(self.path, "models", name)
                wandb.log_model(path)
        
        if inferences:
            for name, value in inferences.items():
                value = value.cpu().numpy()
                value = value.transpose(1, 2, 0)
                wandb_run[name] = wandb.Image(value)
        wandb.log(**wandb_run)
        return 0
             
    
    def get_dataloader(self):
        return 
    
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
    
    def image_grid_tensor(self, batch, normalize=False):
        """
        Turn a batch into a Gird Image Tensor

        Args:
            batch (torch.Tensor):            Image Tensor of [B x C x H x W]
            normalize (bool, optional):     _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        grid_tensor = make_grid(batch, normalize=normalize)
        return grid_tensor    
        
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
    
        
        
    