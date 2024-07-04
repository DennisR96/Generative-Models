from torch import nn, optim
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import wandb
import torchmetrics

class Base():
    """
    General Generative Model Class
    
    """
    def __init__(self, config):      
        self.config = config
        
        
        # Metrics
        self.fid = FrechetInceptionDistance()
        self.psnr = PeakSignalNoiseRatio()
        pass
    
    def get_dataloader(self):
        return 
    
    def get_device(self):
        """
        Setup Available Device
        """
        if torch.cuda.is_available():
            print("-- CUDA Device detected --")
            device = "cuda"
        elif torch.backends.mps.is_available():
            print("-- MPS Device detected --")
            device = "mps"
        else:
            print("-- CPU Device detected --")
            device = "cpu"
            
        if self.config.log == "wandb":
            wandb.log({"device": device})
            
        return device

    def get_optimizer(self, parameters):
        """
        Setup Model Optimizer

        Args:
            parameters (_type_):    PyTorchModel.parameters()
        """
        if self.config.network.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.network.lr, weight_decay=self.config.network.weight_decay,
                              betas=(self.config.network.beta1, 0.999), amsgrad=self.config.network.amsgrad)
        elif self.config.network.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.network.lr, weight_decay=self.config.network.weight_decay)
        elif self.config.network.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.network.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.network.optimizer))
    
    def log_model(self, model, path):
        '''
        Save PyTorch Model as .pth
        '''
        torch.save(model.state_dict(), path)
        
        if self.config.log == "wandb":
            wandb.log_model(path)
        return 0
    
    def log_metric_fid(self, batch_true, batch_false):
        self.fid.update(batch_true, real=True)
        self.fid.update(batch_false, real=False)
        metric_fid = self.fid.compute()
        
        if self.config.log == "wandb":
            wandb.log({"fid" : metric_fid})
        return 0
    
    def log_metric_psnr(self, batch_pred, batch_true):
        metric_psnr = self.psnr(batch_pred, batch_true)
        
        if self.config.log == "wandb":
            wandb.log({"PSNR" : metric_psnr})
        return 0 
    
    def log_metric_():
        return 0
    
    def log_image_grid(self, images, epoch):
        grid = make_grid(images, padding=2, normalize=True)
        
        if self.config.log == "wandb":
             wandb.log({"epoch": epoch, "image": wandb.Image(grid)})   
        #save_image(grid, f"results/{epoch}_{self.config.dataset.name}.png")
        return grid
        
    def get_scheduler(self):
        return 0 
        
    def model_update_parameters(self):
        return 0 
    
    def model_load(self, path):
        self.model.load_state_dict(torch.load(path))
        return 0
        
    def model_save(self, model, path):
        """
        Save PyTorch Model as .pth
        """
        torch.save(model.state_dict(), path)
        return 0
    
    def train(self):
        for epoch in tqdm(range(self.config.train.num_epochs), desc='Epochs'):
            for step, batch in enumerate(tqdm(self.dataloader, desc='Steps', leave=False)):
                self.model_update_parameters(batch)
        return 0
    
    def inference(self):
        return 0
    
        
        
    