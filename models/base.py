from torch import nn, optim
import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

class Base():
    """
    Base Generative Model Class
    
    """
    def __init__(self, config):      
        self.config = config
        pass
    
    def get_device(self):
        """
        Check Available Device
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
        return device
    
    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))
    
    def image_grid(self, images, epoch):
            grid = make_grid(images, padding=2, normalize=True)
            save_image(grid, f"results/{epoch}_{self.config.dataset.name}.png")
            return grid
        
    def metric_fid(self, batch_true, batch_false):
        self.fid.update(batch_true, real=True)
        self.fid.update(batch_false, real=False)
        self.fid.compute()
        
    def model_update_parameters(self):
        return 0 
    
    def model_load(self):
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
    
        
        
    