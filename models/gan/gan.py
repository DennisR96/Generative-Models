from typing import Any
import lightning as L
from torch import nn
import torch
from torchvision.utils import make_grid
from utils.utils import namespace2dict

class GAN(L.LightningModule):
    def __init__(self, config, networks):
        super().__init__()
        
        # Configuration
        self.config = config
        
        # Loss Function
        self.criterion = nn.BCELoss()
        
        # Networks
        self.netG = networks[0]
        self.netD = networks[1]
        
        # Manual Optimization
        self.automatic_optimization = False
        
    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.netG.parameters(), 
                                       lr=self.config.model.optimizer.lr, 
                                       betas=(self.config.model.optimizer.beta1, 
                                              self.config.model.optimizer.beta1))
        optimizer_D = torch.optim.Adam(self.netD.parameters(), 
                                       lr=self.config.model.optimizer.lr, 
                                       betas=(self.config.model.optimizer.beta1, 
                                              self.config.model.optimizer.beta1))
        
        if self.config.model.scheduler.name is not None:
            scheduler_G = {
                "scheduler" : torch.optim.lr_scheduler.MultiStepLR(
                    optimizer_G, 
                    self.config.model.scheduler.milestones),
                "name" : "scheduler_G"
            }
            
            scheduler_D = {
                "scheduler" : torch.optim.lr_scheduler.MultiStepLR(
                    optimizer_D, 
                    self.config.model.scheduler.milestones),
                "name" : "scheduler_D"
                }
            return [optimizer_G, optimizer_D], [scheduler_G, scheduler_D]
        
        return [optimizer_G, optimizer_D]
    
    def on_train_epoch_end(self):        
        grid = make_grid(self.images_gen)
        for logger in self.loggers:
            if isinstance(logger, L.pytorch.loggers.wandb.WandbLogger):
                logger.log_image("Generated Images", [grid])
            elif isinstance(logger, L.pytorch.loggers.tensorboard.TensorBoardLogger):
                logger.experiment.add_image("Generated Images", grid, self.current_epoch)
        return 0
    
    def on_train_start(self):
        for logger in self.loggers:
            logger.log_hyperparams(namespace2dict(self.config))
            
            if isinstance(logger, L.pytorch.loggers.wandb.WandbLogger):
                logger.watch(self.netG)
                logger.watch(self.netD)
        return 0
    
    def training_step(self, batch, batch_idx):
        # -- 1. Pre-Configuration -- 
        optimizer_G, optimizer_D = self.optimizers()

        if self.config.model.scheduler.name is not None:
            scheduler_G, scheduler_D = self.lr_schedulers()
        
        self.images = batch["images"]
        batch_size = self.images.shape[0]
        
        real_labels = torch.ones((batch_size, 1), device=self.device)
        fake_labels = torch.zeros((batch_size, 1), device=self.device)
        
        # -- 2. Sample Noise -- 
        z = torch.randn(batch_size, self.config.network.generator.in_channels, 1, 1, device=self.device)
        
        # -- 3. Update Generator --
        self.toggle_optimizer(optimizer_G)
        self.images_gen = self.netG(z)
        g_loss = self.criterion(self.netD(self.images_gen), real_labels)
        optimizer_G.zero_grad()
        self.manual_backward(g_loss)
        optimizer_G.step()
        self.untoggle_optimizer(optimizer_G)
        
        # -- 4. Update Discriminator --
        self.toggle_optimizer(optimizer_D)
        real_loss = self.criterion(self.netD(self.images), real_labels)
        fake_loss = self.criterion(self.netD(self.netG(z).detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        optimizer_D.zero_grad()
        self.manual_backward(d_loss)
        optimizer_D.step()
        self.untoggle_optimizer(optimizer_D)
        
        if self.config.model.scheduler.name is not None:
            scheduler_G.step()
            scheduler_D.step()
        
        # -- 5. Logging --
        self.log("Generator Loss", g_loss, prog_bar=True)
        self.log("Discriminator Loss", d_loss,  prog_bar=True)
        

        
        
        