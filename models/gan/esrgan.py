import lightning as L
from torch import nn
import torch
from utils.utils import namespace2dict
from torchvision.utils import make_grid
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class ESRGAN(L.LightningModule):
    """
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
    https://arxiv.org/abs/1809.00219
    """
    def __init__(self, config, networks):
        super().__init__()
        self.config = config
        self.save_hyperparameters(namespace2dict(config))
        self.automatic_optimization = False
        
        # Networks
        self.netG = networks[0]
        self.netD = networks[1]
        self.netF = networks[2]
        
        # Loss
        self.cri_fea = nn.L1Loss()
        self.cri_pix = nn.L1Loss()
        self.cri_gan = nn.BCEWithLogitsLoss()
        
        # Labels
        self.real_label_val = 1.0
        self.fake_label_val = 0.0
        
        # Metrics
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()
        
    
    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.netG.parameters(), 
                            lr=self.config.model.optimizer.lr, 
                            weight_decay=self.config.model.optimizer.weight_decay,
                            betas=(self.config.model.optimizer.beta1, 
                                     self.config.model.optimizer.beta2), 
                            amsgrad=self.config.model.optimizer.amsgrad)
        
        optimizer_D = torch.optim.Adam(self.netD.parameters(), 
                            lr=self.config.model.optimizer.lr, 
                            weight_decay=self.config.model.optimizer.weight_decay,
                            betas=(self.config.model.optimizer.beta1, 0.999), 
                            amsgrad=self.config.model.optimizer.amsgrad)
        
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
        for logger in self.loggers:
            logger.experiment.add_image("lowres", make_grid(self.lowres), self.current_epoch)
            logger.experiment.add_image("highres", make_grid(self.highres), self.current_epoch)
            logger.experiment.add_image("highres_fake", make_grid(self.fake_H), self.current_epoch)
            logger.log_metrics(
                {
                    "SSIM" : self.ssim(self.fake_H, self.highres).item(), 
                    "PSNR" : self.psnr(self.fake_H, self.highres).item()}, 
                self.current_epoch)  
        return 0
    
    def training_step(self, batch, batch_idx):
        # -- 0. Create Batch and Labels -- 
        optimizer_G, optimizer_D = self.optimizers()
        if self.config.model.scheduler.name is not None:
            scheduler_G, scheduler_D = self.lr_schedulers()
        self.highres = batch["images"]
        self.lowres = batch["lowres"]
        batch_size = self.highres.shape[0]
        
        real_labels = torch.ones((batch_size, 1), device=self.device)
        fake_labels = torch.zeros((batch_size, 1), device=self.device)
                
        # -- 1. Update Generator --
        self.toggle_optimizer(optimizer_G)
        optimizer_G.zero_grad()
        self.fake_H = self.netG(self.lowres)   
        l_g_total = 0
        
        # -- 1.1 Pixel Loss --
        l_g_pix = self.config.model.loss.weight_pixel * self.cri_pix(self.fake_H, self.highres)
        l_g_total += l_g_pix
        
        # -- 1.2 Feature Loss --
        real_fea = self.netF(self.highres).detach()
        fake_fea = self.netF(self.fake_H)
        l_g_fea = self.config.model.loss.weight_feature * self.cri_fea(fake_fea, real_fea)
        l_g_total += l_g_fea
        
        # -- 1.3 GAN Loss -- 
        pred_g_fake = self.netD(self.fake_H)
        pred_d_real = self.netD(self.highres).detach()
        l_g_gan = self.config.model.loss.weight_gan * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), fake_labels) +
                                self.cri_gan(pred_g_fake - torch.mean(pred_d_real), real_labels)) / 2 
        l_g_total += l_g_gan
        self.manual_backward(l_g_total)
        optimizer_G.step()
        self.untoggle_optimizer(optimizer_G)
        
        # -- 2. Update Discriminator --
        self.toggle_optimizer(optimizer_D)
        optimizer_D.zero_grad()
        l_d_total = 0
        pred_d_real = self.netD(self.highres)
        pred_d_fake = self.netD(self.fake_H.detach())  
        l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), real_labels)
        l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), fake_labels)
        l_d_total = (l_d_real + l_d_fake) / 2
        self.manual_backward(l_d_total)
        optimizer_D.step()
        self.untoggle_optimizer(optimizer_D)
        
        if self.config.model.scheduler.name is not None:
            scheduler_G.step()
            scheduler_D.step()
        
        # 3. -- Logging -- 
        self.log("Generator Loss", l_g_total)
        self.log("Discriminator Loss", l_d_total)
        self.log("Pixel Loss", l_g_pix)
        self.log("Feature Loss", l_g_fea)
        self.log("GAN Loss", l_g_gan)

            
        
        
        
        
        
    