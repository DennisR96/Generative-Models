import lightning as L
from torch import nn
import torch


class ESRGAN(L.LightningModule):
    """
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
    https://arxiv.org/abs/1809.00219
    """
    def __init__(self, config, networks):
        super().__init__()
        self.config = config
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
    
    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.netG.parameters(), 
                                    lr=self.config.model.optimizer.lr, 
                                    weight_decay=self.config.model.optimizer.weight_decay,
                              betas=(self.config.model.optimizer.beta1, 0.999), 
                              amsgrad=self.config.model.optimizer.amsgrad)
        
        optimizer_D = torch.optim.Adam(self.netD.parameters(), 
                                    lr=self.config.model.optimizer.lr, 
                                    weight_decay=self.config.model.optimizer.weight_decay,
                              betas=(self.config.model.optimizer.beta1, 0.999), 
                              amsgrad=self.config.model.optimizer.amsgrad)
        return optimizer_G, optimizer_D
    
    def training_step(self, batch, batch_idx):
        optimizer_G, optimizer_D = self.optimizers()
        
        # -- 0. Create Batch and Labels -- 
        highres = batch["images"]
        lowres = batch["lowres"]
        
        real_labels = torch.ones((highres.shape[0], 1), device=self.device)
        fake_labels = torch.zeros((highres.shape[0], 1), device=self.device)
                
        # -- 1. Update Generator --
        for p in self.netD.parameters():
            p.requires_grad = False

        optimizer_G.zero_grad()
        self.fake_H = self.netG(lowres)
                
        l_g_total = 0
        
        # -- 1.1 Pixel Loss --
        l_g_pix = self.config.model.loss.weight_pixel * self.cri_pix(self.fake_H, highres)
        l_g_total += l_g_pix
        
        # -- 1.2 Feature Loss --
        real_fea = self.netF(highres).detach()
        fake_fea = self.netF(self.fake_H)
        l_g_fea = self.config.model.loss.weight_feature * self.cri_fea(fake_fea, real_fea)
        l_g_total += l_g_fea
        
        # -- 1.3 GAN Loss -- 
        pred_g_fake = self.netD(self.fake_H)
        pred_d_real = self.netD(highres).detach()
    
        l_g_gan = self.config.model.loss.weight_gan * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), fake_labels) +
                                self.cri_gan(pred_g_fake - torch.mean(pred_d_real), real_labels)) / 2 
        l_g_total += l_g_gan
        self.manual_backward(l_g_total)
        optimizer_G.step()
        
        # 2. -- Update Discriminator --
        for p in self.netD.parameters():
            p.requires_grad = True
        
        optimizer_D.zero_grad()
        l_d_total = 0
        
        pred_d_real = self.netD(highres)
        pred_d_fake = self.netD(self.fake_H.detach())  
        
        l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), real_labels)
        l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), fake_labels)

        l_d_total = (l_d_real + l_d_fake) / 2
        self.manual_backward(l_d_total)
        optimizer_D.step()
        
        self.log_dict(
            {"Generator Loss": l_g_total, 
             "Discriminator Loss": l_d_total, 
             "Pixel Loss" : l_g_pix,
             "Feature Loss" : l_g_fea,
             "GAN Loss" : l_g_gan
             }, 
            prog_bar=False)

            
        
        
        
        
        
    