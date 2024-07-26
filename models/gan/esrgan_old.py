import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid, save_image
from torch import nn
import glob
import os
import wandb
import torch

from accelerate import Accelerator
import pytorch_lightning as pl


from tqdm import tqdm

from ..base import Base

class ESRGAN(pl.LightningModule):
    """
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
    https://arxiv.org/abs/1809.00219
    """
    def __init__(self, config, dataset, networks):
        self.config = config
        
        # Networks
        self.netG = networks[0]
        self.netD = networks[1]
        self.netF = networks[2]
        
        # Loss
        self.cri_fea = nn.L1Loss()
        self.cri_pix = nn.L1Loss()
        self.cri_gan = nn.BCEWithLogitsLoss()
        
        # Optimizer
        optim.Adam(parameters, lr=self.config.model.optimizer.lr, weight_decay=self.config.model.optimizer.weight_decay,
                              betas=(self.config.model.optimizer.beta1, 0.999), amsgrad=self.config.model.optimizer.amsgrad)
        
        # Scheduler
    
    
    def configure_optimizers(self):
        optimizer_G = get_optimizer(self.netG.parameters())
        optimizer_D = self.get_optimizer(self.netD.parameters())
        
        
        
        
        return torch.optim.SGD(self.model.parameters(), lr=0.1)
        
    def training_step(self, batch):
        # -- 0. Create Batch and Labels -- 
        self.highres = batch["images"].to(self.config.device)
        self.lowres = batch["lowres"].to(self.config.device)
        
        # -- 1. Update Generator --
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
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
        
        real_labels = torch.empty_like(pred_d_real).fill_(self.real_label_val)
        fake_labels = torch.empty_like(pred_g_fake).fill_(self.fake_label_val)
    
        l_g_gan = self.config.model.loss.weight_gan * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), fake_labels) +
                                self.cri_gan(pred_g_fake - torch.mean(pred_d_real), real_labels)) / 2 
        l_g_total += l_g_gan
        l_g_total.backward()
        self.optimizer_G.step()
        self.scheduler_G.step()
        
        # 2. -- Update Discriminator --
        for p in self.netD.parameters():
            p.requires_grad = True
        
        self.optimizer_D.zero_grad()
        l_d_total = 0
        
        pred_d_real = self.netD(self.highres)
        pred_d_fake = self.netD(self.fake_H.detach())  
        
        l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), real_labels)
        l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), fake_labels)

        l_d_total = (l_d_real + l_d_fake) / 2
        l_d_total.backward()
        self.optimizer_D.step()
        
        
        
        
        
        return 0
        
        








class ESRGAN(Base):
    """
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
    https://arxiv.org/abs/1809.00219
    """
    def __init__(self, config, dataset, networks):
        super().__init__(config, dataset, networks)
        
        # Networks
        self.netG = networks[0].to(config.device)
        self.netD = networks[1].to(config.device)
        self.netF = networks[2].to(config.device)
        
        # Enable DataParallel on Multiple GPUs
        if torch.cuda.device_count() > 1:
            print("Running on", torch.cuda.device_count(), "GPUs")
            self.netG = nn.DataParallel(self.netG)
            self.netD = nn.DataParallel(self.netD)
            self.netF = nn.DataParallel(self.netF)
        
        # Metrics
        self.psnr = PeakSignalNoiseRatio().to(config.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(config.device)
        
        # Loss
        self.cri_fea = nn.L1Loss().to(config.device)
        self.cri_pix = nn.L1Loss().to(config.device)
        self.cri_gan = nn.BCEWithLogitsLoss().to(config.device)
        
        # Optimizer
        self.optimizer_G = self.get_optimizer(self.netG.parameters())
        self.optimizer_D = self.get_optimizer(self.netD.parameters())
        
        # Scheduler
        if config.model.scheduler.name:
            self.scheduler_G = self.get_scheduler(self.optimizer_G)
            self.scheduler_D = self.get_scheduler(self.optimizer_D)
        
        # Label Values
        self.real_label_val = 1.0
        self.fake_label_val = 0.0
        
        self.epoch = 0
        
        # Load Previous Model
        if self.config.resume.active:
            self.model_load(self.config.resume.path)       
    
    def log(self, step):
        if step == 0:
            # Inference
            highres_fake, lowres, images = self.inference()
            
            # Images 
            highres_fake_tensor = make_grid(highres_fake)
            lowres_tensor = make_grid(lowres)
            images_tensor = make_grid(images)
            
            save_image(highres_fake_tensor, os.path.join(self.path, "inference", f"{self.epoch}_highres_fake.png"))
            save_image(images_tensor, os.path.join(self.path, "inference", f"{self.epoch}_highres_real.png"))
            save_image(lowres_tensor, os.path.join(self.path, "inference", f"{self.epoch}_lowres.png"))
            
            self.run["Highres_Fake"] = wandb.Image(highres_fake_tensor)
            self.run["Highres_Real"] = wandb.Image(images_tensor)
            self.run["Lowres"] = wandb.Image(lowres_tensor)
            
            # Metrics
            self.run["SSIM"] = self.ssim(highres_fake, images).item()
            self.run["PSNR"] = self.psnr(highres_fake, images).item()
            
            # Model
            model_path = self.model_save()
            wandb.log_model(model_path)
            
            # Wandb Log
            wandb.log(self.run)
            
        else:
            wandb.log(self.run)
        return 0     
        
    
    def model_load(self, path):
        
        # Load Checkpoint
        checkpoint = torch.load(path)
        
        # Update Networks
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        
        # Update Optimiziers
        self.optimizer_G.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizerD_state_dict'])
        
        self.epoch = checkpoint['epoch']
        return 0
    
    def model_save(self):
        directory = os.path.join(self.path, "models", f"{self.epoch}_{self.config.log.project}_{self.config.log.id}.pth")
        torch.save(
                        {
                            'epoch': self.epoch,
                            'netG_state_dict' : self.netG.state_dict(),
                            'netD_state_dict' : self.netD.state_dict(),
                            'optimizerG_state_dict' : self.optimizer_G.state_dict(),
                            'optimizerD_state_dict' : self.optimizer_D.state_dict(),  
                        },
                        directory)
        return directory     
    
    def train(self):
        if self.config.resume.active:
            print(f"-- Resuming Training at Epoch {self.epoch} --")
        for epoch in tqdm(range(self.epoch, self.config.model.train.num_epochs), desc='Epochs'):
            for step, batch in enumerate(tqdm(self.dataloader, desc='Steps', leave=False)):
                # -- 0. Create Batch and Labels -- 
                self.highres = batch["images"].to(self.config.device)
                self.lowres = batch["lowres"].to(self.config.device)
                
                # -- 1. Update Generator --
                for p in self.netD.parameters():
                    p.requires_grad = False

                self.optimizer_G.zero_grad()
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
                
                real_labels = torch.empty_like(pred_d_real).fill_(self.real_label_val)
                fake_labels = torch.empty_like(pred_g_fake).fill_(self.fake_label_val)
            
                l_g_gan = self.config.model.loss.weight_gan * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), fake_labels) +
                                        self.cri_gan(pred_g_fake - torch.mean(pred_d_real), real_labels)) / 2 
                l_g_total += l_g_gan
                l_g_total.backward()
                self.optimizer_G.step()
                self.scheduler_G.step()
                
                # 2. -- Update Discriminator --
                for p in self.netD.parameters():
                    p.requires_grad = True
                
                self.optimizer_D.zero_grad()
                l_d_total = 0
                
                pred_d_real = self.netD(self.highres)
                pred_d_fake = self.netD(self.fake_H.detach())  
                
                l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), real_labels)
                l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), fake_labels)

                l_d_total = (l_d_real + l_d_fake) / 2
                l_d_total.backward()
                self.optimizer_D.step()
                
                # 3. -- Logging --
                self.run = {
                    "Epoch" : epoch,
                    "Generator Loss" : l_g_total.item(),
                    "Discriminator Loss" : l_d_total.item(),
                    "Pixel Loss" : l_g_pix.item(),
                    "Feature Loss" : l_g_fea.item(),
                    "GAN Loss" : l_g_gan.item(),
                    "Learning Rate" : self.optimizer_G.param_groups[0]["lr"],
                }
                
                self.log(step)    
        return 0
    
    @torch.no_grad()
    def inference(self):
        # -- 0. Batch -- 
        batch = next(iter(self.dataloader))
        lowres = batch["lowres"].to(self.device)
        images = batch["images"].to(self.device)
        
        # -- 1. Inference --
        highres_fake = self.netG(lowres)        
        return highres_fake, lowres, images
            
        