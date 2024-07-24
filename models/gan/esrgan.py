import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..base import Base

class ESRGAN(Base):
    """
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
    https://arxiv.org/abs/1809.00219
    """
    def __init__(self, config, dataset, networks):
        super().__init__(config)
        self.config = config
        
        # Dataset
        self.dataloader = DataLoader(dataset, 
                                     batch_size=self.config.model.dataloader.batch_size, 
                                     shuffle=True, 
                                     num_workers=self.config.model.dataloader.num_workers)
        
        # Networks
        self.netG = networks[0].to(config.device)
        self.netD = networks[1].to(config.device)
        self.netF = networks[2].to(config.device)
        
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
        
    
    def train(self):
        self.step_count = 0 
        for epoch in tqdm(range(self.config.model.train.num_epochs), desc='Epochs'):
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
                    "Generator Loss" : l_g_total.item(),
                    "Discriminator Loss" : l_d_total.item(),
                    "Pixel Loss" : l_g_pix.item(),
                    "Feature Loss" : l_g_fea.item(),
                    "GAN Loss" : l_g_gan.item(),
                    "Learning Rate" : self.optimizer_G.param_groups[0]["lr"],
                }
                
                if step % 100 == 0:
                    # Inference
                    highres_fake, lowres, images = self.inference()
                    
                    highres_fake_tensor = self.image_grid_tensor(highres_fake)
                    lowres_tensor = self.image_grid_tensor(lowres)
                    images_tensor = self.image_grid_tensor(images)
                    
                    self.inferences = {
                            f"{epoch}_{step}_highres_real.png" : images_tensor,
                            f"{epoch}_{step}_highres_fake.png" : highres_fake_tensor,
                            f"{epoch}_{step}_lowres.png" : lowres_tensor,
                    }
                    
                    # Metrics
                    self.run["SSIM"] = self.ssim(highres_fake, images).item()
                    self.run["PSNR"] = self.psnr(highres_fake, images).item()
                    
                    # Models
                    self.models = {
                        f"{epoch}_{step}_Generator.pth" : self.netG,
                        f"{epoch}_{step}_Discriminator.pth" : self.netD,
                    }
                    self.log(self.run, self.models, self.inferences)                
                else:
                    self.log(self.run)
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