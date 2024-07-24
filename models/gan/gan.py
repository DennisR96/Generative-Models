"""
Model Implementations
Generative Adversarial Networks:
- Vanilla GAN
- ESRGAN
"""
from ..base import Base
from torch import nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import wandb
from torchvision.models import vgg19
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance


class GAN(Base):
    def __init__(self, config, dataset, network):
        super(GAN, self).__init__(config)  # Call the Base class initializer

        self.config = config
        self.device = self.get_device()
        
        # Dataloader
        self.dataloader = DataLoader(dataset, batch_size=self.config.model.dataloader.batch_size, shuffle=True, num_workers=self.config.model.dataloader.num_workers)
        self.criterion = nn.BCELoss()
        
        # FID
        self.fid = FrechetInceptionDistance(feature=64, normalize=True)
        
        # Discriminator and Generator
        self.netG = network[0].to(self.device)
        self.netD = network[1].to(self.device)
        
        # Resume State
        if self.config.model.resume.active:
            print("–– Loading Model –– ")
            self.netG.load_state_dict(torch.load(self.config.model.resume.generator))
            self.netD.load_state_dict(torch.load(self.config.model.resume.discriminator))
        
        
        wandb.watch(self.netG)
        wandb.watch(self.netD)
        
        # Optimizer
        self.optimizerG = self.get_optimizer(self.netG.parameters())
        self.optimizerD = self.get_optimizer(self.netD.parameters())
        
    def train(self):
        for epoch in tqdm(range(self.config.model.train.num_epochs), desc='Epochs'):
            for step, batch in enumerate(tqdm(self.dataloader, desc='Steps', leave=False)):
                self.model_update_parameters(batch)
                
            # Save Models
            self.model_save(self.netG, f"results/{epoch}_Generator.pth")
            self.model_save(self.netD, f"results/{epoch}_Discriminator.pth")
            
            # Save Inference            
            images = self.inference() 
            self.log_image_grid(images, epoch) 
            
            # Metric: FID
            if self.config.dataset.channels == 3:
                self.fid.update(self.real, real=True)
                self.fid.update(self.fake, real=False)
                wandb.log({"FID" : self.fid.compute()})
                 
        return 0
        
    def model_update_parameters(self, batch):
        # 0. Pre-Configuration
        ## Format Batch
        images = batch["images"].to(self.device)
        batch_size = images.size(0)
        
        ## Real and Fake Label Tensor
        Real_Labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=self.device)
        Fake_Labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=self.device)
        
        # 1. Train Discriminator
        ## Real Images D_x
        self.netD.zero_grad()
        output = self.netD(images).view(-1)
        errD_real = self.criterion(output, Real_Labels)
        errD_real.backward()
        
        ## Fake Images
        noise = torch.randn(batch_size, self.config.network.generator.in_channels, 1, 1, device=self.device)
        fake = self.netG(noise)
        output = self.netD(fake.detach()).view(-1)
        errD_fake = self.criterion(output, Fake_Labels)
        errD_fake.backward()
        
        ## Compute Discriminator Loss and Update
        errD = errD_real + errD_fake
        self.optimizerD.step()
        
        # 2. Train Generator
        ## Create Fake Images
        self.netG.zero_grad()
        output = self.netD(fake).view(-1)
        errG = self.criterion(output, Real_Labels)
        errG.backward()
        self.optimizerG.step() 
        
        # Logging
        self.log_loss(errD = errD.item(), 
                      errG = errG.item())
        
        return 0
    
    @torch.no_grad()
    def inference(self):
        self.real = next(iter(self.dataloader))["images"]
        fixed_noise = torch.randn(self.config.model.dataloader.batch_size, self.config.network.generator.in_channels, 1, 1, device=self.device)
        self.fake = self.netG(fixed_noise).detach().cpu()
        return self.fake
    
class ESRGAN(GAN):
    """
    Enhanced Super-Resolution GAN
    https://arxiv.org/abs/1809.00219
    """
    def __init__(self, config, dataset, model):
        super().__init__(config, dataset, model)
        
        # Loss Functions
        self.criterion_l1 = nn.L1Loss()
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)
        
        # Perceptual
        self.vgg = vgg19(weights='IMAGENET1K_V1').to(self.device)
        self.vgg19_54 = nn.Sequential(*list(self.vgg.features.children())[:35])
        
    def criterion_perceptual(self, real, fake):
        real_feat = self.vgg19_54(real)
        fake_feat = self.vgg19_54(fake.detach())
    
        loss = self.criterion_l1(real_feat, fake_feat)
        return loss
    
    def train(self):
        for epoch in tqdm(range(self.config.model.train.num_epochs), desc='Epochs'):
            for step, batch in enumerate(tqdm(self.dataloader, desc='Steps', leave=False)):
                self.model_update_parameters(batch)
            
            # Metrics
            wandb.log({"SSIM" : self.ssim(self.highres_fake, self.images).item(), 
                       "PSNR" : self.psnr(self.highres_fake, self.images).item()
                       })
            # print(f"SSIM: {self.ssim(self.highres_fake, self.images).item()}")
            # print(f"PSNR: {self.psnr(self.highres_fake, self.images).item()}")
            
            # Save Models
            self.model_save(self.netG, f"results/{epoch}_Generator.pth")
            self.model_save(self.netD, f"results/{epoch}_Discriminator.pth")
            
            # Save Inference
            grid = self.inference()       
        return 0
        
    def model_update_parameters(self, batch):
        # 0. Pre-Configuration
        
        ## Format Batch
        self.images = batch["images"].to(self.device)
        self.lowres = batch["lowres"].to(self.device)
        batch_size = self.images.size(0)
        
        ## Real and Fake Label Tensor
        Real_Labels = torch.full((batch_size,1), 1, dtype=torch.float, device=self.device)
        Fake_Labels = torch.full((batch_size,1,), 0, dtype=torch.float, device=self.device)
        
        # 1. Train Generator
        self.optimizerG.zero_grad()
        self.highres_fake = self.netG(self.lowres)
        
        ## 1.1 Pixel Loss (L1)
        loss_pixel = self.criterion_l1(self.highres_fake, self.images)

        ## 2.2 Perceptual Loss (L1 (VGG19))
        loss_content = self.criterion_perceptual(self.highres_fake, self.images)

        ## 2.3 Adversarial Loss (Relativistic average GAN)
        pred_real = self.netD(self.images).detach()
        pred_fake = self.netD(self.highres_fake)
        loss_GAN = self.criterion(pred_fake - pred_real.mean(0, keepdim=True), Real_Labels)

        ## 2.4 Complete Loss
        ## add content loss back + loss_content
        L_G = loss_content + self.config.model.loss.hyper_lambda * loss_GAN + self.config.model.loss.hyper_eta * loss_pixel
        L_G.backward() 
        self.optimizerG.step()  
        
        # 2. Train Discriminator
        self.optimizerD.zero_grad()

        pred_real = self.netD(self.images)
        pred_fake = self.netD(self.highres_fake.detach())

        loss_real = self.criterion(pred_real - pred_fake.mean(0, keepdim=True), Real_Labels)
        loss_fake = self.criterion(pred_fake - pred_real.mean(0, keepdim=True), Fake_Labels)

        L_D = (loss_real + loss_fake) / 2
        L_D.backward()
        
        self.log_loss(L_G = L_G.item(), 
                      L_D = L_D.item(),
                      Pixel = loss_pixel.item(),
                      Content = loss_content.item(),
                      GAN = loss_GAN.item())
        self.optimizerD.step()
    
    @torch.no_grad()
    def inference(self):
        batch = next(iter(self.dataloader))
        
        lowres = batch["lowres"].to(self.device)
        images = batch["images"].to(self.device)
        highres_fake = self.netG(lowres)
        self.log_image_grid(images, "High Resolution – True")
        self.log_image_grid(highres_fake, "High Resolution – Fake")
        self.log_image_grid(lowres, "Low Resolutiin")
        return 0
