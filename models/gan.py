"""
Model Implementations
Generative Adversarial Networks:
- Vanilla GAN
- ESRGAN
"""
from .base import Base
from torch import nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torchvision.models import vgg19
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
import torch.nn.functional as F


class GAN(Base):
    def __init__(self, config, dataset, netD, netG):
        super(GAN, self).__init__(*args)  # Call the Base class initializer

        self.config = config
        self.device = self.get_device()
        
        # Dataloader
        self.dataloader = DataLoader(dataset, batch_size=self.config.train.batch_size, shuffle=True)

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Discriminator and Generator
        self.netG = netG.to(self.device)
        self.netD = netD.to(self.device)
        
        # Optimizer
        self.optimizerG = self.get_optimizer(self.netG.parameters())
        self.optimizerD = self.get_optimizer(self.netD.parameters())
        
    def criterion(self, input, target_is_real, gan_type, is_disc=False, 
             real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        
        """ 
        Compute GAN Loss
        
        Args:
            input (torch.Tensor)        Network Prediction
            target_is_real (bool)       Whether the target is Real or Fake
            gan_type (str)              Type of GAN Loss
            is_disc (bool)              Is Discriminator Loss
            real_label_val (float):     Value for real label.   Default: 1.0.
            fake_label_val (float):     Value for fake label.   Default: 0.0.
            loss_weight (float): Loss weight for generator. Default: 1.0.
        
        Returns: 
            Tensor: Computed GAN Loss
        """
        # Vanilla – Binary Cross Entropy with Logits Loss.
        if self.config.train.loss_gan == "vanilla":
            loss_fn = nn.BCEWithLogitsLoss()
            target_label = torch.full_like(input, real_label_val if target_is_real else fake_label_val)
            loss = loss_fn(input, target_label)
            
        # LSGAN – Mean Squared Error Loss (MSELoss)
        elif gan_type == "lsgan":
            loss_fn = nn.MSELoss()
            target_label = torch.full_like(input, real_label_val if target_is_real else fake_label_val)
            loss = loss_fn(input, target_label)
        
        # WGAN – Wasserstein GAN Loss
        elif gan_type == "wgan":
            loss = -input.mean() if target_is_real else input.mean()
            
        elif gan_type == "wgan_softplus":
            loss = F.softplus(-input).mean() if target_is_real else F.softplus(input).mean()
        
        # Hinge GAN Loss
        elif gan_type == "hinge":
            if is_disc:
                input = -input if target_is_real else input
                loss = F.relu(1 + input).mean()
            else:
                loss = -input.mean()
        return loss if is_disc else loss * loss_weight
        
    def train(self):
        for epoch in tqdm(range(self.config.train.num_epochs), desc='Epochs'):
            for step, batch in enumerate(tqdm(self.dataloader, desc='Steps', leave=False)):
                self.model_update_parameters(batch)
            
            # Save Models
            self.model_save(self.netG, f"results/{epoch}_Generator.pth")
            self.model_save(self.netD, f"results/{epoch}_Discriminator.pth")
            
            # Save Inference
            images = self.inference() 
            self.image_grid(images, epoch)        
        return 0
        
    def model_update_parameters(self, batch):
        # 0. Pre-Configuration
        ## Format Batch
        images = batch["images"].to(self.device)
        labels = batch["labels"].to(self.device)
        batch_size = images.size(0)
        
        ## Real and Fake Label Tensor
        #Real_Labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=self.device)
        #Fake_Labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=self.device)
        
        # 1. Train Discriminator
        ## Real Images D_x
        self.netD.zero_grad()
        output = self.netD(images).view(-1)
        errD_real = self.criterion(output, True, config.gan)
        errD_real.backward()
        #D_x = output.mean().item()
        
        ## Fake Images
        noise = torch.randn(batch_size, self.config.network.latent_vector, 1, 1, device=self.device)
        fake = self.netG(noise)
        output = self.netD(fake.detach()).view(-1)
        errD_fake = self.criterion(output, Fake_Labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        ## Compute Discriminator Loss and Update
        errD = errD_real + errD_fake
        self.optimizerD.step()
        
        # 2. Train Generator
        ## Create Fake Images
        self.netG.zero_grad()
        output = self.netD(fake).view(-1)
        errG = self.criterion(output, Real_Labels)
        errG.backward()
        
        ## Calculate Loss
        D_G_z2 = output.mean().item()
        self.optimizerG.step() 
        return 0
    
    @torch.no_grad()
    def inference(self):
        fixed_noise = torch.randn(self.config.train.batch_size, self.config.network.latent_vector, 1, 1, device=self.device)
        fake = self.netG(fixed_noise).detach().cpu()
        return fake
    
class ESRGAN(GAN):
    """
    Enhanced Super-Resolution GAN
    https://arxiv.org/abs/1809.00219
    """
    def __init__(self, config, dataset, netD, netG):
        super().__init__(config, dataset, netD, netG)
        
        # Loss Functions
        self.criterion_l1 = nn.L1Loss()
        
        # Metrics
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)
        
        # Perceptual
        self.vgg = vgg19(weights='IMAGENET1K_V1')
        self.vgg19_54 = nn.Sequential(*list(self.vgg.features.children())[:35])
        
    def criterion_perceptual(self, real, fake):
        real_feat = self.vgg19_54(real)
        fake_feat = self.vgg19_54(fake.detach())
    
        loss = self.l1(real_feat, fake_feat)
        return loss
    
    def train(self):
        for epoch in tqdm(range(self.config.train.num_epochs), desc='Epochs'):
            for step, batch in enumerate(tqdm(self.dataloader, desc='Steps', leave=False)):
                self.model_update_parameters(batch)
            
            # Metrics
            print(f"SSIM: {self.ssim(self.highres_fake, self.images).item()}")
            print(f"PSNR: {self.psnr(self.highres_fake, self.images).item()}")
            
            # Save Models
            self.model_save(self.netG, f"results/{epoch}_Generator.pth")
            self.model_save(self.netD, f"results/{epoch}_Discriminator.pth")
            
            # Save Inference
            grid = self.inference() 
            self.image_grid(grid, epoch)        
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

        ## 2.3 Adversarial Loss 
        pred_real = self.netD(self.images).detach()
        pred_fake = self.netD(self.highres_fake)
        loss_GAN = self.criterion(pred_fake - pred_real.mean(0, keepdim=True), Real_Labels)

        ## 2.4 Complete Loss
        ## add content loss back + loss_content
        L_G =  self.config.train.hyper_lambda * loss_GAN + self.config.train.hyper_eta * loss_pixel
        L_G = loss_GAN
        L_G.backward() 
        self.optimizerG.step() 
        #print(f"Generator Loss {L_G.item()}")  
        
        # 2. Train Discriminator
        self.optimizerD.zero_grad()

        pred_real = self.netD(self.images)
        pred_fake = self.netD(self.highres_fake.detach())

        loss_real = self.criterion(pred_real - pred_fake.mean(0, keepdim=True), Real_Labels)
        loss_fake = self.criterion(pred_fake - pred_real.mean(0, keepdim=True), Fake_Labels)

        L_D = (loss_real + loss_fake) / 2
        L_D.backward()
        self.optimizerD.step()
    
    @torch.no_grad()
    def inference(self):
        self.highres_fake = self.netG(self.lowres)
        combined_tensor = torch.cat((self.highres_fake[0:8], self.lowres[0:8]), dim=0)
        return combined_tensor
