import os, yaml
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import torchvision
from torchvision.transforms import ToPILImage
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from network.StarSRGAN.discriminator import AttentionUNetDiscriminator
from network.StarSRGAN.generator import StarSRNet
from dataset.dataset import Dataset
from utils.utils import dict2namespace
from model.loss import Pixel_Loss, Perceptual_Loss

# Load Config YAML as Namespace 
config_path = "config/config.yaml"

with open(os.path.join(config_path), "r") as f:
    config_yaml = yaml.safe_load(f)

config = dict2namespace(config_yaml)

# Check for Available Devices
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
# Initialize Models
model_D = AttentionUNetDiscriminator(3, 64).to(device)
model_G = StarSRNet().to(device)

optimizer_D = torch.optim.AdamW(model_D.parameters())
optimizer_G = torch.optim.AdamW(model_G.parameters())

# Dataset
DS = Dataset(config)
Train_DataLoader = DataLoader(DS, batch_size=config.train.batch_size, shuffle=True)

# Loss 
criterion_pixel = Pixel_Loss().to(device)
criterion_content = Perceptual_Loss().to(device)
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)

# Metrics 
psnr = PeakSignalNoiseRatio().to(device)
ssim = StructuralSimilarityIndexMeasure().to(device)

# Training Loop
for epoch in range(config.train.epochs):
    print(f"Epoch {epoch}")
    for step, batch in enumerate(Train_DataLoader):
        print(f"Step {step}")
        
        # 1. Prepreation
        
        ## 1.1 Split Batch and move to Device 
        low_resolution = batch["LR"].to(device)
        high_resolution = batch["HR"].to(device)
        
        ## 1.2 Adversarial Loss
        Valid = torch.ones((low_resolution.size(0), *(1, 256, 256)), requires_grad=False).to(device)
        Fake = torch.zeros((low_resolution.size(0), *(1, 256, 256)), requires_grad=False).to(device)
        
        # 2. Train Generator
        optimizer_G.zero_grad()
        high_resolution_fake = model_G(low_resolution)
        
        ## 2.1 Pixel Loss (L1)
        loss_pixel = criterion_pixel(high_resolution_fake, high_resolution)
        
        ## 2.2 Perceptual Loss (L1 (VGG19))
        loss_content = criterion_content(high_resolution_fake, high_resolution)
        
        ## 2.3 Adversarial Loss 
        pred_real = model_D(high_resolution).detach()
        pred_fake = model_D(high_resolution_fake)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), Valid)
        
        ## 2.4 Complete Loss
        L_G = loss_content + config.train.hyper_lambda * loss_GAN + config.train.hyper_eta * loss_pixel
        
        L_G.backward() 
        optimizer_G.step() 
        print(f"Generator Loss {L_G.item()}")  
        
        # 3. Train Discriminator
        optimizer_D.zero_grad()
        
        pred_real = model_D(high_resolution)
        pred_fake = model_D(high_resolution_fake.detach())
        
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), Valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), Fake)
        
        L_D = (loss_real + loss_fake) / 2
        L_D.backward()
        optimizer_D.step()
        print(f"Discriminator Loss {L_D.item()}")  
        
        # 4. Metrics
        print(f"SSIM: {ssim(high_resolution_fake, high_resolution).item()}")
        print(f"PSNR: {psnr(high_resolution_fake, high_resolution).item()}")
        
        # 5. Inference  
        combined_tensor = torch.cat((high_resolution[0:8], high_resolution_fake[0:8]), dim=0)
        grid = torchvision.utils.make_grid(combined_tensor, nrow=8, padding=2, normalize=True,value_range=(-1, 1))
        to_pil = ToPILImage()
        grid_image = to_pil(grid)
        grid_image.save(f'results/Epoch_{epoch}_Step{step}.png')


        


    
    
    
    
    


