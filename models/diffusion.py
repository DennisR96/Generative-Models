from .base import Base
import torch
from tqdm import tqdm
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from PIL import Image
import wandb
from torchvision import transforms
from torchvision.utils import make_grid, save_image

class DDPM(Base):
    """
    DDPM – Denoising Diffusion Probabilistic Model
    """
    def __init__(self, config, model, dataset):
        super(DDPM, self).__init__(config)
        
       
        self.timesteps = 100
        
        # Model and Dataset
        self.device = self.get_device()
        self.model = model.to(self.device)
        self.optimizer = self.get_optimizer(self.model.parameters())
        #self.dataloader = DataLoader(dataset, batch_size=self.config.train.batch_size, shuffle=True)

        # Beta Schedule
       #self.betas = linear_beta_schedule(timesteps=self.timesteps)
        beta_start = 0.0001
        beta_end  = 0.02
        self.betas = torch.linspace(beta_start, beta_end, self.timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)   
    
    def extract(self, a, t, x_shape):
        """
        Allows to extract the appropriate t index for a batch of indices.
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def train(self):
        noise = None
        for epoch in tqdm(range(self.config.train.num_epochs), desc='sampling loop time step', total=self.config.train.num_epochs):
            loss_list = []
            for step, batch in enumerate(tqdm(self.dataloader)):
                self.optimizer.zero_grad()
            
                batch_size = batch["images"].shape[0]
                x_0 = batch["images"].to(self.device)
                noise = torch.randn_like(x_0)
                    
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                
                self.sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_0.shape)
                self.sqrt_one_minus_alphas_cumprod_t = self.extract(
                    self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
                x_t = self.sqrt_alphas_cumprod_t * x_0 + self.sqrt_one_minus_alphas_cumprod_t * noise
                predicted_noise = self.model(x_t, t)
                
                loss = F.l1_loss(noise, predicted_noise)
                loss.backward()
                self.optimizer.step()
                wandb.log({"loss": loss})
 
            model_mean = self.inference()
            grid = self.image_grid(model_mean, epoch)
           
        return 0
    
    @torch.no_grad()
    def inference(self):
        shape = (self.config.train.batch_size, self.config.dataset.channels, 
                 self.config.dataset.resolution, self.config.dataset.resolution)

        b = shape[0]
        x = torch.randn(shape, device=self.device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            betas_t = self.extract(self.betas, t, x.shape)
            self.sqrt_one_minus_alphas_cumprod_t = self.extract(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            self.sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
            
            model_mean = self.sqrt_recip_alphas_t * (
                x - betas_t * self.model(x, t)[0] / self.sqrt_one_minus_alphas_cumprod_t
            )

            if i > 0:
                posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                x = model_mean
                
            # Logging
            if i % 250 == 0:
                self.log_image_grid(x, i)      
        return model_mean
    
    @torch.no_grad()
    def sdedit(self, x_G, t_0):
        # Load Image
        x_G_PIL = Image.open(x_G)
        x_G_PIL = x_G_PIL.resize((256, 256))
        transform = transforms.ToTensor()
        x_G_Tensor = transform(x_G_PIL)
        x_G_Batch = x_G_Tensor.unsqueeze(0)
        
        # Add Noise
        e = torch.randn_like(x_G_Batch)
        a = (1 - self.betas).cumprod(dim=0)
        x = x_G_Batch * a[t_0 - 1].sqrt() + e * (1.0 - a[t_0 - 1]).sqrt()
        x = model_mean + torch.sqrt(posterior_variance_t) * noise
        x = x.to(self.device)
        
        # Log
        save_image(x_G_Batch, f"01_GT.png")
        save_image(x, f"01_t0_{t_0}.png")
        
        # Inference
        shape = (self.config.train.batch_size, self.config.dataset.channels, 
                 self.config.dataset.resolution, self.config.dataset.resolution)

        b = shape[0]
        
        for i in tqdm(reversed(range(0, t_0)), desc='sampling loop time step', total=t_0):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            
            betas_t = self.extract(self.betas, t, x.shape)
            self.sqrt_one_minus_alphas_cumprod_t = self.extract(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            self.sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
            
            model_mean = self.sqrt_recip_alphas_t * (
                x - betas_t * self.model(x, t)[0] / self.sqrt_one_minus_alphas_cumprod_t
            )

            if i > 0:
                posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                x = model_mean
        
        save_image(x, f"Final.png")
        print("END") 
        return 0
        

class SBDM(Base):
    """
    Score-Based Diffusion Model
    """
    def __init__(self):
        
        self.sigmas = torch.exp(torch.linspace(
            torch.log(torch.tensor(self.config.model.sigma_begin)),
            torch.log(torch.tensor(self.config.model.sigma_end)),
            self.config.model.num_classes)).float().to(self.config.device)
        
    def train():
        return 0
    
    def anneal_dsm_score_estimation(self):
        used_sigmas = self.sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
        perturbed_samples = self.samples + torch.randn_like(samples) * used_sigmas
        target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
        scores = scorenet(perturbed_samples, labels)
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** 2
        return loss.mean(dim=0)
    
    def anneal_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
        images = []
        with torch.no_grad():
            for c, sigma in enumerate(sigmas):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * torch.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
            return images