from .base import Base
import torch

class SBDM(Base):
    """
    Score-Based Diffusion Model
    """
    def __init__(self):
        
        self.sigmas = torch.exp(torch.linspace(
            torch.log(torch.tensor(self.config.model.sigma_begin)),
            torch.log(torch.tensor(self.config.model.sigma_end)),
            self.config.model.num_classes)).float().to(self.config.device)
    
    def anneal_dsm_score_estimation(self):
        used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
        perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
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