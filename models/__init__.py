from .base import Base
from .diffusion import DDPM
from .gan import GAN, ESRGAN

def load_model(model_type, config):
    if model_type == "DDPM":
        return DDPM(config)
    if model_type == "GAN":
        return GAN(config)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")