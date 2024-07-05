from .base import Base
from .diffusion import DDPM
from .gan import GAN, ESRGAN

def load_model(config, dataset, network):
    if config.model.name == "DDPM":
        return DDPM(config, dataset, network)
    elif config.model.name == "GAN":
        return GAN(config, dataset, network)
    elif config.model.name == "ESRGAN":
        return ESRGAN(config, dataset, network)
    else:
        raise ValueError(f"Unknown model type: {config.model.name}")