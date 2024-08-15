from .base import Base
from .diffusion.diffusion import DDPM
from .gan.gan import GAN
from .gan.esrgan import ESRGAN
from .gan.srgan import SRGAN


def load_model(config, network):
    if config.model.name == "DDPM":
        return DDPM(config, network)
    elif config.model.name == "GAN":
        return GAN(config, network)
    elif config.model.name == "SRGAN":
        return SRGAN(config, network)
    elif config.model.name == "ESRGAN":
        return ESRGAN(config, network)
    else:
        raise ValueError(f"Unknown model type: {config.model.name}")