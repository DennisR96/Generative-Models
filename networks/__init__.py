from .ddpm.unet import Unet
from .dummy import dummy 
from .esrgan_plus.architecture import RRDBNet, Discriminator_VGG_128, VGGFeatureExtractor

def load_network(config): 
    if config.network.name == "DDPM":
        model = Unet(config)
    elif config.network.name == "DUMMY":
        netG = dummy.Generator(config)
        netD = dummy.Discriminator(config)
        return [netG, netD]
    elif config.network.name == "ESRGAN+":
        netG = RRDBNet(config)
        netD = Discriminator_VGG_128(config)
        netF = VGGFeatureExtractor(config)
        return [netG, netD, netF]
    return model