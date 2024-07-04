from ddpm.unet import Unet
import dummy.dummy

def load_network(config):
    print("Hallo Welt")
    
    if config.network.name == "DDPM":
        model = Unet(config)
        
    if config.network.dummy == "DUMMY":
        netG = dummy.dummy.Generator(config)
        netD = dummy.dummy.Generator(config)
        return [netG, netD]
        
    if config.network.name == "ESRGAN":
        model = 
    return model