# wfrom .ddpm.unet import Unet
# from .esrgan_plus.architecture import RRDBNet, Discriminator_VGG_128, VGGFeatureExtractor
# from .esrgan.discriminator import VGGStyleDiscriminator
# from .esrgan.rrdbnet_arch import RRDBNet as RRDBNet_2
# from .esrgan.vgg_arch import VGGFeatureExtractor as VGGFeatureExtractor_2
# from .HAT import discriminator_arch, hat_arch, srvgg_arch
# from .SwinIR import network_swinir

# def load_network(config): 
#     if config.network.name == "DDPM":
#         model = Unet(config)
#         return model
#     elif config.network.name == "ESRGAN":
#         netG = RRDBNet_2(config)
#         netD = VGGStyleDiscriminator(config)
#         netF = VGGFeatureExtractor_2(config)
#         return [netG, netD, netF]
#     elif config.network.name == "ESRGAN+":
#         netG = RRDBNet(config)
#         netD = Discriminator_VGG_128(config)
#         netF = VGGFeatureExtractor(config)
#         return [netG, netD, netF]
#     elif config.network.name == "HAT":
#         netG = hat_arch.HAT(config)
#         netD = discriminator_arch.UNetDiscriminatorSN(config)    
#         netF = VGGFeatureExtractor(config)              # ESRGAN + 
#         return [netG, netD, netF]
#     elif config.network.name == "SWINIR":
#         netG = network_swinir.SwinIR(config)
#         netD = discriminator_arch.UNetDiscriminatorSN(config)    
#         netF = VGGFeatureExtractor(config)              
#         return [netG, netD, netF]
        
        
    
