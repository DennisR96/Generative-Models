from models.gan import ESRGAN, GAN
import os 
import yaml
from utils.utils import dict2namespace
import torchvision.datasets as datasets
from torchvision import transforms

from datasets.MNIST import MNIST
#from networks.dummy.upscale import Generator, Discriminator
from networks.dummy.dummy import Generator, Discriminator
from networks.StarSRGAN.discriminator import UNetDiscriminator

# Load Config YAML as Namespace 
config_path = "config/test.yaml"
with open(os.path.join(config_path), "r") as f:
    config_yaml = yaml.safe_load(f)
config = dict2namespace(config_yaml)

DS = MNIST(config)

# Load 
#netG = Generator(nz=config.network.latent_vector, ngf=config.network.feature_maps,nc=config.network.channels)
#netD = Discriminator(nc=1,ndf=config.network.feature_maps)
netG = Generator(config)
netD = Discriminator(nc=1, ndf=64)

x = GAN(config, DS, netD=netD, netG=netG)
x.train()