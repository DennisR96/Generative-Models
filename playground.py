import os 
import yaml
from datasets import load_dataset
from utils.utils import dict2namespace
from torchvision.utils import save_image, make_grid

from diffusers import AutoPipelineForImage2Image
import torch
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader

from networks.esrgan_plus.architecture import RRDBNet, Discriminator_VGG_128

## Load Configuration
config_path = "config/ffhq_diffusion.yaml"
with open(os.path.join(config_path), "r") as f:
    config_yaml = yaml.safe_load(f)
config = dict2namespace(config_yaml)

netG = RRDBNet(config).to("mps")
netD = Discriminator_VGG_128(config).to("mps")
x = torch.rand((64, 3, 128, 128)).to("mps")
out = netG(x)

z = netD(out)
print(z.shape)



# ffhq = load_dataset(config)

# dl = DataLoader(ffhq, 16, shuffle=True)
# x = next(iter(dl))

# save_image(x["images"], "images.png", normalize=True)
# save_image(x["lowres"], "lowres.png", normalize=True)