from torch import nn 

class Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.up1 = nn.ConvTranspose2d(in_channels=32, out_channels=256)
        
    def forward(self, z):
        z = self.up1(z)
        return z

class Discriminator(nn.Module):
          