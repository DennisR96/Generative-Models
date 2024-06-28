from torchvision.models import vgg19
from torch import nn
from torch.autograd import Variable

class Pixel_Loss(nn.Module):
    def __init__(self):
        super(Pixel_Loss, self).__init__()
        self.l1 = nn.L1Loss()
    
    def forward(self, real, fake):
        loss = self.l1(real, fake)
        return loss
    
class Perceptual_Loss(nn.Module):
    def __init__(self):
        super(Perceptual_Loss, self).__init__()
        self.vgg = vgg19(weights='IMAGENET1K_V1')
        self.vgg19_54 = nn.Sequential(*list(self.vgg.features.children())[:35])
        self.l1 = nn.L1Loss()
        
    def forward(self, real, fake):
        real_feat = self.vgg19_54(real)
        fake_feat = self.vgg19_54(fake.detach())
        
        loss = self.l1(real_feat, fake_feat)
        return loss
         
         