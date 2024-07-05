from torch import nn 
import torch
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.model = nn.Sequential(
            # First transposed convolution layer
            nn.ConvTranspose2d(self.config.network.generator.in_channels, self.config.network.generator.feat_channels, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Second transposed convolution layer
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Third transposed convolution layer
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # Fourth transposed convolution layer
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            # Fifth transposed convolution layer
            nn.ConvTranspose2d(16, self.config.network.generator.out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(self.config.network.discriminator.in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Fully connected layer
        self.fc = nn.Linear(512*4*4, 1)
        
        # Leaky ReLU activation
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Sigmoid activation for output
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc(x))
        return x

Generator
