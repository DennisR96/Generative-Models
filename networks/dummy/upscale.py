from torch import nn
import torch

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        
        # Downsampling layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        
        # Fully connected layer
        self.fc = nn.Linear(512 * 4 * 4, 1)
        
        # Activation and normalization layers
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        # Downsampling
        x = self.leaky_relu(self.conv1(x))  # First layer without batch norm
        x = self.leaky_relu(self.bn128(self.conv2(x)))
        x = self.leaky_relu(self.bn256(self.conv3(x)))
        x = self.leaky_relu(self.bn512(self.conv4(x)))
        
        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        # Output layer
        x = self.sigmoid(x)
        
        return x

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        
        # Downsampling
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        
        # Upsampling
        self.conv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)
        
        # Activation and normalization layers
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        residual = x
        
        # Downsampling
        x = self.relu(self.bn64(self.conv1(x)))
        x = self.relu(self.bn128(self.conv2(x)))
        x = self.relu(self.bn256(self.conv3(x)))
        
        # Upsampling
        x = self.relu(self.bn128(self.conv4(x)))
        x = self.relu(self.bn64(self.conv5(x)))
        x = self.tanh(self.conv6(x))
        
        return x
netG = Generator()  
netD = Discriminator()     
random_tensor = torch.randn(64, 1, 64, 64)

output = netD(random_tensor)
print(output.shape)