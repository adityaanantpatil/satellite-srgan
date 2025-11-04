import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# ==========================================================
# Bicubic Upsampling (Non-learnable baseline)
# ==========================================================
class BicubicUpsampler(nn.Module):
    """Performs simple bicubic upsampling."""
    def __init__(self, scale_factor=4):
        super(BicubicUpsampler, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)


# ==========================================================
# SRCNN (Super-Resolution CNN)
# Based on: "Image Super-Resolution Using Deep Convolutional Networks" (Dong et al., 2014)
# ==========================================================
class SRCNN(nn.Module):
    def __init__(self, num_channels=3, scale_factor=4):
        super(SRCNN, self).__init__()
        self.scale_factor = scale_factor

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
        self.layer1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.layer2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.layer3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.upsample(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return torch.clamp(x, 0.0, 1.0)


# ==========================================================
# Improved SRCNN (slightly deeper version)
# ==========================================================
class ImprovedSRCNN(nn.Module):
    def __init__(self, num_channels=3, scale_factor=4):
        super(ImprovedSRCNN, self).__init__()
        self.scale_factor = scale_factor

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return torch.clamp(x, 0.0, 1.0)
