import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual  # Skip connection
        return out


class UpsampleBlock(nn.Module):
    """Upsampling block using PixelShuffle (sub-pixel convolution)"""
    def __init__(self, in_channels, upscale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), 
                              kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    """
    SRGAN Generator Network
    Input: Low-resolution image (64x64x3)
    Output: High-resolution image (256x256x3) - 4x upscaling
    """
    def __init__(self, num_residual_blocks=16, num_channels=64):
        super(Generator, self).__init__()
        
        # Initial feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        
        # Residual blocks
        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(ResidualBlock(num_channels))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Post-residual block
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels)
        )
        
        # Upsampling blocks (4x = 2x + 2x)
        self.upsample1 = UpsampleBlock(num_channels, upscale_factor=2)
        self.upsample2 = UpsampleBlock(num_channels, upscale_factor=2)
        
        # Final output layer
        self.conv3 = nn.Conv2d(num_channels, 3, kernel_size=9, stride=1, padding=4)
        
    def forward(self, x):
        # Initial feature extraction
        x1 = self.conv1(x)
        
        # Residual blocks
        x2 = self.res_blocks(x1)
        
        # Post-residual with skip connection
        x3 = self.conv2(x2)
        x3 = x3 + x1  # Global skip connection
        
        # Upsampling
        x4 = self.upsample1(x3)
        x5 = self.upsample2(x4)
        
        # Final output
        out = self.conv3(x5)
        out = torch.tanh(out)  # Output range [-1, 1]
        
        return out


# Test the generator
if __name__ == "__main__":
    # Create model
    gen = Generator()
    
    # Test with random input
    x = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64
    output = gen(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in gen.parameters()):,}")