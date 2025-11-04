"""
SRCNN (Super-Resolution Convolutional Neural Network)
Fixed version that upsamples LR images to HR size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """
    SRCNN: Super-Resolution using CNN
    
    Architecture:
    1. Bicubic upsampling (LR -> HR size)
    2. Feature extraction (Conv 9x9)
    3. Non-linear mapping (Conv 1x1)
    4. Reconstruction (Conv 5x5)
    
    Reference: "Image Super-Resolution Using Deep Convolutional Networks"
    """
    
    def __init__(self, num_channels=3, scale_factor=4):
        """
        Args:
            num_channels: Number of input/output channels (default: 3 for RGB)
            scale_factor: Upsampling factor (default: 4)
        """
        super(SRCNN, self).__init__()
        
        self.scale_factor = scale_factor
        
        # Layer 1: Feature extraction
        # 9x9 convolution, 64 filters
        self.conv1 = nn.Conv2d(
            num_channels, 
            64, 
            kernel_size=9, 
            padding=4
        )
        
        # Layer 2: Non-linear mapping
        # 1x1 convolution, 32 filters
        self.conv2 = nn.Conv2d(
            64, 
            32, 
            kernel_size=1, 
            padding=0
        )
        
        # Layer 3: Reconstruction
        # 5x5 convolution, output channels
        self.conv3 = nn.Conv2d(
            32, 
            num_channels, 
            kernel_size=5, 
            padding=2
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input LR image tensor [B, C, H, W]
            
        Returns:
            Super-resolved HR image tensor [B, C, H*scale, W*scale]
        """
        # Step 1: Bicubic upsampling to HR size
        x = F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode='bicubic', 
            align_corners=False
        )
        
        # Step 2: Feature extraction
        x = F.relu(self.conv1(x))
        
        # Step 3: Non-linear mapping
        x = F.relu(self.conv2(x))
        
        # Step 4: Reconstruction
        x = self.conv3(x)
        
        return x
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SRCNNDeep(nn.Module):
    """
    Deeper version of SRCNN with more layers
    Better performance but slower training
    """
    
    def __init__(self, num_channels=3, scale_factor=4):
        super(SRCNNDeep, self).__init__()
        
        self.scale_factor = scale_factor
        
        # Feature extraction
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        
        # Non-linear mapping (multiple layers)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=1, padding=0)
        
        # Reconstruction
        self.conv5 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Bicubic upsampling
        x = F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode='bicubic', 
            align_corners=False
        )
        
        # Feature extraction and mapping
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Reconstruction
        x = self.conv5(x)
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_srcnn():
    """Test SRCNN model"""
    print("Testing SRCNN Model...")
    print("=" * 60)
    
    # Create model
    model = SRCNN(num_channels=3, scale_factor=4)
    print(f"✅ SRCNN created")
    print(f"   Parameters: {model.count_parameters():,}")
    
    # Test with sample input
    batch_size = 4
    lr_size = 64
    hr_size = 256
    
    # Create dummy LR input
    x = torch.randn(batch_size, 3, lr_size, lr_size)
    print(f"\n📥 Input shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"📤 Output shape: {output.shape}")
    print(f"✅ Expected shape: torch.Size([{batch_size}, 3, {hr_size}, {hr_size}])")
    
    # Verify output size
    assert output.shape == (batch_size, 3, hr_size, hr_size), \
        f"Output shape mismatch! Got {output.shape}"
    
    print("\n" + "=" * 60)
    print("✅ SRCNN test passed!")
    print("=" * 60)
    
    # Test deep version
    print("\nTesting SRCNNDeep...")
    model_deep = SRCNNDeep(num_channels=3, scale_factor=4)
    print(f"✅ SRCNNDeep created")
    print(f"   Parameters: {model_deep.count_parameters():,}")
    
    with torch.no_grad():
        output_deep = model_deep(x)
    print(f"📤 Output shape: {output_deep.shape}")
    assert output_deep.shape == (batch_size, 3, hr_size, hr_size)
    print("✅ SRCNNDeep test passed!")


if __name__ == "__main__":
    test_srcnn()