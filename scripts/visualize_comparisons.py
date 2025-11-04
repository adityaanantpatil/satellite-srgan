# scripts/visualize_comparisons.py
"""
Visualization script for comparing baseline models
Generates side-by-side comparisons of LR, Bicubic, SRCNN, Improved SRCNN, and HR images
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import config
from utils.data_loader import get_data_loaders
from models.saved_models.baseline_models import BicubicUpsampler, SRCNN, ImprovedSRCNN


def denormalize(tensor):
    """Denormalize tensor from ImageNet stats"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def generate_comparisons(num_samples=20):
    """Generate comparison images for baseline models"""
    
    print("="*70)
    print("GENERATING BASELINE COMPARISONS")
    print("="*70)
    
    device = config.DEVICE
    
    # Load test data
    _, _, test_loader = get_data_loaders(config)
    
    # Load models
    print("\n📦 Loading models...")
    
    # Bicubic
    bicubic_model = BicubicUpsampler(scale_factor=4).to(device)
    
    # SRCNN - ADD weights_only=False
    srcnn_model = SRCNN(num_channels=3, scale_factor=4).to(device)
    checkpoint = torch.load(
        "models/saved_models/checkpoints/srcnn/best_model.pth",
        weights_only=False  # FIX: Add this parameter
    )
    srcnn_model.load_state_dict(checkpoint['model_state_dict'])
    srcnn_model.eval()
    
    # Improved SRCNN - ADD weights_only=False
    improved_srcnn = ImprovedSRCNN(num_channels=3, scale_factor=4).to(device)
    checkpoint = torch.load(
        "models/saved_models/checkpoints/improvedsrcnn/best_model.pth",
        weights_only=False  # FIX: Add this parameter
    )
    improved_srcnn.load_state_dict(checkpoint['model_state_dict'])
    improved_srcnn.eval()
    
    print("✅ Models loaded successfully!")
    
    # Create output directory
    output_dir = "results/comparisons"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🎨 Generating {num_samples} comparison images...")
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (lr_images, hr_images) in enumerate(test_loader):
            if sample_count >= num_samples:
                break
            
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            # Generate SR images with all models
            bicubic_sr = bicubic_model(lr_images)
            srcnn_sr = srcnn_model(lr_images)
            improved_sr = improved_srcnn(lr_images)
            
            # Ensure all images have the same dimensions
            target_size = (hr_images.size(2), hr_images.size(3))
            
            if bicubic_sr.shape != hr_images.shape:
                bicubic_sr = F.interpolate(bicubic_sr, size=target_size, mode='bicubic', align_corners=False)
            if srcnn_sr.shape != hr_images.shape:
                srcnn_sr = F.interpolate(srcnn_sr, size=target_size, mode='bicubic', align_corners=False)
            if improved_sr.shape != hr_images.shape:
                improved_sr = F.interpolate(improved_sr, size=target_size, mode='bicubic', align_corners=False)
            
            # Process each image in the batch
            for i in range(lr_images.size(0)):
                if sample_count >= num_samples:
                    break
                
                # Denormalize images
                lr_img = denormalize(lr_images[i].cpu())
                hr_img = denormalize(hr_images[i].cpu())
                bicubic_img = denormalize(bicubic_sr[i].cpu())
                srcnn_img = denormalize(srcnn_sr[i].cpu())
                improved_img = denormalize(improved_sr[i].cpu())
                
                # Clamp values to [0, 1]
                lr_img = torch.clamp(lr_img, 0, 1)
                hr_img = torch.clamp(hr_img, 0, 1)
                bicubic_img = torch.clamp(bicubic_img, 0, 1)
                srcnn_img = torch.clamp(srcnn_img, 0, 1)
                improved_img = torch.clamp(improved_img, 0, 1)
                
                # Create comparison figure
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                
                # Convert to numpy for plotting
                images = [lr_img, bicubic_img, srcnn_img, improved_img, hr_img]
                titles = ['Low Resolution (64×64)', 'Bicubic (256×256)', 
                         'SRCNN (256×256)', 'Improved SRCNN (256×256)', 
                         'Ground Truth (256×256)']
                
                for ax, img, title in zip(axes, images, titles):
                    img_np = img.permute(1, 2, 0).numpy()
                    ax.imshow(img_np)
                    ax.set_title(title, fontsize=10)
                    ax.axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/comparison_{sample_count+1:03d}.png", 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                sample_count += 1
                
                if (sample_count) % 5 == 0:
                    print(f"  ✅ Generated {sample_count}/{num_samples} comparisons")
    
    print(f"\n✅ All {num_samples} comparison images saved to: {output_dir}/")
    print("\n" + "="*70)
    print("🎉 VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Check the results in: {output_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    generate_comparisons(num_samples=20)
