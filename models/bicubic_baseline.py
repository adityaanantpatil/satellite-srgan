# models/bicubic_baseline.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.metrics import calculate_psnr, calculate_ssim

def bicubic_upscale(lr_image, scale_factor=4):
    """
    Upscale image using bicubic interpolation
    """
    return F.interpolate(
        lr_image,
        scale_factor=scale_factor,
        mode='bicubic',
        align_corners=False
    )

def evaluate_bicubic(test_loader, device):
    """
    Evaluate bicubic interpolation on test set
    """
    psnr_values = []
    ssim_values = []
    
    print("🔍 Evaluating Bicubic Interpolation...")
    
    with torch.no_grad():
        for lr_images, hr_images in tqdm(test_loader, desc="Testing Bicubic"):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            # Bicubic upscaling
            sr_images = bicubic_upscale(lr_images, scale_factor=4)
            
            # Calculate metrics
            psnr = calculate_psnr(sr_images, hr_images)
            ssim = calculate_ssim(sr_images, hr_images)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
    
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    
    print(f"\n✅ Bicubic Results:")
    print(f"   Average PSNR: {avg_psnr:.2f} dB")
    print(f"   Average SSIM: {avg_ssim:.4f}")
    
    return avg_psnr, avg_ssim