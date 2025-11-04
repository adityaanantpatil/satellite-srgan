"""
Comprehensive Model Comparison Script
Compares Bicubic, SRCNN, and SRGAN against ground truth
Usage: python scripts/compare_models.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import cv2

from models.saved_models.generator import Generator
from models.srcnn import SRCNN  # SRCNN is in models/, not models/saved_models/
from utils.metrics import calculate_psnr, calculate_ssim
from utils.data_loader import SatelliteDataset
from torch.utils.data import DataLoader
from config import config


def load_srgan(checkpoint_path, device):
    """Load trained SRGAN model"""
    generator = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    return generator


def load_srcnn(checkpoint_path, device):
    """Load trained SRCNN model with architecture auto-detection"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_state = checkpoint['model_state_dict']
    
    # Detect architecture from checkpoint
    # Check conv2 kernel size to determine which architecture was used
    conv2_weight_key = None
    for key in checkpoint_state.keys():
        if 'layer2.weight' in key or 'conv2.weight' in key:
            conv2_weight_key = key
            break
    
    if conv2_weight_key is None:
        raise ValueError("Could not find conv2/layer2 weight in checkpoint")
    
    conv2_shape = checkpoint_state[conv2_weight_key].shape
    kernel_size = conv2_shape[2]  # Get kernel size from shape [out_ch, in_ch, k, k]
    
    print(f"  Detected SRCNN architecture: conv2 kernel_size={kernel_size}x{kernel_size}")
    
    # Define SRCNN architecture matching the checkpoint
    import torch.nn as nn
    import torch.nn.functional as F
    
    class SRCNNCompatible(nn.Module):
        def __init__(self, kernel_size_conv2=1):
            super(SRCNNCompatible, self).__init__()
            self.scale_factor = 4
            
            # Adjust padding for conv2 based on kernel size
            padding_conv2 = (kernel_size_conv2 - 1) // 2
            
            self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
            self.conv2 = nn.Conv2d(64, 32, kernel_size=kernel_size_conv2, padding=padding_conv2)
            self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        
        def forward(self, x):
            # Bicubic upsampling
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.conv3(x)
            return x
    
    # Create model with detected architecture
    srcnn = SRCNNCompatible(kernel_size_conv2=kernel_size).to(device)
    
    # Rename keys if needed (layer -> conv)
    new_state_dict = {}
    for key in checkpoint_state.keys():
        new_key = key.replace('layer1', 'conv1').replace('layer2', 'conv2').replace('layer3', 'conv3')
        new_state_dict[new_key] = checkpoint_state[key]
    
    srcnn.load_state_dict(new_state_dict)
    srcnn.eval()
    return srcnn


def bicubic_upscale(lr_img, scale_factor=4):
    """Perform bicubic upscaling"""
    # Convert tensor to numpy
    if isinstance(lr_img, torch.Tensor):
        lr_img = lr_img.cpu().numpy()
    
    # Denormalize from [-1, 1] to [0, 1]
    lr_img = (lr_img + 1) / 2
    
    # Handle channel order (C, H, W) -> (H, W, C)
    if lr_img.shape[0] == 3:
        lr_img = np.transpose(lr_img, (1, 2, 0))
    
    # Get target size
    h, w = lr_img.shape[:2]
    target_h, target_w = h * scale_factor, w * scale_factor
    
    # Bicubic interpolation
    bicubic = cv2.resize(lr_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    # Convert back to tensor format (H, W, C) -> (C, H, W)
    bicubic = np.transpose(bicubic, (2, 0, 1))
    
    # Normalize back to [-1, 1]
    bicubic = bicubic * 2 - 1
    
    return torch.from_numpy(bicubic).float()


def process_single_image(lr_img, hr_img, srgan, srcnn, device):
    """Process a single image through all methods"""
    with torch.no_grad():
        # Prepare inputs
        lr_batch = lr_img.unsqueeze(0).to(device)
        hr_batch = hr_img.unsqueeze(0).to(device)
        
        # Get HR size for consistency
        hr_size = hr_img.shape[-2:]  # (H, W)
        
        # Bicubic upscaling
        bicubic_img = bicubic_upscale(lr_img, scale_factor=4)
        bicubic_batch = bicubic_img.unsqueeze(0).to(device)
        
        # SRCNN (needs LR input, does bicubic internally)
        srcnn_batch = srcnn(lr_batch)
        srcnn_img = srcnn_batch.squeeze(0).cpu()
        
        # Ensure SRCNN output matches HR size (in case of size mismatch)
        if srcnn_img.shape[-2:] != hr_size:
            srcnn_img = torch.nn.functional.interpolate(
                srcnn_img.unsqueeze(0),
                size=hr_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # SRGAN
        srgan_batch = srgan(lr_batch)
        srgan_img = srgan_batch.squeeze(0).cpu()
        
        # Ensure SRGAN output matches HR size
        if srgan_img.shape[-2:] != hr_size:
            srgan_img = torch.nn.functional.interpolate(
                srgan_img.unsqueeze(0),
                size=hr_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Ensure bicubic matches HR size
        if bicubic_img.shape[-2:] != hr_size:
            bicubic_img = torch.nn.functional.interpolate(
                bicubic_img.unsqueeze(0),
                size=hr_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Move to CPU for metrics
        hr_cpu = hr_img.cpu()
        bicubic_cpu = bicubic_img.cpu()
        
        # Calculate metrics for each method
        metrics = {
            'bicubic': {
                'psnr': calculate_psnr(bicubic_cpu, hr_cpu),
                'ssim': calculate_ssim(bicubic_cpu, hr_cpu)
            },
            'srcnn': {
                'psnr': calculate_psnr(srcnn_img, hr_cpu),
                'ssim': calculate_ssim(srcnn_img, hr_cpu)
            },
            'srgan': {
                'psnr': calculate_psnr(srgan_img, hr_cpu),
                'ssim': calculate_ssim(srgan_img, hr_cpu)
            }
        }
        
        return bicubic_cpu, srcnn_img, srgan_img, metrics


def create_4way_comparison(lr_img, bicubic_img, srcnn_img, srgan_img, hr_img, metrics, save_path):
    """Create comparison grid with all 5 images"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Convert tensors to displayable numpy arrays
    def tensor_to_display(tensor):
        if isinstance(tensor, torch.Tensor):
            img = tensor.detach().cpu().numpy()
        else:
            img = tensor
        
        # Denormalize from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        
        # Handle channel order
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        return np.clip(img, 0, 1)
    
    lr_np = tensor_to_display(lr_img)
    bicubic_np = tensor_to_display(bicubic_img)
    srcnn_np = tensor_to_display(srcnn_img)
    srgan_np = tensor_to_display(srgan_img)
    hr_np = tensor_to_display(hr_img)
    
    # Top row: LR Input, Bicubic, SRCNN
    axes[0, 0].imshow(lr_np)
    axes[0, 0].set_title('Low Resolution Input\n(Original)', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(bicubic_np)
    axes[0, 1].set_title(
        f'Bicubic Interpolation\nPSNR: {metrics["bicubic"]["psnr"]:.2f}dB | SSIM: {metrics["bicubic"]["ssim"]:.4f}',
        fontsize=11, fontweight='bold'
    )
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(srcnn_np)
    axes[0, 2].set_title(
        f'SRCNN\nPSNR: {metrics["srcnn"]["psnr"]:.2f}dB | SSIM: {metrics["srcnn"]["ssim"]:.4f}',
        fontsize=11, fontweight='bold'
    )
    axes[0, 2].axis('off')
    
    # Bottom row: SRGAN, Ground Truth, Improvement Chart
    axes[1, 0].imshow(srgan_np)
    axes[1, 0].set_title(
        f'SRGAN (Ours)\nPSNR: {metrics["srgan"]["psnr"]:.2f}dB | SSIM: {metrics["srgan"]["ssim"]:.4f}',
        fontsize=11, fontweight='bold', color='green'
    )
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(hr_np)
    axes[1, 1].set_title('Ground Truth\n(High Resolution)', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Metrics comparison bar chart
    methods = ['Bicubic', 'SRCNN', 'SRGAN']
    psnr_values = [metrics['bicubic']['psnr'], metrics['srcnn']['psnr'], metrics['srgan']['psnr']]
    ssim_values = [metrics['bicubic']['ssim'], metrics['srcnn']['ssim'], metrics['srgan']['ssim']]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax = axes[1, 2]
    bars1 = ax.bar(x - width/2, psnr_values, width, label='PSNR (dB)', color='skyblue')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, ssim_values, width, label='SSIM', color='lightgreen')
    
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontweight='bold')
    ax2.set_ylabel('SSIM', fontweight='bold')
    ax.set_title('Metrics Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, psnr_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    for bar, val in zip(bars2, ssim_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_all_methods(srgan, srcnn, test_loader, device, num_samples=20):
    """Evaluate all methods on test set"""
    print("\n" + "="*70)
    print("Comparing Bicubic, SRCNN, and SRGAN")
    print("="*70)
    
    # Create output directory
    comparisons_dir = Path('results/model_comparisons')
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all metrics
    all_metrics = {
        'bicubic': {'psnr': [], 'ssim': []},
        'srcnn': {'psnr': [], 'ssim': []},
        'srgan': {'psnr': [], 'ssim': []}
    }
    
    # Generate comparisons
    print(f"\nGenerating {num_samples} comparison images...")
    sample_count = 0
    
    for lr_imgs, hr_imgs in tqdm(test_loader, total=num_samples, desc="Processing"):
        if sample_count >= num_samples:
            break
        
        lr_img = lr_imgs[0]
        hr_img = hr_imgs[0]
        
        # Process through all methods
        bicubic_img, srcnn_img, srgan_img, metrics = process_single_image(
            lr_img, hr_img, srgan, srcnn, device
        )
        
        # Store metrics
        for method in ['bicubic', 'srcnn', 'srgan']:
            all_metrics[method]['psnr'].append(metrics[method]['psnr'])
            all_metrics[method]['ssim'].append(metrics[method]['ssim'])
        
        # Save comparison image
        save_path = comparisons_dir / f'comparison_{sample_count+1:03d}.png'
        create_4way_comparison(lr_img, bicubic_img, srcnn_img, srgan_img, hr_img, metrics, save_path)
        
        sample_count += 1
    
    # Evaluate on full test set
    print("\nEvaluating on full test set...")
    for idx, (lr_imgs, hr_imgs) in enumerate(tqdm(test_loader, desc="Full evaluation"), start=1):
        if idx <= num_samples:
            continue
        
        lr_img = lr_imgs[0]
        hr_img = hr_imgs[0]
        
        _, _, _, metrics = process_single_image(lr_img, hr_img, srgan, srcnn, device)
        
        for method in ['bicubic', 'srcnn', 'srgan']:
            all_metrics[method]['psnr'].append(metrics[method]['psnr'])
            all_metrics[method]['ssim'].append(metrics[method]['ssim'])
    
    # Calculate statistics
    results = {}
    print("\n" + "="*70)
    print("FINAL RESULTS - Method Comparison")
    print("="*70)
    
    for method in ['bicubic', 'srcnn', 'srgan']:
        psnr_values = all_metrics[method]['psnr']
        ssim_values = all_metrics[method]['ssim']
        
        results[method] = {
            'avg_psnr': float(np.mean(psnr_values)),
            'std_psnr': float(np.std(psnr_values)),
            'avg_ssim': float(np.mean(ssim_values)),
            'std_ssim': float(np.std(ssim_values)),
            'min_psnr': float(np.min(psnr_values)),
            'max_psnr': float(np.max(psnr_values)),
            'min_ssim': float(np.min(ssim_values)),
            'max_ssim': float(np.max(ssim_values))
        }
        
        print(f"\n{method.upper()}:")
        print(f"  PSNR: {results[method]['avg_psnr']:.2f} ± {results[method]['std_psnr']:.2f} dB")
        print(f"  SSIM: {results[method]['avg_ssim']:.4f} ± {results[method]['std_ssim']:.4f}")
    
    # Calculate improvements
    print("\n" + "="*70)
    print("IMPROVEMENTS OVER BICUBIC")
    print("="*70)
    
    for method in ['srcnn', 'srgan']:
        psnr_improvement = results[method]['avg_psnr'] - results['bicubic']['avg_psnr']
        ssim_improvement = results[method]['avg_ssim'] - results['bicubic']['avg_ssim']
        psnr_percent = (psnr_improvement / results['bicubic']['avg_psnr']) * 100
        ssim_percent = (ssim_improvement / results['bicubic']['avg_ssim']) * 100
        
        print(f"\n{method.upper()} vs Bicubic:")
        print(f"  PSNR: +{psnr_improvement:.2f} dB ({psnr_percent:+.1f}%)")
        print(f"  SSIM: +{ssim_improvement:.4f} ({ssim_percent:+.1f}%)")
    
    # SRGAN vs SRCNN
    psnr_improvement = results['srgan']['avg_psnr'] - results['srcnn']['avg_psnr']
    ssim_improvement = results['srgan']['avg_ssim'] - results['srcnn']['avg_ssim']
    psnr_percent = (psnr_improvement / results['srcnn']['avg_psnr']) * 100
    ssim_percent = (ssim_improvement / results['srcnn']['avg_ssim']) * 100
    
    print(f"\nSRGAN vs SRCNN:")
    print(f"  PSNR: {psnr_improvement:+.2f} dB ({psnr_percent:+.1f}%)")
    print(f"  SSIM: {ssim_improvement:+.4f} ({ssim_percent:+.1f}%)")
    
    # Save results
    results_path = Path('results/metrics/comparison_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    results['improvements'] = {
        'srcnn_vs_bicubic': {
            'psnr_gain': float(results['srcnn']['avg_psnr'] - results['bicubic']['avg_psnr']),
            'ssim_gain': float(results['srcnn']['avg_ssim'] - results['bicubic']['avg_ssim'])
        },
        'srgan_vs_bicubic': {
            'psnr_gain': float(results['srgan']['avg_psnr'] - results['bicubic']['avg_psnr']),
            'ssim_gain': float(results['srgan']['avg_ssim'] - results['bicubic']['avg_ssim'])
        },
        'srgan_vs_srcnn': {
            'psnr_gain': float(results['srgan']['avg_psnr'] - results['srcnn']['avg_psnr']),
            'ssim_gain': float(results['srgan']['avg_ssim'] - results['srcnn']['avg_ssim'])
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to {results_path}")
    print(f"✓ Comparison images saved to {comparisons_dir}")
    
    return results, all_metrics


def plot_overall_comparison(results, save_dir):
    """Create overall comparison visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    methods = ['Bicubic', 'SRCNN', 'SRGAN']
    psnr_values = [results['bicubic']['avg_psnr'], 
                   results['srcnn']['avg_psnr'], 
                   results['srgan']['avg_psnr']]
    ssim_values = [results['bicubic']['avg_ssim'], 
                   results['srcnn']['avg_ssim'], 
                   results['srgan']['avg_ssim']]
    psnr_stds = [results['bicubic']['std_psnr'], 
                 results['srcnn']['std_psnr'], 
                 results['srgan']['std_psnr']]
    ssim_stds = [results['bicubic']['std_ssim'], 
                 results['srcnn']['std_ssim'], 
                 results['srgan']['std_ssim']]
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    # PSNR comparison
    bars1 = axes[0].bar(methods, psnr_values, color=colors, alpha=0.8, yerr=psnr_stds, capsize=5)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    axes[0].set_title('Average PSNR Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, psnr_values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # SSIM comparison
    bars2 = axes[1].bar(methods, ssim_values, color=colors, alpha=0.8, yerr=ssim_stds, capsize=5)
    axes[1].set_ylabel('SSIM', fontsize=12, fontweight='bold')
    axes[1].set_title('Average SSIM Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, ssim_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'overall_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Overall comparison plot saved to {save_path}")


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load models
    print("\nLoading models...")
    print(f"  - SRGAN from: {args.srgan_checkpoint}")
    srgan = load_srgan(args.srgan_checkpoint, device)
    print("  ✓ SRGAN loaded")
    
    print(f"  - SRCNN from: {args.srcnn_checkpoint}")
    srcnn = load_srcnn(args.srcnn_checkpoint, device)
    print("  ✓ SRCNN loaded")
    
    print("  - Bicubic interpolation (built-in)")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = SatelliteDataset(
        'data/processed/test',
        hr_size=config.HR_SIZE,
        lr_size=config.LR_SIZE,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    print(f"✓ Loaded {len(test_dataset)} test images")
    
    # Run comparison
    results, all_metrics = evaluate_all_methods(
        srgan, srcnn, test_loader, device, num_samples=args.num_samples
    )
    
    # Generate visualizations
    results_dir = Path('results/metrics')
    plot_overall_comparison(results, results_dir)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - Model comparisons: results/model_comparisons/")
    print(f"  - Overall comparison: results/metrics/overall_comparison.png")
    print(f"  - Detailed results: results/metrics/comparison_results.json")
    print("\n✓ ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare Bicubic, SRCNN, and SRGAN')
    parser.add_argument('--srgan-checkpoint', type=str, default='checkpoints/srgan/best.pth',
                       help='Path to SRGAN checkpoint')
    parser.add_argument('--srcnn-checkpoint', type=str, default='models/saved_models/checkpoints/srcnn/best_model.pth',
                       help='Path to SRCNN checkpoint')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of comparison images to generate')
    
    args = parser.parse_args()
    main(args)