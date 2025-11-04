"""
Comprehensive SRGAN Testing Script
Tests the trained model and generates comparisons and visualizations
Usage: python scripts/test_srgan.py --checkpoint checkpoints/srgan/best.pth
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

from models.saved_models.generator import Generator
from utils.metrics import calculate_psnr, calculate_ssim
from utils.data_loader import SatelliteDataset
from torch.utils.data import DataLoader
from config import config


def load_model(checkpoint_path, device):
    """Load trained SRGAN model"""
    print(f"Loading model from {checkpoint_path}...")
    generator = Generator().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    print("✓ Model loaded successfully")
    return generator


def test_single_image(generator, lr_img, hr_img, device):
    """Test on a single image and return metrics"""
    with torch.no_grad():
        # Ensure inputs don't have batch dimension
        if lr_img.ndim == 3:
            lr_img = lr_img.unsqueeze(0)
        if hr_img.ndim == 3:
            hr_img = hr_img.unsqueeze(0)
            
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)
        
        sr_img = generator(lr_img)
        
        # Remove batch dimension for metric calculation
        sr_img_cpu = sr_img.squeeze(0).cpu()
        hr_img_cpu = hr_img.squeeze(0).cpu()
        
        # Calculate metrics on CPU tensors without batch dimension
        psnr = calculate_psnr(sr_img_cpu, hr_img_cpu)
        ssim = calculate_ssim(sr_img_cpu, hr_img_cpu)
    
    return sr_img_cpu, psnr, ssim


def create_comparison_grid(lr_img, sr_img, hr_img, psnr, ssim, save_path):
    """Create side-by-side comparison image"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert tensors to numpy images
    def tensor_to_numpy(tensor):
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
    
    lr_np = tensor_to_numpy(lr_img)
    sr_np = tensor_to_numpy(sr_img)
    hr_np = tensor_to_numpy(hr_img)
    
    # Display images
    axes[0].imshow(lr_np)
    axes[0].set_title('Low Resolution Input', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(sr_np)
    axes[1].set_title(f'SRGAN Output\nPSNR: {psnr:.2f}dB | SSIM: {ssim:.4f}', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(hr_np)
    axes[2].set_title('Ground Truth (High Res)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_on_test_set(generator, test_loader, device, num_samples=20):
    """Evaluate model on test set and generate comparisons"""
    print("\n" + "="*50)
    print("Evaluating on Test Set")
    print("="*50)
    
    # Create output directories
    comparisons_dir = Path('results/test_comparisons')
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    
    all_psnr = []
    all_ssim = []
    
    # Generate comparisons for first num_samples
    print(f"\nGenerating {num_samples} comparison images...")
    sample_count = 0
    for lr_imgs, hr_imgs in tqdm(test_loader, total=num_samples, desc="Generating comparisons"):
        if sample_count >= num_samples:
            break
        
        # Get first image from batch (batch_size=1)
        lr_img = lr_imgs[0]
        hr_img = hr_imgs[0]
        
        sr_img, psnr, ssim = test_single_image(generator, lr_img, hr_img, device)
        
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        
        # Save comparison
        save_path = comparisons_dir / f'test_comparison_{sample_count+1:03d}.png'
        create_comparison_grid(lr_img, sr_img, hr_img, psnr, ssim, save_path)
        
        sample_count += 1
    
    # Evaluate on remaining test set
    if len(test_loader) > num_samples:
        print(f"\nEvaluating remaining {len(test_loader) - num_samples} images...")
        for idx, (lr_imgs, hr_imgs) in enumerate(tqdm(test_loader, desc="Testing remaining"), start=1):
            if idx <= num_samples:
                continue
            
            lr_img = lr_imgs[0]
            hr_img = hr_imgs[0]
            
            _, psnr, ssim = test_single_image(generator, lr_img, hr_img, device)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
    
    # Calculate statistics
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    std_psnr = np.std(all_psnr)
    std_ssim = np.std(all_ssim)
    
    print("\n" + "="*50)
    print("Test Results")
    print("="*50)
    print(f"Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"Min PSNR: {min(all_psnr):.2f} dB")
    print(f"Max PSNR: {max(all_psnr):.2f} dB")
    print(f"Min SSIM: {min(all_ssim):.4f}")
    print(f"Max SSIM: {max(all_ssim):.4f}")
    
    # Save results
    results = {
        'avg_psnr': float(avg_psnr),
        'std_psnr': float(std_psnr),
        'avg_ssim': float(avg_ssim),
        'std_ssim': float(std_ssim),
        'min_psnr': float(min(all_psnr)),
        'max_psnr': float(max(all_psnr)),
        'min_ssim': float(min(all_ssim)),
        'max_ssim': float(max(all_ssim)),
        'num_samples': len(all_psnr)
    }
    
    results_path = Path('results/metrics/test_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to {results_path}")
    print(f"✓ Comparison images saved to {comparisons_dir}")
    
    return results, all_psnr, all_ssim


def plot_metrics_distribution(psnr_values, ssim_values, save_dir):
    """Plot distribution of PSNR and SSIM values"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PSNR histogram
    axes[0].hist(psnr_values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(psnr_values), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(psnr_values):.2f} dB')
    axes[0].set_xlabel('PSNR (dB)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('PSNR Distribution on Test Set', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SSIM histogram
    axes[1].hist(ssim_values, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(ssim_values), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(ssim_values):.4f}')
    axes[1].set_xlabel('SSIM', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('SSIM Distribution on Test Set', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'metrics_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics distribution plot saved to {save_path}")


def plot_training_curves(history_path, save_dir):
    """Plot training curves from saved history"""
    print("\nGenerating training curves...")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['g_loss']) + 1)
    
    # Generator and Discriminator Loss
    axes[0, 0].plot(epochs, history['g_loss'], 'b-', linewidth=2, label='Generator Loss')
    axes[0, 0].plot(epochs, history['d_loss'], 'r-', linewidth=2, label='Discriminator Loss')
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Generator vs Discriminator Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training PSNR
    axes[0, 1].plot(epochs, history['psnr'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training PSNR over Epochs', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Validation PSNR
    axes[1, 0].plot(epochs, history['val_psnr'], 'purple', linewidth=2, marker='o')
    axes[1, 0].axhline(max(history['val_psnr']), color='red', linestyle='--', 
                      label=f'Best: {max(history["val_psnr"]):.2f} dB')
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Validation PSNR over Epochs', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation SSIM
    axes[1, 1].plot(epochs, history['val_ssim'], 'orange', linewidth=2, marker='s')
    axes[1, 1].axhline(max(history['val_ssim']), color='red', linestyle='--',
                      label=f'Best: {max(history["val_ssim"]):.4f}')
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('SSIM', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Validation SSIM over Epochs', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to {save_path}")


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    generator = load_model(args.checkpoint, device)
    
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
    
    # Evaluate on test set
    results, psnr_values, ssim_values = evaluate_on_test_set(
        generator, test_loader, device, num_samples=args.num_samples
    )
    
    # Generate visualizations
    results_dir = Path('results/metrics')
    
    # Plot metrics distribution
    plot_metrics_distribution(psnr_values, ssim_values, results_dir)
    
    # Plot training curves if history exists
    history_files = list(Path('results/training_history').glob('srgan_history_*.json'))
    if history_files:
        latest_history = max(history_files, key=lambda p: p.stat().st_mtime)
        print(f"\nUsing training history: {latest_history}")
        plot_training_curves(latest_history, results_dir)
    else:
        print("\nNo training history found. Skipping training curves.")
    
    print("\n" + "="*50)
    print("Testing Complete!")
    print("="*50)
    print(f"\nGenerated files:")
    print(f"  - Test comparisons: results/test_comparisons/")
    print(f"  - Metrics distribution: results/metrics/metrics_distribution.png")
    print(f"  - Training curves: results/metrics/training_curves.png")
    print(f"  - Test results: results/metrics/test_results.json")
    print("\n✓ All done! Results ready for LinkedIn/GitHub showcase.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SRGAN on test dataset')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/srgan/best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of comparison images to generate')
    
    args = parser.parse_args()
    main(args)