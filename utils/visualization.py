"""
Visualization utilities for SRGAN training
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def save_comparison_images(generator, data_loader, device, save_dir, num_samples=10):
    """
    Generate and save comparison images (LR, SR, HR)
    
    Args:
        generator: Trained generator model
        data_loader: DataLoader containing test images
        device: torch device
        save_dir: Directory to save comparison images
        num_samples: Number of samples to generate
    """
    generator.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    saved_count = 0
    
    with torch.no_grad():
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(data_loader):
            if saved_count >= num_samples:
                break
            
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Generate SR images
            sr_imgs = generator(lr_imgs)
            
            # Process each image in batch
            batch_size = lr_imgs.size(0)
            for i in range(batch_size):
                if saved_count >= num_samples:
                    break
                
                # Convert tensors to numpy arrays
                lr_img = tensor_to_image(lr_imgs[i])
                sr_img = tensor_to_image(sr_imgs[i])
                hr_img = tensor_to_image(hr_imgs[i])
                
                # Create comparison figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(lr_img)
                axes[0].set_title('Low Resolution (Input)')
                axes[0].axis('off')
                
                axes[1].imshow(sr_img)
                axes[1].set_title('Super Resolution (Generated)')
                axes[1].axis('off')
                
                axes[2].imshow(hr_img)
                axes[2].set_title('High Resolution (Ground Truth)')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                # Save figure
                save_path = os.path.join(save_dir, f'comparison_{saved_count + 1:03d}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                saved_count += 1
                
                if (saved_count) % 5 == 0:
                    print(f"  Saved {saved_count}/{num_samples} comparison images")
    
    print(f"✓ Saved {saved_count} comparison images to {save_dir}")


def plot_training_curves(history, save_dir):
    """
    Plot and save training curves
    
    Args:
        history: Dictionary containing training history
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot Generator Loss
    if 'g_loss' in history and len(history['g_loss']) > 0:
        axes[0, 0].plot(history['g_loss'], label='Generator Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot Discriminator Loss
    if 'd_loss' in history and len(history['d_loss']) > 0:
        axes[0, 1].plot(history['d_loss'], label='Discriminator Loss', color='orange', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot PSNR
    if 'val_psnr' in history and len(history['val_psnr']) > 0:
        axes[1, 0].plot(history['val_psnr'], label='Validation PSNR', color='green', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].set_title('Validation PSNR')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot SSIM
    if 'val_ssim' in history and len(history['val_ssim']) > 0:
        axes[1, 1].plot(history['val_ssim'], label='Validation SSIM', color='red', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].set_title('Validation SSIM')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Training curves saved to {save_path}")


def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a numpy image
    
    Args:
        tensor: PyTorch tensor of shape (C, H, W) with values in range [-1, 1] or [0, 1]
    
    Returns:
        numpy array of shape (H, W, C) with values in range [0, 1]
    """
    # Clone and detach tensor
    image = tensor.clone().detach().cpu()
    
    # Convert from (C, H, W) to (H, W, C)
    image = image.permute(1, 2, 0).numpy()
    
    # Normalize to [0, 1] range
    if image.min() < 0:  # If data is in [-1, 1] range
        image = (image + 1) / 2
    
    # Clip values to valid range
    image = np.clip(image, 0, 1)
    
    return image


def save_single_image(tensor, save_path):
    """
    Save a single tensor as an image
    
    Args:
        tensor: PyTorch tensor of shape (C, H, W)
        save_path: Path to save the image
    """
    image = tensor_to_image(tensor)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_batch(lr_batch, sr_batch, hr_batch, save_path=None, num_images=4):
    """
    Visualize a batch of LR, SR, and HR images
    
    Args:
        lr_batch: Low resolution image batch
        sr_batch: Super resolution image batch
        hr_batch: High resolution image batch
        save_path: Optional path to save the visualization
        num_images: Number of images to visualize from the batch
    """
    num_images = min(num_images, lr_batch.size(0))
    
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        lr_img = tensor_to_image(lr_batch[i])
        sr_img = tensor_to_image(sr_batch[i])
        hr_img = tensor_to_image(hr_batch[i])
        
        axes[i, 0].imshow(lr_img)
        axes[i, 0].set_title('LR Input' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(sr_img)
        axes[i, 1].set_title('SR Generated' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(hr_img)
        axes[i, 2].set_title('HR Target' if i == 0 else '')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Batch visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)