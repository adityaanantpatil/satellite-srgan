import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def denormalize(tensor):
    """
    Denormalize tensor from [-1, 1] to [0, 1]
    Handles both PyTorch tensors and numpy arrays
    """
    # Check if it's a PyTorch tensor
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensor = tensor.numpy()
    
    # Now it's a numpy array
    return (tensor + 1) / 2


def prepare_image_for_metrics(img):
    """
    Prepare image for metric calculation
    Converts to numpy array in (H, W, C) format with values in [0, 1]
    """
    # Convert to numpy if needed
    if isinstance(img, torch.Tensor):
        if img.is_cuda:
            img = img.cpu()
        img = img.detach().numpy()
    
    # Denormalize from [-1, 1] to [0, 1]
    img = (img + 1) / 2
    
    # Handle different input shapes
    if img.ndim == 4:
        # Batch dimension present (B, C, H, W) - take first image
        img = img[0]
    
    if img.ndim == 3:
        # Check if channels are first (C, H, W) or last (H, W, C)
        if img.shape[0] == 3 or img.shape[0] == 1:
            # Channels first (C, H, W) -> transpose to (H, W, C)
            img = np.transpose(img, (1, 2, 0))
    
    # Ensure float64 for better precision
    img = img.astype(np.float64)
    
    # Clip to valid range
    img = np.clip(img, 0, 1)
    
    return img


def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images
    Args:
        img1, img2: Images in range [-1, 1] (tensor or numpy array)
    Returns:
        PSNR value in dB
    """
    # Prepare images
    img1 = prepare_image_for_metrics(img1)
    img2 = prepare_image_for_metrics(img2)
    
    # Calculate PSNR
    return psnr(img1, img2, data_range=1.0)


def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images
    Args:
        img1, img2: Images in range [-1, 1] (tensor or numpy array)
    Returns:
        SSIM value between 0 and 1
    """
    # Prepare images
    img1 = prepare_image_for_metrics(img1)
    img2 = prepare_image_for_metrics(img2)
    
    # Get image dimensions
    height, width = img1.shape[0], img1.shape[1]
    min_dim = min(height, width)
    
    # Determine appropriate window size
    if min_dim < 7:
        # For very small images, use smaller window
        win_size = 3 if min_dim >= 3 else min_dim
        if win_size % 2 == 0:  # Must be odd
            win_size = max(3, win_size - 1)
    else:
        win_size = 7  # Default window size
    
    # Ensure window size doesn't exceed image dimensions
    win_size = min(win_size, min_dim)
    if win_size % 2 == 0:  # Must be odd
        win_size -= 1
    
    # Additional safety check
    if win_size < 3:
        print(f"Warning: Image too small ({height}x{width}) for reliable SSIM calculation")
        win_size = 3
    
    try:
        # Calculate SSIM
        return ssim(
            img1, img2, 
            data_range=1.0, 
            channel_axis=2 if img1.ndim == 3 else None,
            win_size=win_size
        )
    except Exception as e:
        print(f"Error calculating SSIM for images of shape {img1.shape}: {e}")
        print(f"Attempted win_size: {win_size}, min_dim: {min_dim}")
        raise


def calculate_metrics(sr_images, hr_images):
    """
    Calculate average PSNR and SSIM for a batch of images
    Args:
        sr_images: Super-resolved images (batch of tensors or numpy arrays)
        hr_images: High-resolution ground truth images
    Returns:
        avg_psnr, avg_ssim
    """
    batch_size = len(sr_images)
    total_psnr = 0
    total_ssim = 0
    
    for i in range(batch_size):
        sr_img = sr_images[i]
        hr_img = hr_images[i]
        
        # Convert to numpy if needed
        if isinstance(sr_img, torch.Tensor):
            sr_img = sr_img.detach().cpu().numpy()
        if isinstance(hr_img, torch.Tensor):
            hr_img = hr_img.detach().cpu().numpy()
        
        total_psnr += calculate_psnr(sr_img, hr_img)
        total_ssim += calculate_ssim(sr_img, hr_img)
    
    return total_psnr / batch_size, total_ssim / batch_size