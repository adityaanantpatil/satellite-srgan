"""
Quick SRCNN Test Script
Load the trained SRCNN model and evaluate on test set
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from utils.data_loader import get_dataloaders
from utils.metrics import calculate_psnr, calculate_ssim
from models.srcnn import SRCNN


def test_model(model, test_loader, device):
    """Test the model on test set"""
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    
    print("\n🧪 Testing model...")
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(test_loader, desc="Testing"):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Forward pass
            sr_imgs = model(lr_imgs)
            
            # Calculate loss
            loss = criterion(sr_imgs, hr_imgs)
            total_loss += loss.item()
            
            # Calculate metrics for each image in batch
            sr_imgs_np = sr_imgs.detach().cpu().numpy()
            hr_imgs_np = hr_imgs.detach().cpu().numpy()
            
            for i in range(len(sr_imgs_np)):
                psnr = calculate_psnr(sr_imgs_np[i], hr_imgs_np[i])
                ssim = calculate_ssim(sr_imgs_np[i], hr_imgs_np[i])
                total_psnr += psnr
                total_ssim += ssim
    
    # Calculate averages
    num_samples = len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader)
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    return avg_loss, avg_psnr, avg_ssim


def main():
    print("="*80)
    print("🧪 SRCNN Test Evaluation")
    print("="*80)
    
    # Setup
    config = Config()
    device = torch.device(config.DEVICE)
    
    # Load model
    print(f"\n📦 Loading SRCNN model on {device}...")
    model = SRCNN().to(device)
    
    # Load best checkpoint
    checkpoint_path = Path("checkpoints/srcnn/srcnn_best.pth")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Make sure you've trained the model first!")
        return
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Best validation PSNR: {checkpoint['best_psnr']:.2f} dB")
    
    # Load test data
    print("\n📁 Loading test dataset...")
    _, _, test_loader = get_dataloaders(config)
    print(f"✅ Loaded {len(test_loader.dataset)} test images")
    
    # Test the model
    test_loss, test_psnr, test_ssim = test_model(model, test_loader, device)
    
    # Print results
    print("\n" + "="*80)
    print("📊 TEST RESULTS")
    print("="*80)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test PSNR: {test_psnr:.2f} dB")
    print(f"Test SSIM: {test_ssim:.4f}")
    print("="*80)
    
    # Save results
    results_dir = Path("results/srcnn_training")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    test_results = {
        'test_loss': float(test_loss),
        'test_psnr': float(test_psnr),
        'test_ssim': float(test_ssim),
        'checkpoint_epoch': int(checkpoint['epoch']),
        'validation_psnr': float(checkpoint['best_psnr']),
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_path = results_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")


if __name__ == "__main__":
    main()