# scripts/train_baseline.py
"""
Complete baseline training and evaluation script
Trains SRCNN models and evaluates all baselines including Bicubic
"""


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
from datetime import datetime


# Import from your existing modules
from config import config
from utils.data_loader import get_data_loaders
from utils.metrics import calculate_psnr, calculate_ssim
from models.saved_models.baseline_models import BicubicUpsampler, SRCNN, ImprovedSRCNN



def evaluate_model(model, test_loader, device, model_name):
    """Evaluate a single model on test set"""
    model.eval()
    psnr_values = []
    ssim_values = []
    
    print(f"\n🔍 Evaluating {model_name}...")
    
    with torch.no_grad():
        for lr_images, hr_images in tqdm(test_loader, desc=f"Testing {model_name}"):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            # Generate super-resolution images
            sr_images = model(lr_images)
            
            # CRITICAL FIX: Ensure sr_images match hr_images dimensions exactly
            if sr_images.shape != hr_images.shape:
                sr_images = F.interpolate(
                    sr_images,
                    size=(hr_images.size(2), hr_images.size(3)),
                    mode='bicubic',
                    align_corners=False
                )
            
            # Calculate metrics
            psnr = calculate_psnr(sr_images, hr_images)
            ssim = calculate_ssim(sr_images, hr_images)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
    
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    
    print(f"✅ {model_name} Results:")
    print(f"   PSNR: {avg_psnr:.2f} dB")
    print(f"   SSIM: {avg_ssim:.4f}")
    
    return avg_psnr, avg_ssim



def train_srcnn(model, train_loader, val_loader, device, num_epochs=50, model_name="SRCNN"):
    """Train SRCNN model"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    best_psnr = 0
    save_dir = f"models/saved_models/checkpoints/{model_name.lower()}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n🚀 Training {model_name}...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for lr_images, hr_images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            # Forward pass
            sr_images = model(lr_images)
            
            # Ensure dimensions match for loss calculation
            if sr_images.shape != hr_images.shape:
                sr_images = F.interpolate(
                    sr_images,
                    size=(hr_images.size(2), hr_images.size(3)),
                    mode='bicubic',
                    align_corners=False
                )
            
            loss = criterion(sr_images, hr_images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_psnr_values = []
        
        with torch.no_grad():
            for lr_images, hr_images in val_loader:
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)
                
                sr_images = model(lr_images)
                
                # Ensure dimensions match
                if sr_images.shape != hr_images.shape:
                    sr_images = F.interpolate(
                        sr_images,
                        size=(hr_images.size(2), hr_images.size(3)),
                        mode='bicubic',
                        align_corners=False
                    )
                
                psnr = calculate_psnr(sr_images, hr_images)
                val_psnr_values.append(psnr)
        
        avg_psnr = sum(val_psnr_values) / len(val_psnr_values)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_train_loss:.4f}, PSNR: {avg_psnr:.2f} dB")
        
        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr
            }, f"{save_dir}/best_model.pth")
            print(f"  ✅ Best model saved! PSNR: {best_psnr:.2f} dB")
        
        scheduler.step()
    
    return model, best_psnr



def main():
    """Main training and evaluation pipeline"""
    
    print("="*70)
    print("SATELLITE IMAGE SUPER-RESOLUTION - BASELINE TRAINING")
    print("="*70)
    
    # Setup
    device = config.DEVICE
    print(f"\n📱 Device: {device}")
    
    # Load data
    print("\n📊 Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(config)
    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")
    print(f"✅ Test batches: {len(test_loader)}")
    
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {}
    }
    
    # 1. Evaluate Bicubic Baseline (no training needed)
    print("\n" + "="*70)
    print("BASELINE 1: BICUBIC INTERPOLATION")
    print("="*70)
    
    bicubic_model = BicubicUpsampler(scale_factor=4).to(device)
    bicubic_psnr, bicubic_ssim = evaluate_model(bicubic_model, test_loader, device, "Bicubic")
    
    results["models"]["bicubic"] = {
        "psnr": float(bicubic_psnr),
        "ssim": float(bicubic_ssim),
        "parameters": 0
    }
    
    # 2. Train and evaluate SRCNN
    print("\n" + "="*70)
    print("BASELINE 2: SRCNN")
    print("="*70)
    
    srcnn_model = SRCNN(num_channels=3, scale_factor=4)
    num_params = sum(p.numel() for p in srcnn_model.parameters())
    print(f"📊 SRCNN Parameters: {num_params:,}")
    
    srcnn_model, srcnn_train_psnr = train_srcnn(
        srcnn_model, train_loader, val_loader, device, 
        num_epochs=30, model_name="SRCNN"
    )
    
    # Load best model and evaluate on test set - FIXED with weights_only=False
    checkpoint = torch.load(
        "models/saved_models/checkpoints/srcnn/best_model.pth",
        weights_only=False  # FIX: Add this parameter
    )
    srcnn_model.load_state_dict(checkpoint['model_state_dict'])
    srcnn_psnr, srcnn_ssim = evaluate_model(srcnn_model, test_loader, device, "SRCNN")
    
    results["models"]["srcnn"] = {
        "psnr": float(srcnn_psnr),
        "ssim": float(srcnn_ssim),
        "parameters": num_params
    }
    
    # 3. Train and evaluate Improved SRCNN
    print("\n" + "="*70)
    print("BASELINE 3: IMPROVED SRCNN")
    print("="*70)
    
    improved_srcnn = ImprovedSRCNN(num_channels=3, scale_factor=4)
    num_params_improved = sum(p.numel() for p in improved_srcnn.parameters())
    print(f"📊 Improved SRCNN Parameters: {num_params_improved:,}")
    
    improved_srcnn, improved_train_psnr = train_srcnn(
        improved_srcnn, train_loader, val_loader, device,
        num_epochs=30, model_name="ImprovedSRCNN"
    )
    
    # Load best model and evaluate - FIXED with weights_only=False
    checkpoint = torch.load(
        "models/saved_models/checkpoints/improvedsrcnn/best_model.pth",
        weights_only=False  # FIX: Add this parameter
    )
    improved_srcnn.load_state_dict(checkpoint['model_state_dict'])
    improved_psnr, improved_ssim = evaluate_model(improved_srcnn, test_loader, device, "Improved SRCNN")
    
    results["models"]["improved_srcnn"] = {
        "psnr": float(improved_psnr),
        "ssim": float(improved_ssim),
        "parameters": num_params_improved
    }
    
    # Final comparison
    print("\n" + "="*70)
    print("📊 BASELINE COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Model':<20} {'PSNR (dB)':<12} {'SSIM':<10} {'Parameters':<15}")
    print("-" * 70)
    
    for model_name, metrics in results["models"].items():
        print(f"{model_name.upper():<20} {metrics['psnr']:<12.2f} {metrics['ssim']:<10.4f} {metrics['parameters']:<15,}")
    
    # Calculate improvements
    print("\n" + "="*70)
    print("📈 IMPROVEMENTS OVER BICUBIC")
    print("="*70)
    
    for model_name, metrics in results["models"].items():
        if model_name != "bicubic":
            psnr_gain = metrics['psnr'] - results["models"]["bicubic"]["psnr"]
            ssim_gain = metrics['ssim'] - results["models"]["bicubic"]["ssim"]
            print(f"{model_name.upper()}: +{psnr_gain:.2f} dB PSNR, +{ssim_gain:.4f} SSIM")
    
    # Save results
    os.makedirs('results/metrics', exist_ok=True)
    with open('results/metrics/baseline_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n✅ Results saved to: results/metrics/baseline_results.json")
    print("\n" + "="*70)
    print("🎉 BASELINE TRAINING COMPLETE!")
    print("="*70)
    print("\n📋 Next Steps:")
    print("  1. Review the results in results/metrics/baseline_results.json")
    print("  2. Run: python scripts/visualize_comparisons.py")
    print("  3. Begin SRGAN implementation (Week 2)")
    print("="*70 + "\n")



if __name__ == "__main__":
    main()
