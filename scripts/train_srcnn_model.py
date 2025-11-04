"""
SRCNN Training Script
Train the SRCNN baseline model for satellite image super-resolution
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from utils.data_loader import get_dataloaders
from utils.metrics import calculate_psnr, calculate_ssim

# Import SRCNN model
try:
    from models.srcnn import SRCNN
    print("✅ Imported SRCNN model")
except ImportError:
    print("❌ Could not import SRCNN. Make sure models/srcnn.py exists")
    sys.exit(1)


class SRCNNTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Create directories - FIXED: Use consistent naming
        self.checkpoint_dir = Path("checkpoints/srcnn")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = Path("results/srcnn_training")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = SRCNN().to(self.device)
        print(f"🎯 SRCNN initialized on {self.device}")
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE if hasattr(config, 'LEARNING_RATE') else 1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'learning_rate': []
        }
        
        # Best metrics
        self.best_val_loss = float('inf')
        self.best_psnr = 0
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            sr_imgs = self.model(lr_imgs)
            
            # Calculate loss
            loss = self.criterion(sr_imgs, hr_imgs)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in tqdm(val_loader, desc="Validating", leave=False):
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                # Forward pass
                sr_imgs = self.model(lr_imgs)
                
                # Calculate loss
                loss = self.criterion(sr_imgs, hr_imgs)
                total_loss += loss.item()
                
                # Calculate metrics for each image in batch
                # Convert to numpy arrays for metric calculation
                sr_imgs_np = sr_imgs.detach().cpu().numpy()
                hr_imgs_np = hr_imgs.detach().cpu().numpy()
                
                for i in range(len(sr_imgs_np)):
                    psnr = calculate_psnr(sr_imgs_np[i], hr_imgs_np[i])
                    ssim = calculate_ssim(sr_imgs_np[i], hr_imgs_np[i])
                    total_psnr += psnr
                    total_ssim += ssim
        
        # Calculate averages
        num_samples = len(val_loader.dataset)
        avg_loss = total_loss / len(val_loader)
        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples
        
        return avg_loss, avg_psnr, avg_ssim
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_psnr': self.best_psnr,
            'history': self.history
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'srcnn_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'srcnn_best.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not Path(checkpoint_path).exists():
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return 0
        
        # FIXED: Added weights_only=False for PyTorch 2.6+
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_psnr = checkpoint['best_psnr']
        self.history = checkpoint['history']
        
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def save_history(self):
        """Save training history to JSON file"""
        # FIXED: Use self.results_dir instead of self.save_dir
        history_path = self.results_dir / 'training_history.json'
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_history = {}
        for key, values in self.history.items():
            serializable_history[key] = [float(v) if hasattr(v, 'item') else v for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"💾 Saved training history to {history_path}")
    
    def train(self, train_loader, val_loader, num_epochs, resume=False):
        """Main training loop"""
        start_epoch = 0
        
        if resume:
            latest_checkpoint = self.checkpoint_dir / 'srcnn_latest.pth'
            start_epoch = self.load_checkpoint(latest_checkpoint)
        
        # Enable memory optimizations
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'amp'):
            print("✅ Mixed precision training enabled")
        
        print("\n" + "="*80)
        print("🚀 Starting SRCNN Training")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Epochs: {start_epoch + 1} to {num_epochs}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batch size: {train_loader.batch_size}")
        print("="*80 + "\n")
        
        for epoch in range(start_epoch + 1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_psnr, val_ssim = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_psnr'].append(val_psnr)
            self.history['val_ssim'].append(val_ssim)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   Val PSNR:   {val_psnr:.2f} dB")
            print(f"   Val SSIM:   {val_ssim:.4f}")
            print(f"   LR:         {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_psnr > self.best_psnr
            if is_best:
                self.best_psnr = val_psnr
                self.best_val_loss = val_loss
                print(f"   🌟 New best PSNR: {val_psnr:.2f} dB")
            
            self.save_checkpoint(epoch, is_best)
            
            # Save history every 5 epochs
            if epoch % 5 == 0:
                self.save_history()
        
        # Final save
        self.save_history()
        
        print("\n" + "="*80)
        print("✅ Training Complete!")
        print("="*80)
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"Training history saved to: {self.results_dir}")


def main():
    # Load configuration
    config = Config()
    
    # Get data loaders
    print("📁 Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Initialize trainer
    trainer = SRCNNTrainer(config)
    
    # Training settings
    num_epochs = 25  # You can change this
    resume = False   # Set to True to resume from checkpoint
    
    # Train model
    trainer.train(train_loader, val_loader, num_epochs, resume)
    
    # Test the best model
    print("\n" + "="*80)
    print("🧪 Testing Best Model")
    print("="*80)
    
    # Load best checkpoint
    best_checkpoint = trainer.checkpoint_dir / 'srcnn_best.pth'
    trainer.load_checkpoint(best_checkpoint)
    
    # Test
    test_loss, test_psnr, test_ssim = trainer.validate(test_loader)
    
    print(f"\n📊 Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test PSNR: {test_psnr:.2f} dB")
    print(f"   Test SSIM: {test_ssim:.4f}")
    
    # Save test results
    test_results = {
        'test_loss': float(test_loss),
        'test_psnr': float(test_psnr),
        'test_ssim': float(test_ssim),
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    test_results_path = trainer.results_dir / 'test_results.json'
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✅ Test results saved to: {test_results_path}")


if __name__ == "__main__":
    main()