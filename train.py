import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import VGG19_Weights
import os
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np


from models.saved_models.generator import Generator
from models.saved_models.discriminator import Discriminator
from utils.data_loader import SatelliteDataset
from utils.metrics import calculate_psnr, calculate_ssim


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features"""
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        # Fixed: Use weights parameter instead of deprecated pretrained
        vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        
        # Extract features from specific layers
        self.slice1 = nn.Sequential(*list(vgg.children())[:5])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[5:10])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[10:19]) # relu3_4
        self.slice4 = nn.Sequential(*list(vgg.children())[19:28]) # relu4_4
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.mse_loss = nn.MSELoss()
        
    def forward(self, x, y):
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        x = (x - mean) / std
        y = (y - mean) / std
        
        # Extract features
        x_relu1 = self.slice1(x)
        y_relu1 = self.slice1(y)
        
        x_relu2 = self.slice2(x_relu1)
        y_relu2 = self.slice2(y_relu1)
        
        x_relu3 = self.slice3(x_relu2)
        y_relu3 = self.slice3(y_relu2)
        
        x_relu4 = self.slice4(x_relu3)
        y_relu4 = self.slice4(y_relu3)
        
        # Compute MSE loss at each layer
        loss = self.mse_loss(x_relu1, y_relu1) + \
               self.mse_loss(x_relu2, y_relu2) + \
               self.mse_loss(x_relu3, y_relu3) + \
               self.mse_loss(x_relu4, y_relu4)
        
        return loss / 4.0


class SRGANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = VGGPerceptualLoss().to(self.device)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        
        # Optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config['g_lr'],
            betas=(0.9, 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config['d_lr'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate schedulers
        self.g_scheduler = optim.lr_scheduler.StepLR(
            self.g_optimizer,
            step_size=config['lr_decay_step'],
            gamma=config['lr_decay_gamma']
        )
        self.d_scheduler = optim.lr_scheduler.StepLR(
            self.d_optimizer,
            step_size=config['lr_decay_step'],
            gamma=config['lr_decay_gamma']
        )
        
        # Training history
        self.history = {
            'g_loss': [], 'd_loss': [], 'content_loss': [],
            'perceptual_loss': [], 'adversarial_loss': [],
            'psnr': [], 'ssim': [], 'val_psnr': [], 'val_ssim': []
        }
        
    def train_epoch(self, train_loader, epoch):
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_psnr = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(pbar):
            batch_size = lr_imgs.size(0)
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device)
            
            # Generate fake high-res images
            fake_hr = self.generator(lr_imgs)
            
            # =================== Train Discriminator ===================
            self.d_optimizer.zero_grad()
            
            # Real images
            real_labels = torch.ones(batch_size, 1).to(self.device)
            real_output = self.discriminator(hr_imgs)
            d_real_loss = self.adversarial_loss(real_output, real_labels)
            
            # Fake images
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            fake_output = self.discriminator(fake_hr.detach())
            d_fake_loss = self.adversarial_loss(fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.d_optimizer.step()
            
            # =================== Train Generator ===================
            self.g_optimizer.zero_grad()
            
            # Adversarial loss (fool discriminator)
            fake_output = self.discriminator(fake_hr)
            adversarial_loss = self.adversarial_loss(fake_output, real_labels)
            
            # Content loss (MSE)
            content_loss = self.mse_loss(fake_hr, hr_imgs)
            
            # Perceptual loss (VGG)
            perceptual_loss = self.perceptual_loss(fake_hr, hr_imgs)
            
            # Total generator loss
            g_loss = (self.config['content_weight'] * content_loss + 
                     self.config['perceptual_weight'] * perceptual_loss +
                     self.config['adversarial_weight'] * adversarial_loss)
            
            g_loss.backward()
            self.g_optimizer.step()
            
            # Calculate PSNR
            with torch.no_grad():
                psnr = calculate_psnr(fake_hr, hr_imgs)
            
            # Update metrics
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_psnr += psnr
            
            # Update progress bar
            pbar.set_postfix({
                'G_Loss': f'{g_loss.item():.4f}',
                'D_Loss': f'{d_loss.item():.4f}',
                'PSNR': f'{psnr:.2f}dB'
            })
        
        # Calculate epoch averages
        num_batches = len(train_loader)
        return {
            'g_loss': epoch_g_loss / num_batches,
            'd_loss': epoch_d_loss / num_batches,
            'psnr': epoch_psnr / num_batches
        }
    
    def validate(self, val_loader):
        self.generator.eval()
        
        val_psnr = 0
        val_ssim = 0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                # Generate fake high-res images
                fake_hr = self.generator(lr_imgs)
                
                # Calculate metrics
                psnr = calculate_psnr(fake_hr, hr_imgs)
                ssim = calculate_ssim(fake_hr, hr_imgs)
                
                val_psnr += psnr
                val_ssim += ssim
        
        num_batches = len(val_loader)
        return {
            'psnr': val_psnr / num_batches,
            'ssim': val_ssim / num_batches
        }
    
    def train(self, train_loader, val_loader, num_epochs):
        best_psnr = 0
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update learning rates
            self.g_scheduler.step()
            self.d_scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train - G_Loss: {train_metrics['g_loss']:.4f}, "
                  f"D_Loss: {train_metrics['d_loss']:.4f}, "
                  f"PSNR: {train_metrics['psnr']:.2f}dB")
            print(f"Val - PSNR: {val_metrics['psnr']:.2f}dB, "
                  f"SSIM: {val_metrics['ssim']:.4f}")
            
            # Save history
            self.history['g_loss'].append(train_metrics['g_loss'])
            self.history['d_loss'].append(train_metrics['d_loss'])
            self.history['psnr'].append(train_metrics['psnr'])
            self.history['val_psnr'].append(val_metrics['psnr'])
            self.history['val_ssim'].append(val_metrics['ssim'])
            self.history['ssim'].append(val_metrics['ssim'])
            
            # Save best model
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                self.save_checkpoint(epoch, 'best')
                print(f"✓ New best PSNR: {best_psnr:.2f}dB")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch}')
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, epoch, name):
        checkpoint_dir = 'checkpoints/srgan'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'history': self.history
        }
        
        path = os.path.join(checkpoint_dir, f'{name}.pth')
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        print(f"✓ Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def save_history(self):
        """Save training history to JSON with numpy type conversion"""
        history_dir = 'results/training_history'
        os.makedirs(history_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(history_dir, f'srgan_history_{timestamp}.json')
        
        # Convert numpy types to Python native types
        def convert_to_native(obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        with open(path, 'w') as f:
            json.dump(convert_to_native(self.history), f, indent=4)
        
        print(f"✓ Training history saved: {path}")


if __name__ == "__main__":
    # Training configuration
    config = {
        'batch_size': 16,
        'num_epochs': 100,
        'g_lr': 1e-4,
        'd_lr': 1e-4,
        'lr_decay_step': 50,
        'lr_decay_gamma': 0.1,
        'content_weight': 1.0,
        'perceptual_weight': 0.006,
        'adversarial_weight': 0.001,
    }
    
    # Load datasets
    train_dataset = SatelliteDataset('data/train', augment=True)
    val_dataset = SatelliteDataset('data/val', augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=4)
    
    # Initialize trainer
    trainer = SRGANTrainer(config)
    
    # Start training
    print("Starting SRGAN training...")
    history = trainer.train(train_loader, val_loader, config['num_epochs'])
    
    print("\n✓ Training complete!")