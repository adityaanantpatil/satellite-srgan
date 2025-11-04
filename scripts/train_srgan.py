"""
Quick start script for SRGAN training
Usage: python scripts/train_srgan.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
import json

from models.saved_models.generator import Generator
from models.saved_models.discriminator import Discriminator
from utils.data_loader import SatelliteDataset
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import save_comparison_images, plot_training_curves
import config


def setup_training():
    """Setup training environment"""
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    # Create necessary directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.COMPARISONS_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    
    # Device setup
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    return device


def load_datasets():
    """Load train, validation, and test datasets"""
    print("\n" + "="*50)
    print("Loading Datasets")
    print("="*50)
    
    train_dataset = SatelliteDataset(
        config.TRAIN_DIR,
        hr_size=config.HR_SIZE,
        lr_size=config.LR_SIZE,
        augment=True
    )
    
    val_dataset = SatelliteDataset(
        config.VAL_DIR,
        hr_size=config.HR_SIZE,
        lr_size=config.LR_SIZE,
        augment=False
    )
    
    test_dataset = SatelliteDataset(
        config.TEST_DIR,
        hr_size=config.HR_SIZE,
        lr_size=config.LR_SIZE,
        augment=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    return train_loader, val_loader, test_loader


def initialize_models(device):
    """Initialize Generator and Discriminator"""
    print("\n" + "="*50)
    print("Initializing Models")
    print("="*50)
    
    generator = Generator(
        num_residual_blocks=config.NUM_RESIDUAL_BLOCKS,
        num_channels=config.NUM_CHANNELS
    ).to(device)
    
    discriminator = Discriminator().to(device)
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    print(f"Total parameters: {g_params + d_params:,}")
    
    return generator, discriminator


def save_config(config_dict, save_dir):
    """Save configuration to JSON"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_path = os.path.join(save_dir, f'config_{timestamp}.json')
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    print(f"\nConfiguration saved to: {config_path}")


def main(args):
    """Main training function"""
    print("\n" + "="*50)
    print("SRGAN Training - Satellite Image Super-Resolution")
    print("="*50)
    
    # Setup
    device = setup_training()
    
    # Save configuration
    save_config(config.get_config(), config.RESULTS_DIR)
    
    # Load data
    train_loader, val_loader, test_loader = load_datasets()
    
    # Initialize models
    generator, discriminator = initialize_models(device)
    
    # Import trainer (avoid circular imports)
    from train import SRGANTrainer
    
    # Training configuration
    training_config = {
        'batch_size': config.BATCH_SIZE,
        'num_epochs': args.epochs if args.epochs else config.NUM_EPOCHS,
        'g_lr': config.GENERATOR_LR,
        'd_lr': config.DISCRIMINATOR_LR,
        'lr_decay_step': config.LR_DECAY_STEP,
        'lr_decay_gamma': config.LR_DECAY_GAMMA,
        'content_weight': config.CONTENT_WEIGHT,
        'perceptual_weight': config.PERCEPTUAL_WEIGHT,
        'adversarial_weight': config.ADVERSARIAL_WEIGHT,
    }
    
    # Initialize trainer
    trainer = SRGANTrainer(training_config)
    
    # Load checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    print(f"Epochs: {training_config['num_epochs']}")
    print(f"Batch size: {training_config['batch_size']}")
    print(f"Generator LR: {training_config['g_lr']}")
    print(f"Discriminator LR: {training_config['d_lr']}")
    print("="*50 + "\n")
    
    try:
        history = trainer.train(
            train_loader,
            val_loader,
            training_config['num_epochs']
        )
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        
        # Plot training curves
        print("\nGenerating training curves...")
        plot_training_curves(history, save_dir=config.RESULTS_DIR)
        
        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_metrics = trainer.validate(test_loader)
        print(f"Test PSNR: {test_metrics['psnr']:.2f} dB")
        print(f"Test SSIM: {test_metrics['ssim']:.4f}")
        
        # Save sample comparisons
        print("\nGenerating comparison images...")
        save_comparison_images(
            trainer.generator,
            test_loader,
            device,
            save_dir=config.COMPARISONS_DIR,
            num_samples=20
        )
        
        print("\n✓ All done! Check the results directory for outputs.")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer.save_checkpoint(-1, 'interrupted')
        print("✓ Checkpoint saved. You can resume training with --resume flag.")
    
    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SRGAN for satellite image super-resolution')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--pretrain', action='store_true',
                       help='Pre-train generator only (no GAN)')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    main(args)