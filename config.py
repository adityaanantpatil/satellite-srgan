"""
Configuration file for Satellite Super-Resolution GAN
OPTIMIZED FOR GTX 1050 Ti (4GB VRAM) + 8GB RAM
"""

import torch
import os
from datetime import datetime


class Config:
    """Configuration for SRGAN training"""
    
    # ==================== PROJECT PATHS ====================
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
    
    # Data directories
    RAW_DATA_DIR = os.path.join(DATA_ROOT, 'UCMerced_LandUse', 'Images')
    PROCESSED_DATA_DIR = os.path.join(DATA_ROOT, 'processed')
    
    TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, 'train')
    VAL_DIR = os.path.join(PROCESSED_DATA_DIR, 'val')
    TEST_DIR = os.path.join(PROCESSED_DATA_DIR, 'test')
    
    # Model checkpoints
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'models', 'saved_models', 'checkpoints')
    
    # Results directory
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    COMPARISON_DIR = os.path.join(RESULTS_DIR, 'comparisons')
    METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
    
    # ==================== IMAGE SETTINGS ====================
    HR_SIZE = 256  # High-resolution image size
    LR_SIZE = 64   # Low-resolution image size
    SCALE_FACTOR = 4  # Upsampling factor
    CHANNELS = 3   # RGB channels
    
    # ==================== TRAINING SETTINGS ====================
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # OPTIMIZED: Batch size for GTX 1050 Ti (4GB VRAM)
    BATCH_SIZE = 4  # Ultra-safe for 4GB VRAM - maximum stability
    
    # OPTIMIZED: Number of epochs (reduced for faster training)
    NUM_EPOCHS = 50  # Reduced from 100 (still gets good results)
    
    # Learning rates
    LEARNING_RATE = 1e-4
    GENERATOR_LR = 1e-4
    DISCRIMINATOR_LR = 1e-4
    
    # Loss weights
    CONTENT_LOSS_WEIGHT = 1.0
    ADVERSARIAL_LOSS_WEIGHT = 1e-3
    PERCEPTUAL_LOSS_WEIGHT = 1.0
    
    # OPTIMIZED: Training parameters for 8GB RAM
    NUM_WORKERS = 2  # Increased from 0 for faster data loading
    PIN_MEMORY = True if DEVICE == 'cuda' else False
    
    # Learning rate scheduler
    LR_DECAY_STEP = 15  # Adjusted for 50 epochs (was 30)
    LR_DECAY_GAMMA = 0.1
    
    # OPTIMIZED: Memory saving features
    GRADIENT_ACCUMULATION_STEPS = 2  # Simulate larger batch size
    MIXED_PRECISION = True  # Use FP16 to save VRAM (if available)
    
    # ==================== MODEL ARCHITECTURE ====================
    # Generator
    GENERATOR_FEATURES = 64
    GENERATOR_BLOCKS = 16
    
    # Discriminator
    DISCRIMINATOR_FEATURES = 64
    
    # ==================== LOGGING & SAVING ====================
    # Save frequency
    SAVE_EVERY = 5  # Save checkpoint every 5 epochs (was 10)
    VISUALIZE_EVERY = 5  # Generate comparison images every 5 epochs
    
    # Logging
    LOG_INTERVAL = 10  # Log training metrics every N batches
    
    # ==================== EVALUATION METRICS ====================
    # Metrics to track
    METRICS = ['PSNR', 'SSIM']
    
    # ==================== DATA AUGMENTATION ====================
    # Augmentation settings
    USE_AUGMENTATION = True
    HORIZONTAL_FLIP_PROB = 0.5
    VERTICAL_FLIP_PROB = 0.5
    ROTATION_PROB = 0.5
    COLOR_JITTER_PROB = 0.5
    
    # Color jitter parameters
    BRIGHTNESS = 0.2
    CONTRAST = 0.2
    SATURATION = 0.2
    HUE = 0.1
    
    # ==================== RANDOM SEED ====================
    SEED = 42
    
    # ==================== METHODS ====================
    def __init__(self):
        """Initialize config and create necessary directories"""
        self.create_directories()
        self._check_hardware()
    
    def _check_hardware(self):
        """Check hardware and warn if settings might not be optimal"""
        if self.DEVICE == 'cuda':
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                device_name = torch.cuda.get_device_name(0)
                
                # Silent check - only warn if problematic
                if total_memory < 6 and self.BATCH_SIZE > 8:
                    print(f"⚠️  Warning: GPU has {total_memory:.1f}GB VRAM. Consider reducing BATCH_SIZE")
            except:
                pass
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.CHECKPOINT_DIR,
            self.RESULTS_DIR,
            self.COMPARISON_DIR,
            self.METRICS_DIR,
            self.TRAIN_DIR,
            self.VAL_DIR,
            self.TEST_DIR
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_timestamp(self):
        """Get current timestamp for saving files"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def print_config(self):
        """Print configuration settings"""
        print("=" * 80)
        print("CONFIGURATION SETTINGS - OPTIMIZED FOR GTX 1050 Ti")
        print("=" * 80)
        
        print("\n📂 Data Paths:")
        print(f"  Train Dir:      {self.TRAIN_DIR}")
        print(f"  Val Dir:        {self.VAL_DIR}")
        print(f"  Test Dir:       {self.TEST_DIR}")
        print(f"  Checkpoint Dir: {self.CHECKPOINT_DIR}")
        
        print("\n🖼️  Image Settings:")
        print(f"  HR Size:        {self.HR_SIZE}x{self.HR_SIZE}")
        print(f"  LR Size:        {self.LR_SIZE}x{self.LR_SIZE}")
        print(f"  Scale Factor:   {self.SCALE_FACTOR}x")
        print(f"  Channels:       {self.CHANNELS}")
        
        print("\n⚙️  Training Settings (OPTIMIZED):")
        print(f"  Device:         {self.DEVICE}")
        print(f"  Batch Size:     {self.BATCH_SIZE} ⚡ (ultra-safe for 4GB VRAM)")
        print(f"  Epochs:         {self.NUM_EPOCHS} ⚡ (reduced for faster training)")
        print(f"  Learning Rate:  {self.LEARNING_RATE}")
        print(f"  Num Workers:    {self.NUM_WORKERS} ⚡ (2=faster data loading)")
        print(f"  Mixed Precision: {self.MIXED_PRECISION}")
        
        print("\n🗃️  Model Architecture:")
        print(f"  Generator Features:      {self.GENERATOR_FEATURES}")
        print(f"  Generator Blocks:        {self.GENERATOR_BLOCKS}")
        print(f"  Discriminator Features:  {self.DISCRIMINATOR_FEATURES}")
        
        print("\n📊 Loss Weights:")
        print(f"  Content Loss:      {self.CONTENT_LOSS_WEIGHT}")
        print(f"  Adversarial Loss:  {self.ADVERSARIAL_LOSS_WEIGHT}")
        print(f"  Perceptual Loss:   {self.PERCEPTUAL_LOSS_WEIGHT}")
        
        print("\n💾 Saving Settings:")
        print(f"  Save Every:       {self.SAVE_EVERY} epochs")
        print(f"  Visualize Every:  {self.VISUALIZE_EVERY} epochs")
        
        print("\n💡 Performance Tips:")
        print(f"  - Close other applications for best performance")
        print(f"  - Estimated VRAM usage: ~1.5GB / 4GB (very safe!)")
        print(f"  - Estimated training time: ~40-50 minutes (SRCNN)")
        print(f"  - Batch size 4 = maximum stability")
        print(f"  - Num workers 2 = faster data loading")
        
        print("=" * 80 + "\n")
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'hr_size': self.HR_SIZE,
            'lr_size': self.LR_SIZE,
            'scale_factor': self.SCALE_FACTOR,
            'batch_size': self.BATCH_SIZE,
            'num_epochs': self.NUM_EPOCHS,
            'learning_rate': self.LEARNING_RATE,
            'device': self.DEVICE,
            'generator_features': self.GENERATOR_FEATURES,
            'generator_blocks': self.GENERATOR_BLOCKS,
            'content_loss_weight': self.CONTENT_LOSS_WEIGHT,
            'adversarial_loss_weight': self.ADVERSARIAL_LOSS_WEIGHT,
            'perceptual_loss_weight': self.PERCEPTUAL_LOSS_WEIGHT,
            'num_workers': self.NUM_WORKERS,
            'mixed_precision': self.MIXED_PRECISION,
        }
    
    @staticmethod
    def set_seed(seed=42):
        """Set random seed for reproducibility"""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Create default config instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    print("Testing configuration...")
    config = Config()
    config.print_config()
    
    print("\n✅ Configuration test passed!")
    print(f"📂 All directories created successfully!")
    
    # Show GPU info if available
    if torch.cuda.is_available():
        print(f"\n🎮 GPU Info:")
        print(f"  Name: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")