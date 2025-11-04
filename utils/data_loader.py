"""
Data loader for satellite imagery super-resolution
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random


class SatelliteDataset(Dataset):
    """
    Dataset for satellite image super-resolution
    Loads pre-generated HR and LR image pairs
    """
    def __init__(self, data_dir, hr_size=256, lr_size=64, augment=False):
        """
        Args:
            data_dir: Directory containing 'hr' and 'lr' subdirectories
            hr_size: High-resolution image size (default: 256)
            lr_size: Low-resolution image size (default: 64)
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.augment = augment
        
        # Get image paths
        self.hr_dir = self.data_dir / 'hr'
        self.lr_dir = self.data_dir / 'lr'
        
        if not self.hr_dir.exists() or not self.lr_dir.exists():
            raise FileNotFoundError(
                f"HR or LR directory not found in {data_dir}\n"
                f"Expected: {self.hr_dir} and {self.lr_dir}\n"
                f"Please run scripts/prepare_data.py first"
            )
        
        # Get list of image files - FIXED: Now includes .tif files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        self.hr_images = []
        self.lr_images = []
        
        for ext in image_extensions:
            self.hr_images.extend(list(self.hr_dir.glob(ext)))
            self.lr_images.extend(list(self.lr_dir.glob(ext)))
        
        # Sort to ensure matching pairs
        self.hr_images = sorted(self.hr_images)
        self.lr_images = sorted(self.lr_images)
        
        # Verify matching pairs
        assert len(self.hr_images) == len(self.lr_images), \
            f"Mismatch: {len(self.hr_images)} HR images, {len(self.lr_images)} LR images"
        
        # Verify filenames match
        for hr_img, lr_img in zip(self.hr_images, self.lr_images):
            assert hr_img.name == lr_img.name, \
                f"Filename mismatch: {hr_img.name} != {lr_img.name}"
        
        # Define transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        print(f"Loaded {len(self.hr_images)} image pairs from {data_dir}")
    
    def __len__(self):
        return len(self.hr_images)
    
    def __getitem__(self, idx):
        # Load images
        hr_img = Image.open(self.hr_images[idx]).convert('RGB')
        lr_img = Image.open(self.lr_images[idx]).convert('RGB')
        
        # FIXED: Ensure consistent sizes by resizing if needed
        if hr_img.size != (self.hr_size, self.hr_size):
            hr_img = hr_img.resize((self.hr_size, self.hr_size), Image.BICUBIC)
        if lr_img.size != (self.lr_size, self.lr_size):
            lr_img = lr_img.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        
        # Apply augmentation if enabled
        if self.augment:
            hr_img, lr_img = self._augment(hr_img, lr_img)
        
        # Convert to tensor
        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)
        
        # Normalize (optional - can also train without normalization)
        # Uncomment if you want to use ImageNet normalization
        # hr_tensor = self.normalize(hr_tensor)
        # lr_tensor = self.normalize(lr_tensor)
        
        return lr_tensor, hr_tensor
    
    def _augment(self, hr_img, lr_img):
        """Apply same augmentation to both HR and LR images"""
        # Random horizontal flip
        if random.random() > 0.5:
            hr_img = transforms.functional.hflip(hr_img)
            lr_img = transforms.functional.hflip(lr_img)
        
        # Random vertical flip
        if random.random() > 0.5:
            hr_img = transforms.functional.vflip(hr_img)
            lr_img = transforms.functional.vflip(lr_img)
        
        # Random rotation (0, 90, 180, 270 degrees)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            hr_img = transforms.functional.rotate(hr_img, angle)
            lr_img = transforms.functional.rotate(lr_img, angle)
        
        # Color jitter (only on HR, will affect LR through network)
        if random.random() > 0.5:
            color_jitter = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
            hr_img = color_jitter(hr_img)
            lr_img = color_jitter(lr_img)
        
        return hr_img, lr_img


class UnpairedSatelliteDataset(Dataset):
    """
    Dataset that loads images and creates LR versions on-the-fly
    Useful if you only have HR images
    """
    def __init__(self, data_dir, hr_size=256, lr_size=64, augment=False):
        """
        Args:
            data_dir: Directory containing high-resolution images
            hr_size: High-resolution image size
            lr_size: Low-resolution image size
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.augment = augment
        
        # Get image paths
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            self.image_paths.extend(list(self.data_dir.glob(ext)))
        
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {data_dir}")
        
        self.to_tensor = transforms.ToTensor()
        
        print(f"Loaded {len(self.image_paths)} images from {data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Resize to HR size
        hr_img = img.resize((self.hr_size, self.hr_size), Image.BICUBIC)
        
        # Create LR version by downsampling
        lr_img = hr_img.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        
        # Apply augmentation if enabled
        if self.augment:
            hr_img, lr_img = self._augment(hr_img, lr_img)
        
        # Convert to tensor
        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)
        
        return lr_tensor, hr_tensor
    
    def _augment(self, hr_img, lr_img):
        """Same augmentation as SatelliteDataset"""
        # Random horizontal flip
        if random.random() > 0.5:
            hr_img = transforms.functional.hflip(hr_img)
            lr_img = transforms.functional.hflip(lr_img)
        
        # Random vertical flip
        if random.random() > 0.5:
            hr_img = transforms.functional.vflip(hr_img)
            lr_img = transforms.functional.vflip(lr_img)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            hr_img = transforms.functional.rotate(hr_img, angle)
            lr_img = transforms.functional.rotate(lr_img, angle)
        
        return hr_img, lr_img


def get_dataloaders(config):
    """
    Create train, validation, and test data loaders
    
    Args:
        config: Configuration object with data paths and settings
        
    Returns:
        train_loader, val_loader, test_loader
    """
    print("📁 Creating data loaders...")
    
    # Create datasets
    train_dataset = SatelliteDataset(
        data_dir=config.TRAIN_DIR,
        hr_size=config.HR_SIZE,
        lr_size=config.LR_SIZE,
        augment=True  # Enable augmentation for training
    )
    
    val_dataset = SatelliteDataset(
        data_dir=config.VAL_DIR,
        hr_size=config.HR_SIZE,
        lr_size=config.LR_SIZE,
        augment=False  # No augmentation for validation
    )
    
    test_dataset = SatelliteDataset(
        data_dir=config.TEST_DIR,
        hr_size=config.HR_SIZE,
        lr_size=config.LR_SIZE,
        augment=False  # No augmentation for testing
    )
    
    # Create data loaders - FIXED: Set num_workers=0 to avoid multiprocessing issues on Windows
    num_workers = 0 if config.DEVICE == 'cuda' else 0  # Use 0 for Windows compatibility
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        persistent_workers=False  # Disable persistent workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        persistent_workers=False
    )
    
    print(f"✅ Data loaders created:")
    print(f"   Train: {len(train_dataset)} images ({len(train_loader)} batches)")
    print(f"   Val:   {len(val_dataset)} images ({len(val_loader)} batches)")
    print(f"   Test:  {len(test_dataset)} images ({len(test_loader)} batches)")
    
    return train_loader, val_loader, test_loader


def test_dataset():
    """Test the dataset loader"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from config import Config
        config = Config()
    except ImportError:
        print("❌ Cannot import config. Creating dummy config...")
        class Config:
            TRAIN_DIR = 'data/processed/train'
            VAL_DIR = 'data/processed/val'
            TEST_DIR = 'data/processed/test'
            HR_SIZE = 256
            LR_SIZE = 64
            BATCH_SIZE = 4
            DEVICE = 'cuda'
            NUM_WORKERS = 2
        config = Config()
    
    print("Testing SatelliteDataset...")
    
    # Test train dataset
    try:
        train_dataset = SatelliteDataset(
            config.TRAIN_DIR,
            hr_size=config.HR_SIZE,
            lr_size=config.LR_SIZE,
            augment=True
        )
        
        print(f"\n✅ Train dataset loaded: {len(train_dataset)} samples")
        
        # Get a sample
        lr, hr = train_dataset[0]
        print(f"\nSample shapes:")
        print(f"  LR: {lr.shape} (C, H, W)")
        print(f"  HR: {hr.shape} (C, H, W)")
        print(f"\nValue ranges:")
        print(f"  LR: [{lr.min():.3f}, {lr.max():.3f}]")
        print(f"  HR: [{hr.min():.3f}, {hr.max():.3f}]")
        
        # Test data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # Set to 0 for testing
        )
        
        print(f"\n✅ DataLoader created")
        
        # Get a batch
        lr_batch, hr_batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  LR: {lr_batch.shape} (B, C, H, W)")
        print(f"  HR: {hr_batch.shape} (B, C, H, W)")
        
        print("\n✅ Dataset test passed!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run: python scripts/prepare_data.py")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dataset()