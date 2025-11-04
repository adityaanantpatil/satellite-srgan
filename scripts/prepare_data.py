"""
Data Preparation Script for Satellite Super-Resolution
Prepares training, validation, and test datasets
"""

import os
import sys
from pathlib import Path
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from config import Config
except ImportError:
    print("❌ Could not import Config. Make sure config.py exists.")
    sys.exit(1)


def create_directory_structure():
    """Create the required directory structure"""
    config = Config()
    
    # Convert to Path objects if they're strings
    train_dir = Path(config.TRAIN_DIR)
    val_dir = Path(config.VAL_DIR)
    test_dir = Path(config.TEST_DIR)
    
    dirs = [
        train_dir / 'hr',
        train_dir / 'lr',
        val_dir / 'hr',
        val_dir / 'lr',
        test_dir / 'hr',
        test_dir / 'lr',
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")


def generate_lr_from_hr(hr_path, lr_path, scale_factor=4):
    """
    Generate LR image from HR image by downscaling
    
    Args:
        hr_path: Path to HR image
        lr_path: Path to save LR image
        scale_factor: Downscaling factor
    """
    try:
        img = Image.open(hr_path)
        
        # Calculate new size
        new_width = img.width // scale_factor
        new_height = img.height // scale_factor
        
        # Downscale using bicubic interpolation
        lr_img = img.resize((new_width, new_height), Image.BICUBIC)
        
        # Save LR image
        lr_img.save(lr_path)
        return True
    except Exception as e:
        print(f"❌ Error processing {hr_path}: {e}")
        return False


def process_raw_data():
    """
    Process raw satellite images and create HR/LR pairs
    Assumes raw images are in data/raw/ (supports nested directories)
    """
    config = Config()
    data_dir = Path(config.DATA_DIR) if hasattr(config, 'DATA_DIR') else Path('data')
    raw_dir = data_dir / 'raw'
    
    if not raw_dir.exists():
        print(f"❌ Raw data directory not found: {raw_dir}")
        print(f"📁 Please place your satellite images in: {raw_dir}")
        return False
    
    # Get all image files (including nested directories)
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    raw_images = []
    
    # Recursively find all images
    for item in raw_dir.rglob('*'):
        if item.is_file() and item.suffix.lower() in image_extensions:
            raw_images.append(item)
    
    if not raw_images:
        print(f"❌ No images found in {raw_dir}")
        print(f"📝 Searched recursively for: {', '.join(image_extensions)}")
        return False
    
    print(f"📁 Found {len(raw_images)} images in raw directory")
    
    # Split: 70% train, 15% val, 15% test
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(raw_images)
    n = len(raw_images)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train_images = raw_images[:n_train]
    val_images = raw_images[n_train:n_train + n_val]
    test_images = raw_images[n_train + n_val:]
    
    print(f"📊 Split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    # Process each split
    train_dir = Path(config.TRAIN_DIR)
    val_dir = Path(config.VAL_DIR)
    test_dir = Path(config.TEST_DIR)
    
    splits = {
        'train': (train_images, train_dir),
        'val': (val_images, val_dir),
        'test': (test_images, test_dir)
    }
    
    for split_name, (images, output_dir) in splits.items():
        if not images:
            print(f"⚠️  No images for {split_name} split")
            continue
            
        print(f"\n🔄 Processing {split_name} split...")
        
        hr_dir = output_dir / 'hr'
        lr_dir = output_dir / 'lr'
        
        for img_path in tqdm(images, desc=f"Processing {split_name}"):
            # Copy HR image
            hr_dest = hr_dir / img_path.name
            shutil.copy2(img_path, hr_dest)
            
            # Generate LR image
            lr_dest = lr_dir / img_path.name
            generate_lr_from_hr(hr_dest, lr_dest, config.SCALE_FACTOR)
    
    return True


def verify_dataset():
    """Verify that the dataset was created correctly"""
    config = Config()
    processed_dir = Path(config.PROCESSED_DIR) if hasattr(config, 'PROCESSED_DIR') else Path('data/processed')
    
    print("\n" + "="*80)
    print("🔍 Verifying Dataset")
    print("="*80)
    
    splits = ['train', 'val', 'test']
    all_valid = True
    
    for split in splits:
        split_dir = processed_dir / split
        hr_dir = split_dir / 'hr'
        lr_dir = split_dir / 'lr'
        
        hr_count = len(list(hr_dir.glob('*'))) if hr_dir.exists() else 0
        lr_count = len(list(lr_dir.glob('*'))) if lr_dir.exists() else 0
        
        # Filter out directories, only count files
        if hr_dir.exists():
            hr_count = len([f for f in hr_dir.iterdir() if f.is_file()])
        if lr_dir.exists():
            lr_count = len([f for f in lr_dir.iterdir() if f.is_file()])
        
        status = "✅" if hr_count > 0 and hr_count == lr_count else "❌"
        print(f"{status} {split.upper()}: {hr_count} HR images, {lr_count} LR images")
        
        if hr_count == 0 or hr_count != lr_count:
            all_valid = False
    
    return all_valid


def create_sample_data():
    """
    Create sample synthetic data for testing
    Only used if no raw data exists
    """
    config = Config()
    processed_dir = Path(config.PROCESSED_DIR) if hasattr(config, 'PROCESSED_DIR') else Path('data/processed')
    
    print("\n🎨 Creating sample synthetic data...")
    print("⚠️  This is for testing only. Use real satellite images for actual training.")
    
    # Create a few sample images
    n_samples = {'train': 20, 'val': 5, 'test': 5}
    
    for split_name, count in n_samples.items():
        output_dir = processed_dir / split_name
        hr_dir = output_dir / 'hr'
        lr_dir = output_dir / 'lr'
        
        print(f"🔄 Generating {count} samples for {split_name}...")
        
        for i in range(count):
            # Create synthetic HR image (256x256)
            # Create more interesting patterns instead of pure random
            x = np.linspace(0, 4*np.pi, 256)
            y = np.linspace(0, 4*np.pi, 256)
            X, Y = np.meshgrid(x, y)
            
            # Create pattern with some structure
            r = (np.sin(X) * np.cos(Y) * 127 + 128).astype(np.uint8)
            g = (np.cos(X) * np.sin(Y) * 127 + 128).astype(np.uint8)
            b = (np.sin(X + Y) * 127 + 128).astype(np.uint8)
            
            # Add some noise
            noise = np.random.randint(-20, 20, (256, 256, 3), dtype=np.int16)
            hr_array = np.stack([r, g, b], axis=-1).astype(np.int16)
            hr_array = np.clip(hr_array + noise, 0, 255).astype(np.uint8)
            
            hr_img = Image.fromarray(hr_array)
            
            # Save HR
            hr_path = hr_dir / f"sample_{i:03d}.png"
            hr_img.save(hr_path)
            
            # Generate LR (64x64 with scale factor 4)
            lr_img = hr_img.resize((64, 64), Image.BICUBIC)
            lr_path = lr_dir / f"sample_{i:03d}.png"
            lr_img.save(lr_path)
    
    print("✅ Sample data created")


def main():
    """Main data preparation pipeline"""
    print("="*80)
    print("🛰️  Satellite Super-Resolution - Data Preparation")
    print("="*80)
    
    # Step 1: Create directory structure
    print("\n📁 Step 1: Creating directory structure...")
    create_directory_structure()
    
    # Step 2: Process raw data
    print("\n📊 Step 2: Processing raw data...")
    config = Config()
    data_dir = Path(config.DATA_DIR) if hasattr(config, 'DATA_DIR') else Path('data')
    raw_dir = data_dir / 'raw'
    
    if not raw_dir.exists() or not list(raw_dir.glob('*')):
        print(f"\n⚠️  No raw data found in {raw_dir}")
        response = input("Would you like to create sample synthetic data for testing? (y/n): ")
        
        if response.lower() == 'y':
            create_sample_data()
        else:
            print(f"\n📝 Instructions:")
            print(f"1. Create directory: {raw_dir}")
            print(f"2. Place your satellite images (HR quality) in that directory")
            print(f"3. Supported formats: PNG, JPG, JPEG, TIF, TIFF")
            print(f"4. Run this script again")
            return
    else:
        success = process_raw_data()
        if not success:
            return
    
    # Step 3: Verify dataset
    if verify_dataset():
        print("\n" + "="*80)
        print("✅ Data preparation complete!")
        print("="*80)
        print("\n🚀 You can now run: python scripts/train_srcnn.py")
    else:
        print("\n❌ Data verification failed. Please check the errors above.")


if __name__ == "__main__":
    main()