"""
Verify that the project setup is correct before training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import torch


def check_python_version():
    """Check Python version"""
    print("\n" + "="*50)
    print("Checking Python Version")
    print("="*50)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Warning: Python 3.8+ recommended")
        return False
    else:
        print("✓ Python version OK")
        return True


def check_pytorch():
    """Check PyTorch installation"""
    print("\n" + "="*50)
    print("Checking PyTorch")
    print("="*50)
    
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print("✓ PyTorch with CUDA OK")
        else:
            print("⚠️  Warning: CUDA not available, will use CPU (very slow)")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not installed!")
        print("\nInstall with:")
        print("  pip install torch torchvision")
        return False


def check_dependencies():
    """Check required dependencies"""
    print("\n" + "="*50)
    print("Checking Dependencies")
    print("="*50)
    
    required = {
        'PIL': 'Pillow',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} not installed")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def check_project_structure():
    """Check project directory structure"""
    print("\n" + "="*50)
    print("Checking Project Structure")
    print("="*50)
    
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        'data',
        'data/raw',
        'models',
        'models/saved_models',
        'utils',
        'scripts',
        'results'
    ]
    
    required_files = [
        'config.py',
        'train.py',
        'evaluate.py',
        'models/saved_models/generator.py',
        'models/saved_models/discriminator.py',
        'utils/data_loader.py',
        'utils/metrics.py',
        'scripts/prepare_data.py'
    ]
    
    all_good = True
    
    # Check directories
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ missing")
            all_good = False
    
    # Check files
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            all_good = False
    
    return all_good


def check_dataset():
    """Check if dataset is prepared"""
    print("\n" + "="*50)
    print("Checking Dataset")
    print("="*50)
    
    import config
    
    # Check raw data
    raw_data_path = Path(config.RAW_DATA_DIR)
    if not raw_data_path.exists():
        print(f"❌ Raw data not found: {raw_data_path}")
        print("\nDownload UC Merced dataset from:")
        print("http://weegee.vision.ucmerced.edu/datasets/landuse.html")
        print(f"Extract to: {raw_data_path}")
        return False
    else:
        print(f"✓ Raw data directory exists")
    
    # Check processed data
    splits = ['train', 'val', 'test']
    all_splits_exist = True
    
    for split in splits:
        split_dir = Path(config.DATA_DIR) / split
        hr_dir = split_dir / 'hr'
        lr_dir = split_dir / 'lr'
        
        if hr_dir.exists() and lr_dir.exists():
            hr_count = len(list(hr_dir.glob('*.png')))
            lr_count = len(list(lr_dir.glob('*.png')))
            print(f"✓ {split}: {hr_count} HR images, {lr_count} LR images")
        else:
            print(f"❌ {split} data not prepared")
            all_splits_exist = False
    
    if not all_splits_exist:
        print("\nPrepare dataset with:")
        print("  python scripts/prepare_data.py")
        return False
    
    return True


def check_models():
    """Check if models can be imported"""
    print("\n" + "="*50)
    print("Checking Models")
    print("="*50)
    
    try:
        from models.saved_models.generator import Generator
        from models.saved_models.discriminator import Discriminator
        
        # Test instantiation
        gen = Generator()
        disc = Discriminator()
        
        # Count parameters
        gen_params = sum(p.numel() for p in gen.parameters())
        disc_params = sum(p.numel() for p in disc.parameters())
        
        print(f"✓ Generator: {gen_params:,} parameters")
        print(f"✓ Discriminator: {disc_params:,} parameters")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gen = gen.to(device)
        disc = disc.to(device)
        
        dummy_lr = torch.randn(1, 3, 64, 64).to(device)
        dummy_hr = torch.randn(1, 3, 256, 256).to(device)
        
        with torch.no_grad():
            gen_output = gen(dummy_lr)
            disc_output = disc(dummy_hr)
        
        print(f"✓ Generator output shape: {gen_output.shape}")
        print(f"✓ Discriminator output shape: {disc_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_disk_space():
    """Check available disk space"""
    print("\n" + "="*50)
    print("Checking Disk Space")
    print("="*50)
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        
        print(f"Total: {total / (2**30):.2f} GB")
        print(f"Used: {used / (2**30):.2f} GB")
        print(f"Free: {free / (2**30):.2f} GB")
        
        if free < 5 * (2**30):  # Less than 5 GB
            print("⚠️  Warning: Low disk space (< 5 GB)")
            return False
        else:
            print("✓ Sufficient disk space")
            return True
    except:
        print("⚠️  Could not check disk space")
        return True


def main():
    """Run all verification checks"""
    print("\n" + "="*70)
    print("SATELLITE-SRGAN SETUP VERIFICATION")
    print("="*70)
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch", check_pytorch),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Dataset", check_dataset),
        ("Models", check_models),
        ("Disk Space", check_disk_space)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error in {name} check: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status:10} - {name}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✓ All checks passed! You're ready to train.")
        print("\nNext steps:")
        print("  1. Train baselines: python scripts/train_baseline.py")
        print("  2. Train SRGAN: python scripts/train_srgan.py")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Prepare dataset: python scripts/prepare_data.py")
        print("  - Check file paths in config.py")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()