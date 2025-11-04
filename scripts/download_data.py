# scripts/download_data.py

import urllib.request
import zipfile
import os
from pathlib import Path

def download_and_extract_dataset():
    """Download and extract UC Merced dataset"""
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    
    # Download UC Merced dataset
    url = "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"
    zip_path = "data/raw/UCMerced_LandUse.zip"
    
    print("📥 Downloading dataset... (this may take 5-10 minutes)")
    urllib.request.urlretrieve(url, zip_path)
    
    # Extract
    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/raw/")
    
    print("✅ Dataset downloaded and extracted!")
    print(f"   Total classes: 21")
    print(f"   Images per class: 100")
    print(f"   Image size: 256x256 pixels")
    print(f"   Location: data/raw/UCMerced_LandUse/Images")

if __name__ == "__main__":
    download_and_extract_dataset()