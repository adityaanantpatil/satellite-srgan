"""
Debug script to check what's in the raw data directory
"""

from pathlib import Path

raw_dir = Path('data/raw')

print(f"Checking directory: {raw_dir.absolute()}")
print(f"Directory exists: {raw_dir.exists()}")
print()

if raw_dir.exists():
    print("Contents of data/raw:")
    print("-" * 80)
    
    all_items = list(raw_dir.rglob('*'))
    print(f"Total items found (recursive): {len(all_items)}")
    print()
    
    # Show first 20 items
    print("First 20 items:")
    for i, item in enumerate(all_items[:20]):
        item_type = "DIR" if item.is_dir() else "FILE"
        print(f"{item_type}: {item.relative_to(raw_dir)}")
    
    print()
    
    # Check for different extensions
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.PNG', '.JPG', '.JPEG', '.TIF', '.TIFF']
    
    print("Searching for images with different extensions:")
    for ext in extensions:
        found = list(raw_dir.rglob(f'*{ext}'))
        if found:
            print(f"  {ext}: {len(found)} files")
            print(f"    Example: {found[0].relative_to(raw_dir)}")
    
    print()
    
    # List all unique extensions
    all_files = [f for f in raw_dir.rglob('*') if f.is_file()]
    unique_extensions = set(f.suffix for f in all_files)
    print(f"All unique file extensions found: {sorted(unique_extensions)}")
    print(f"Total files: {len(all_files)}")
else:
    print("Directory does not exist!")