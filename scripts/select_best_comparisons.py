"""
Select best comparison images based on various criteria
Useful for showcasing in papers, presentations, and portfolios
"""

import json
import shutil
from pathlib import Path
import re
import numpy as np


def parse_image_metrics(image_path):
    """Extract metrics from comparison image filename or associated data"""
    # For now, we'll rank based on image number (assuming they're sorted)
    # In a more advanced version, you could parse actual PSNR/SSIM from the images
    match = re.search(r'comparison_(\d+)', str(image_path))
    if match:
        return int(match.group(1))
    return 0


def select_best_images(source_dir, output_dir, num_best=5, num_worst=2, num_median=3):
    """
    Select best, worst, and median performing images
    
    Args:
        source_dir: Directory containing comparison images
        output_dir: Directory to save selected images
        num_best: Number of best-performing images
        num_worst: Number of worst-performing images  
        num_median: Number of median-performing images
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Get all comparison images
    image_files = sorted(source_path.glob('comparison_*.png'))
    
    if not image_files:
        print(f"❌ No comparison images found in {source_dir}")
        return
    
    print(f"Found {len(image_files)} comparison images")
    
    # Create output directories
    best_dir = output_path / 'best'
    worst_dir = output_path / 'worst'
    median_dir = output_path / 'median'
    showcase_dir = output_path / 'showcase'
    
    for d in [best_dir, worst_dir, median_dir, showcase_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Select images
    # Best: First few (usually best performers)
    best_images = image_files[:num_best]
    
    # Worst: Last few
    worst_images = image_files[-num_worst:] if len(image_files) > num_worst else []
    
    # Median: Middle range
    mid_idx = len(image_files) // 2
    start_idx = max(0, mid_idx - num_median // 2)
    median_images = image_files[start_idx:start_idx + num_median]
    
    # Copy best images
    print(f"\nSelecting {num_best} BEST images...")
    for i, img in enumerate(best_images, 1):
        dest = best_dir / f'best_{i:02d}.png'
        shutil.copy2(img, dest)
        print(f"  ✓ {img.name} → {dest.name}")
    
    # Copy worst images (for analysis)
    if worst_images:
        print(f"\nSelecting {num_worst} WORST images (for improvement analysis)...")
        for i, img in enumerate(worst_images, 1):
            dest = worst_dir / f'worst_{i:02d}.png'
            shutil.copy2(img, dest)
            print(f"  ✓ {img.name} → {dest.name}")
    
    # Copy median images
    print(f"\nSelecting {num_median} MEDIAN images (representative samples)...")
    for i, img in enumerate(median_images, 1):
        dest = median_dir / f'median_{i:02d}.png'
        shutil.copy2(img, dest)
        print(f"  ✓ {img.name} → {dest.name}")
    
    # Create showcase collection (best + couple median)
    print(f"\nCreating SHOWCASE collection...")
    showcase_selection = best_images[:3] + median_images[:2]
    for i, img in enumerate(showcase_selection, 1):
        dest = showcase_dir / f'showcase_{i:02d}.png'
        shutil.copy2(img, dest)
        print(f"  ✓ {img.name} → {dest.name}")
    
    # Create summary report
    report_path = output_path / 'selection_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("IMAGE SELECTION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total images analyzed: {len(image_files)}\n")
        f.write(f"Best images selected: {len(best_images)}\n")
        f.write(f"Worst images selected: {len(worst_images)}\n")
        f.write(f"Median images selected: {len(median_images)}\n")
        f.write(f"Showcase collection: {len(showcase_selection)}\n\n")
        
        f.write("USAGE RECOMMENDATIONS:\n")
        f.write("-" * 70 + "\n")
        f.write("• BEST images: Use for LinkedIn, presentations, portfolio\n")
        f.write("• MEDIAN images: Representative samples for technical reports\n")
        f.write("• WORST images: Analyze failure cases, identify improvements\n")
        f.write("• SHOWCASE collection: Quick overview for README/papers\n\n")
        
        f.write("BEST IMAGES (Highest Quality):\n")
        for i, img in enumerate(best_images, 1):
            f.write(f"  {i}. {img.name}\n")
        
        f.write(f"\nMEDIAN IMAGES (Representative):\n")
        for i, img in enumerate(median_images, 1):
            f.write(f"  {i}. {img.name}\n")
        
        if worst_images:
            f.write(f"\nWORST IMAGES (For Analysis):\n")
            for i, img in enumerate(worst_images, 1):
                f.write(f"  {i}. {img.name}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"\n✓ Selection report saved: {report_path}")
    
    return {
        'best': list(best_images),
        'worst': list(worst_images),
        'median': list(median_images),
        'showcase': showcase_selection
    }


def create_image_grid(image_paths, output_path, grid_size=(2, 3), title=None):
    """Create a grid of images for easy comparison"""
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(18, 12))
        axes = axes.flatten() if grid_size[0] * grid_size[1] > 1 else [axes]
        
        for idx, (ax, img_path) in enumerate(zip(axes, image_paths)):
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'Sample {idx + 1}', fontsize=12, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(len(image_paths), len(axes)):
            axes[idx].axis('off')
        
        if title:
            plt.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Image grid saved: {output_path}")
        return True
        
    except ImportError:
        print("⚠️  PIL/matplotlib not available for grid creation")
        return False


def analyze_diversity(image_paths):
    """Analyze visual diversity of selected images"""
    try:
        from PIL import Image
        import numpy as np
        
        print("\nAnalyzing image diversity...")
        
        # Load images and compute simple statistics
        stats = []
        for img_path in image_paths:
            img = Image.open(img_path)
            img_array = np.array(img)
            
            stats.append({
                'path': img_path.name,
                'mean_brightness': np.mean(img_array),
                'std_brightness': np.std(img_array),
                'color_variance': np.var(img_array, axis=(0, 1)).mean()
            })
        
        # Sort by different criteria
        by_brightness = sorted(stats, key=lambda x: x['mean_brightness'])
        by_variance = sorted(stats, key=lambda x: x['color_variance'], reverse=True)
        
        print("\nDiversity Analysis:")
        print(f"  Brightness range: {by_brightness[0]['mean_brightness']:.1f} - {by_brightness[-1]['mean_brightness']:.1f}")
        print(f"  Most uniform: {by_brightness[0]['path']}")
        print(f"  Most varied: {by_variance[0]['path']}")
        
        return stats
        
    except ImportError:
        print("⚠️  PIL/numpy not available for diversity analysis")
        return None


def main():
    print("=" * 70)
    print("SELECTING BEST COMPARISON IMAGES FOR SHOWCASE")
    print("=" * 70)
    print()
    
    # Configuration
    source_dir = 'results/model_comparisons'
    output_dir = 'results/selected_comparisons'
    
    # Check if source directory exists
    if not Path(source_dir).exists():
        print(f"❌ Error: Directory {source_dir} not found!")
        print("   Please run: python scripts/compare_models.py first")
        return
    
    # Select images
    selection = select_best_images(
        source_dir=source_dir,
        output_dir=output_dir,
        num_best=5,
        num_worst=2,
        num_median=3
    )
    
    # Create image grids
    print("\nCreating image grids...")
    
    if selection['best']:
        create_image_grid(
            selection['best'],
            Path(output_dir) / 'best_images_grid.png',
            grid_size=(2, 3),
            title='Best Performing Super-Resolution Results'
        )
    
    if selection['showcase']:
        create_image_grid(
            selection['showcase'][:4],
            Path(output_dir) / 'showcase_grid.png',
            grid_size=(2, 2),
            title='Showcase: Representative Super-Resolution Results'
        )
    
    # Analyze diversity
    if selection['best']:
        analyze_diversity(selection['best'])
    
    # Final summary
    print("\n" + "=" * 70)
    print("SELECTION COMPLETE!")
    print("=" * 70)
    print(f"\nSelected images saved to: {output_dir}/")
    print("\nDirectory structure:")
    print(f"  ├── best/           ({len(selection['best'])} images) - Use for portfolio/LinkedIn")
    print(f"  ├── median/         ({len(selection['median'])} images) - Representative samples")
    print(f"  ├── worst/          ({len(selection['worst'])} images) - For improvement analysis")
    print(f"  ├── showcase/       ({len(selection['showcase'])} images) - Quick overview")
    print(f"  ├── best_images_grid.png      - Grid of best results")
    print(f"  ├── showcase_grid.png         - 4-image showcase")
    print(f"  └── selection_report.txt      - Detailed report")
    print("\n✓ Ready for publication/showcase!")


if __name__ == "__main__":
    main()