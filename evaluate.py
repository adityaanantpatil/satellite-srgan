"""
Comprehensive Baseline Evaluation Script
Evaluates all baseline models and generates comparison visualizations
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import json

# ==========================================================
# PATH FIX: Ensures imports work no matter where you run from
# ==========================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if os.path.join(ROOT_DIR, 'models') not in sys.path:
    sys.path.append(os.path.join(ROOT_DIR, 'models'))
if os.path.join(ROOT_DIR, 'models', 'saved_models') not in sys.path:
    sys.path.append(os.path.join(ROOT_DIR, 'models', 'saved_models'))
if os.path.join(ROOT_DIR, 'utils') not in sys.path:
    sys.path.append(os.path.join(ROOT_DIR, 'utils'))

# ==========================================================
# IMPORTS
# ==========================================================
from models.saved_models.baseline_models import BicubicUpsampler, SRCNN, ImprovedSRCNN
from utils.data_loader import get_dataloaders
from utils.metrics import calculate_psnr, calculate_ssim


# ==========================================================
# BASELINE EVALUATOR CLASS
# ==========================================================
class BaselineEvaluator:
    """Evaluator for all baseline models"""

    def __init__(self, test_loader, device='cuda', results_dir='results/baseline'):
        self.test_loader = test_loader
        self.device = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.models = {}
        self.results = {}

    def load_models(self):
        """Load all baseline models"""
        print("Loading baseline models...")

        # 1. Bicubic (no training needed)
        self.models['Bicubic'] = BicubicUpsampler(scale_factor=4)
        print("✓ Bicubic loaded")

        # 2. SRCNN
        srcnn = SRCNN(num_channels=3, scale_factor=4).to(self.device)
        checkpoint_path = os.path.join('checkpoints', 'srcnn', 'best_model.pth')

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            srcnn.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ SRCNN loaded from {checkpoint_path}")
        else:
            print(f"⚠ Warning: SRCNN checkpoint not found at {checkpoint_path}")
            print("  Using untrained SRCNN (for testing only)")

        srcnn.eval()
        self.models['SRCNN'] = srcnn

        # 3. Improved SRCNN (optional)
        improved_checkpoint = os.path.join('checkpoints', 'improved_srcnn', 'best_model.pth')
        if os.path.exists(improved_checkpoint):
            improved_srcnn = ImprovedSRCNN(num_channels=3, scale_factor=4).to(self.device)
            checkpoint = torch.load(improved_checkpoint, map_location=self.device)
            improved_srcnn.load_state_dict(checkpoint['model_state_dict'])
            improved_srcnn.eval()
            self.models['Improved SRCNN'] = improved_srcnn
            print(f"✓ Improved SRCNN loaded from {improved_checkpoint}")

        print(f"\nTotal models loaded: {len(self.models)}\n")

    def measure_inference_time(self, model, num_samples=100):
        """Measure average inference time"""
        times = []

        dummy_input = torch.randn(1, 3, 64, 64).to(self.device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        with torch.no_grad():
            for i, (lr_images, _) in enumerate(self.test_loader):
                if i >= num_samples // self.test_loader.batch_size:
                    break

                lr_images = lr_images.to(self.device)
                start_time = time.time()
                _ = model(lr_images)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()

                times.append((end_time - start_time) / lr_images.shape[0])

        return np.mean(times), np.std(times)

    def evaluate_model(self, model_name, model):
        """Evaluate a single model"""
        print(f"\n{'='*70}\nEvaluating: {model_name}\n{'='*70}")
        model.eval()

        all_psnr, all_ssim = [], []

        with torch.no_grad():
            for lr_images, hr_images in tqdm(self.test_loader, desc=f'{model_name}'):
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)
                sr_images = model(lr_images)

                for i in range(sr_images.shape[0]):
                    sr_img = sr_images[i].cpu().numpy()
                    hr_img = hr_images[i].cpu().numpy()
                    all_psnr.append(calculate_psnr(sr_img, hr_img, max_value=1.0))
                    all_ssim.append(calculate_ssim(sr_img, hr_img, max_value=1.0))

        avg_time, std_time = self.measure_inference_time(model)
        num_params = sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else 0

        results = {
            'model': model_name,
            'psnr_mean': np.mean(all_psnr),
            'psnr_std': np.std(all_psnr),
            'ssim_mean': np.mean(all_ssim),
            'ssim_std': np.std(all_ssim),
            'inference_time_mean': avg_time,
            'inference_time_std': std_time,
            'num_parameters': num_params,
            'num_images': len(all_psnr)
        }

        print(f"\nResults:")
        print(f"  PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
        print(f"  SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
        print(f"  Inference Time: {avg_time*1000:.1f} ± {std_time*1000:.1f} ms")
        print(f"  Parameters: {num_params:,}")

        return results

    def evaluate_all(self):
        """Run evaluation for all baseline models"""
        self.load_models()

        print("\n" + "="*70)
        print("BASELINE MODEL EVALUATION")
        print("="*70)

        for model_name, model in self.models.items():
            self.results[model_name] = self.evaluate_model(model_name, model)

        self.save_results()
        self.print_comparison_table()
        self.plot_comparison_charts()

    def save_results(self):
        """Save evaluation results"""
        df = pd.DataFrame(self.results).T
        os.makedirs(self.results_dir, exist_ok=True)
        csv_path = os.path.join(self.results_dir, 'evaluation_results.csv')
        df.to_csv(csv_path)
        print(f"\n✓ Results saved to {csv_path}")

        json_path = os.path.join(self.results_dir, 'evaluation_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"✓ Results saved to {json_path}")

    def print_comparison_table(self):
        """Print formatted comparison table"""
        df = pd.DataFrame(self.results).T.sort_values('psnr_mean', ascending=False)

        print("\n" + "="*70)
        print("BASELINE MODEL COMPARISON")
        print("="*70)

        display_df = pd.DataFrame([{
            'Model': idx,
            'PSNR (dB)': f"{row['psnr_mean']:.2f}",
            'SSIM': f"{row['ssim_mean']:.4f}",
            'Time (ms)': f"{row['inference_time_mean']*1000:.1f}",
            'Params': f"{int(row['num_parameters']):,}"
        } for idx, row in df.iterrows()])

        print(display_df.to_string(index=False))
        print("="*70)

        table_path = os.path.join(self.results_dir, 'comparison_table.txt')
        with open(table_path, 'w') as f:
            f.write(display_df.to_string(index=False))

        best_model = df['psnr_mean'].idxmax()
        print(f"\n🏆 Best Model: {best_model}")
        print("="*70 + "\n")

    def plot_comparison_charts(self):
        """Generate comparison charts"""
        df = pd.DataFrame(self.results).T.sort_values('psnr_mean', ascending=False)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # PSNR
        ax[0].bar(df.index, df['psnr_mean'], color='#3498db')
        ax[0].set_title('PSNR Comparison')
        ax[0].set_ylabel('PSNR (dB)')
        ax[0].grid(axis='y', alpha=0.3)

        # SSIM
        ax[1].bar(df.index, df['ssim_mean'], color='#2ecc71')
        ax[1].set_title('SSIM Comparison')
        ax[1].set_ylabel('SSIM')
        ax[1].grid(axis='y', alpha=0.3)

        plt.suptitle('Baseline Model Metrics Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        fig_path = os.path.join(self.results_dir, 'baseline_comparison.png')
        plt.savefig(fig_path, dpi=300)
        print(f"✓ Comparison chart saved to {fig_path}\n")
        plt.close()


# ==========================================================
# MAIN FUNCTION
# ==========================================================
def main():
    """Main evaluation function"""

    BATCH_SIZE = 16
    SCALE_FACTOR = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nLoading test data...")
    _, _, test_loader = get_dataloaders(data_dir='data', batch_size=BATCH_SIZE, scale_factor=SCALE_FACTOR)
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    print(f"✓ Test batches: {len(test_loader)}")

    evaluator = BaselineEvaluator(test_loader=test_loader, device=device, results_dir='results/baseline')
    evaluator.evaluate_all()

    print("\n" + "="*70)
    print("✓ BASELINE EVALUATION COMPLETE!")
    print("="*70)
    print("Results saved to: results/baseline/")
    print("  - evaluation_results.csv")
    print("  - evaluation_results.json")
    print("  - comparison_table.txt")
    print("  - baseline_comparison.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
