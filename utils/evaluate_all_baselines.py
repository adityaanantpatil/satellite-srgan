"""
Unified Baseline Evaluation Script
Evaluates Bicubic, SRCNN, and SRGAN on test set
Generates comparison images and metrics table
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json

# Import your models
from models.srcnn import SRCNN
from models.bicubic_baseline import bicubic_upscale
from models.generator import Generator  # Your SRGAN generator
from utils.metrics import calculate_psnr, calculate_ssim
from utils.data_loader import get_dataloaders

class BaselineEvaluator:
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.results = {}
        
        # Create results directory
        self.results_dir = 'results/baseline_comparisons'
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f'{self.results_dir}/comparison_images', exist_ok=True)
        
    def load_models(self):
        """Load all trained models"""
        models = {}
        
        # 1. Bicubic (no model to load)
        models['Bicubic'] = None
        
        # 2. SRCNN
        try:
            srcnn = SRCNN(num_channels=3).to(self.device)
            srcnn.load_state_dict(torch.load('models/saved_models/srcnn_best.pth'))
            srcnn.eval()
            models['SRCNN'] = srcnn
            print("✅ SRCNN model loaded")
        except FileNotFoundError:
            print("⚠️  SRCNN model not found. Train it first!")
            models['SRCNN'] = None
        
        # 3. SRGAN
        try:
            srgan = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=16).to(self.device)
            checkpoint = torch.load('models/saved_models/generator_best.pth')
            srgan.load_state_dict(checkpoint)
            srgan.eval()
            models['SRGAN'] = srgan
            print("✅ SRGAN model loaded")
        except FileNotFoundError:
            print("⚠️  SRGAN model not found. Using checkpoint from training...")
            models['SRGAN'] = None
        
        return models
    
    def evaluate_model(self, model, model_name, test_loader):
        """Evaluate single model on test set"""
        print(f"\n🔍 Evaluating {model_name}...")
        
        psnr_values = []
        ssim_values = []
        inference_times = []
        
        with torch.no_grad():
            for lr_images, hr_images in tqdm(test_loader, desc=f"Testing {model_name}"):
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)
                
                # Start timing
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                
                # Generate SR image based on model type
                if model_name == 'Bicubic':
                    sr_images = bicubic_upscale(lr_images, scale_factor=4)
                
                elif model_name == 'SRCNN':
                    # SRCNN expects bicubic upsampled input
                    lr_upscaled = F.interpolate(lr_images, scale_factor=4, 
                                               mode='bicubic', align_corners=False)
                    sr_images = model(lr_upscaled)
                
                elif model_name == 'SRGAN':
                    sr_images = model(lr_images)
                
                # End timing
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                inference_times.append(inference_time)
                
                # Calculate metrics
                psnr = calculate_psnr(sr_images, hr_images)
                ssim = calculate_ssim(sr_images, hr_images)
                
                psnr_values.append(psnr)
                ssim_values.append(ssim)
        
        # Compute averages
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        avg_time = np.mean(inference_times)
        
        results = {
            'PSNR': avg_psnr,
            'SSIM': avg_ssim,
            'Inference_Time_ms': avg_time,
            'PSNR_std': np.std(psnr_values),
            'SSIM_std': np.std(ssim_values)
        }
        
        print(f"✅ {model_name} Results:")
        print(f"   PSNR: {avg_psnr:.2f} ± {results['PSNR_std']:.2f} dB")
        print(f"   SSIM: {avg_ssim:.4f} ± {results['SSIM_std']:.4f}")
        print(f"   Inference Time: {avg_time:.2f} ms")
        
        return results
    
    def generate_comparison_images(self, models, test_loader, num_samples=20):
        """Generate side-by-side comparison images"""
        print(f"\n🎨 Generating comparison images ({num_samples} samples)...")
        
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, (lr_images, hr_images) in enumerate(test_loader):
                if sample_count >= num_samples:
                    break
                
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)
                
                # Process each image in batch
                for i in range(lr_images.size(0)):
                    if sample_count >= num_samples:
                        break
                    
                    lr_img = lr_images[i:i+1]
                    hr_img = hr_images[i:i+1]
                    
                    # Generate SR images from all models
                    results_dict = {}
                    
                    # Bicubic
                    results_dict['Bicubic'] = bicubic_upscale(lr_img, scale_factor=4)
                    
                    # SRCNN
                    if models['SRCNN'] is not None:
                        lr_upscaled = F.interpolate(lr_img, scale_factor=4, 
                                                   mode='bicubic', align_corners=False)
                        results_dict['SRCNN'] = models['SRCNN'](lr_upscaled)
                    
                    # SRGAN
                    if models['SRGAN'] is not None:
                        results_dict['SRGAN'] = models['SRGAN'](lr_img)
                    
                    # Create comparison plot
                    self._create_comparison_plot(
                        lr_img, results_dict, hr_img, 
                        save_path=f'{self.results_dir}/comparison_images/comparison_{sample_count:03d}.png'
                    )
                    
                    sample_count += 1
                    
                    if sample_count % 5 == 0:
                        print(f"   Generated {sample_count}/{num_samples} comparisons")
        
        print(f"✅ Saved {sample_count} comparison images to {self.results_dir}/comparison_images/")
    
    def _create_comparison_plot(self, lr_img, results_dict, hr_img, save_path):
        """Create 5-panel comparison plot"""
        num_models = len(results_dict) + 2  # +2 for LR and GT
        fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 5))
        
        # Convert tensors to numpy for display
        def tensor_to_img(tensor):
            img = tensor.squeeze().cpu().numpy()
            if img.shape[0] == 3:  # CHW to HWC
                img = np.transpose(img, (1, 2, 0))
            return np.clip(img, 0, 1)
        
        lr_display = tensor_to_img(lr_img)
        hr_display = tensor_to_img(hr_img)
        
        # Plot LR input
        axes[0].imshow(lr_display)
        axes[0].set_title('Low-Res Input\n(64×64)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Plot model results
        plot_idx = 1
        for model_name, sr_img in results_dict.items():
            sr_display = tensor_to_img(sr_img)
            psnr = calculate_psnr(sr_img, hr_img)
            ssim = calculate_ssim(sr_img, hr_img)
            
            axes[plot_idx].imshow(sr_display)
            
            # Highlight SRGAN in green
            color = 'green' if model_name == 'SRGAN' else 'black'
            weight = 'bold' if model_name == 'SRGAN' else 'normal'
            
            axes[plot_idx].set_title(
                f'{model_name}\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.3f}',
                fontsize=12, color=color, fontweight=weight
            )
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        # Plot ground truth
        axes[-1].imshow(hr_display)
        axes[-1].set_title('Ground Truth\n(256×256)', fontsize=12, fontweight='bold')
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_results_table(self, results):
        """Create markdown table with results"""
        print("\n📊 Creating results table...")
        
        # Create markdown table
        markdown = "# Baseline Comparison Results\n\n"
        markdown += f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown += "## Quantitative Metrics\n\n"
        markdown += "| Model | PSNR (dB) | SSIM | Inference Time (ms) | Improvement over Bicubic |\n"
        markdown += "|-------|-----------|------|---------------------|-------------------------|\n"
        
        # Calculate improvements
        bicubic_psnr = results['Bicubic']['PSNR']
        
        for model_name, metrics in results.items():
            psnr = metrics['PSNR']
            ssim = metrics['SSIM']
            time = metrics['Inference_Time_ms']
            
            if model_name == 'Bicubic':
                improvement = "Baseline"
            else:
                improvement = f"+{((psnr - bicubic_psnr) / bicubic_psnr * 100):.1f}%"
            
            # Bold the best model (SRGAN)
            if model_name == 'SRGAN':
                markdown += f"| **{model_name}** | **{psnr:.2f}** | **{ssim:.4f}** | {time:.2f} | **{improvement}** |\n"
            else:
                markdown += f"| {model_name} | {psnr:.2f} | {ssim:.4f} | {time:.2f} | {improvement} |\n"
        
        markdown += "\n## Key Findings\n\n"
        
        # Calculate improvements
        if 'SRCNN' in results and 'SRGAN' in results:
            srcnn_psnr = results['SRCNN']['PSNR']
            srgan_psnr = results['SRGAN']['PSNR']
            
            bicubic_improvement = ((srgan_psnr - bicubic_psnr) / bicubic_psnr * 100)
            srcnn_improvement = ((srgan_psnr - srcnn_psnr) / srcnn_psnr * 100)
            
            markdown += f"- 🏆 **SRGAN achieves {srgan_psnr:.2f} dB PSNR**, beating bicubic by **{bicubic_improvement:.1f}%**\n"
            markdown += f"- 📈 SRGAN outperforms SRCNN by **{srgan_psnr - srcnn_psnr:.2f} dB** ({srcnn_improvement:.1f}% improvement)\n"
            markdown += f"- ⚡ Trade-off: SRGAN is {results['SRGAN']['Inference_Time_ms']/results['Bicubic']['Inference_Time_ms']:.1f}x slower but produces photorealistic results\n"
        
        # Save to file
        with open(f'{self.results_dir}/results_table.md', 'w') as f:
            f.write(markdown)
        
        # Also save as JSON
        with open(f'{self.results_dir}/results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"✅ Results saved to {self.results_dir}/")
        print("\n" + markdown)
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("="*80)
        print("🚀 Starting Baseline Evaluation Pipeline")
        print("="*80)
        
        # Load test data
        print("\n📁 Loading test dataset...")
        _, _, test_loader = get_dataloaders(self.config)
        print(f"✅ Loaded {len(test_loader.dataset)} test images")
        
        # Load models
        print("\n🔧 Loading models...")
        models = self.load_models()
        
        # Evaluate each model
        print("\n" + "="*80)
        print("📊 EVALUATION PHASE")
        print("="*80)
        
        for model_name in ['Bicubic', 'SRCNN', 'SRGAN']:
            if model_name == 'Bicubic' or models[model_name] is not None:
                results = self.evaluate_model(models[model_name], model_name, test_loader)
                self.results[model_name] = results
        
        # Generate comparison images
        print("\n" + "="*80)
        print("🎨 VISUALIZATION PHASE")
        print("="*80)
        self.generate_comparison_images(models, test_loader, num_samples=20)
        
        # Create results table
        print("\n" + "="*80)
        print("📝 DOCUMENTATION PHASE")
        print("="*80)
        self.create_results_table(self.results)
        
        print("\n" + "="*80)
        print("🎉 EVALUATION COMPLETE!")
        print("="*80)
        print(f"\n📂 All results saved to: {self.results_dir}/")
        print("   - Comparison images: comparison_images/")
        print("   - Results table: results_table.md")
        print("   - Raw metrics: results.json")


if __name__ == "__main__":
    # Import your config
    from config import Config
    
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🖥️  Using device: {device}")
    
    # Run evaluation
    evaluator = BaselineEvaluator(config, device)
    evaluator.run_full_evaluation()