"""
Training utilities for monitoring, checkpointing, and early stopping
"""

import torch
import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stopping to stop training when metric stops improving"""
    def __init__(self, patience=10, mode='max', min_delta=0):
        """
        Args:
            patience: How many epochs to wait before stopping
            mode: 'max' or 'min' - maximize or minimize the metric
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop


class MetricsTracker:
    """Track and log training metrics"""
    def __init__(self, metrics_names):
        self.metrics = {name: [] for name in metrics_names}
        self.current_epoch_metrics = {}
        
    def update(self, **kwargs):
        """Update current epoch metrics"""
        self.current_epoch_metrics.update(kwargs)
    
    def end_epoch(self):
        """Save current epoch metrics and reset"""
        for key, value in self.current_epoch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        self.current_epoch_metrics = {}
    
    def get_latest(self, metric_name):
        """Get latest value of a metric"""
        if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
            return self.metrics[metric_name][-1]
        return None
    
    def get_best(self, metric_name, mode='max'):
        """Get best value of a metric"""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return None
        
        values = self.metrics[metric_name]
        if mode == 'max':
            return max(values)
        else:
            return min(values)
    
    def save(self, save_path):
        """Save metrics to JSON file"""
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def load(self, load_path):
        """Load metrics from JSON file"""
        with open(load_path, 'r') as f:
            self.metrics = json.load(f)


class CheckpointManager:
    """Manage model checkpoints"""
    def __init__(self, checkpoint_dir, max_to_keep=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.checkpoints = []
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, generator, discriminator, g_optimizer, d_optimizer, 
             epoch, metrics, is_best=False):
        """Save checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': timestamp
        }
        
        # Save regular checkpoint
        filename = f'checkpoint_epoch_{epoch}_{timestamp}.pth'
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        self.checkpoints.append(filepath)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")
        
        # Clean old checkpoints
        self._cleanup_old_checkpoints()
        
        return filepath
    
    def load(self, filepath, generator, discriminator, g_optimizer=None, 
             d_optimizer=None):
        """Load checkpoint"""
        checkpoint = torch.load(filepath)
        
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        if g_optimizer is not None:
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        if d_optimizer is not None:
            d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint.get('metrics', {})
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keep only max_to_keep recent ones"""
        if len(self.checkpoints) > self.max_to_keep:
            # Sort by modification time
            self.checkpoints.sort(key=lambda x: os.path.getmtime(x))
            
            # Remove oldest
            for old_checkpoint in self.checkpoints[:-self.max_to_keep]:
                if os.path.exists(old_checkpoint) and 'best' not in old_checkpoint:
                    os.remove(old_checkpoint)
                    print(f"Removed old checkpoint: {old_checkpoint}")
            
            self.checkpoints = self.checkpoints[-self.max_to_keep:]


class TrainingLogger:
    """Log training progress"""
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Write header
        with open(self.log_file, 'w') as f:
            f.write("Training Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message, print_also=True):
        """Write log message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
        
        if print_also:
            print(message)
    
    def log_epoch(self, epoch, metrics):
        """Log epoch metrics"""
        message = f"\nEpoch {epoch} Summary:"
        for key, value in metrics.items():
            if isinstance(value, float):
                message += f"\n  {key}: {value:.4f}"
            else:
                message += f"\n  {key}: {value}"
        
        self.log(message)


def plot_loss_curves(history, save_path=None):
    """Plot training loss curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generator loss
    axes[0, 0].plot(history['g_loss'], label='Generator Loss', color='blue')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Discriminator loss
    axes[0, 1].plot(history['d_loss'], label='Discriminator Loss', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # PSNR
    if 'psnr' in history:
        axes[1, 0].plot(history['psnr'], label='PSNR', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].set_title('Peak Signal-to-Noise Ratio')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # SSIM
    if 'ssim' in history:
        axes[1, 1].plot(history['ssim'], label='SSIM', color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].set_title('Structural Similarity Index')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Loss curves saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def print_model_summary(model, model_name="Model"):
    """Print model summary"""
    print(f"\n{model_name} Summary:")
    print("="*50)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {calculate_model_size(model):.2f} MB")
    print("="*50 + "\n")


def estimate_training_time(num_epochs, time_per_epoch):
    """Estimate remaining training time"""
    total_seconds = num_epochs * time_per_epoch
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    return f"{hours}h {minutes}m"


if __name__ == "__main__":
    # Test early stopping
    early_stopping = EarlyStopping(patience=5, mode='max')
    
    scores = [0.5, 0.6, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60]
    for i, score in enumerate(scores):
        if early_stopping(score):
            print(f"Early stopping triggered at iteration {i}")
            break
        print(f"Iteration {i}: Score = {score}, Counter = {early_stopping.counter}")