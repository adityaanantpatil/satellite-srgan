# save_history_from_checkpoint.py
import torch
import json
import numpy as np
from datetime import datetime
import os

def convert_to_native(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj

# Load the best checkpoint (with weights_only=False to load history)
checkpoint = torch.load('checkpoints/srgan/best.pth', map_location='cpu', weights_only=False)
history = checkpoint['history']

# Save history to JSON
history_dir = 'results/training_history'
os.makedirs(history_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
path = os.path.join(history_dir, f'srgan_history_{timestamp}.json')

with open(path, 'w') as f:
    json.dump(convert_to_native(history), f, indent=4)

print(f"✓ Training history saved: {path}")
print(f"✓ Best PSNR: {max(history['val_psnr']):.2f}dB")
print(f"✓ Total epochs: {len(history['val_psnr'])}")