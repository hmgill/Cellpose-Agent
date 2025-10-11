"""

"""

import torch
import gc

def clear_gpu_cache():
    """Frees up GPU memory by clearing cache and collecting garbage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print("✓ GPU cache cleared.")
