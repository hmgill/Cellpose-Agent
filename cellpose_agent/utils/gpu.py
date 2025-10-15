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


def get_max_memory(memory_fraction=0.85, cpu_memory="50GB"):
    """
    Automatically configure max memory per GPU.
    
    When used with device_map="auto", this tells the model loader how much memory
    it CAN use per GPU during the INITIAL model loading phase. If a model's layers
    don't fit on one GPU with this limit, the loader will automatically split the
    model across multiple GPUs.
    
    Args:
        memory_fraction: Fraction of GPU memory to allocate (0.0-1.0). 
                        Default 0.85 leaves 15% headroom.
        cpu_memory: Maximum CPU memory to use as offload space.
    
    Returns:
        dict: Memory limits per device, or None if no CUDA available
    """
    if not torch.cuda.is_available():
        print("⚠ No CUDA GPUs available")
        return None
    
    max_memory = {}
    total_available = 0
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory
        usable_memory = int(total_memory * memory_fraction)
        max_memory[i] = usable_memory
        total_available += usable_memory
        
        print(f"GPU {i} ({props.name}): "
              f"{usable_memory / 1024**3:.2f}GB / {total_memory / 1024**3:.2f}GB "
              f"({memory_fraction*100:.0f}% limit)")
    
    # CPU memory for offloading if needed
    max_memory["cpu"] = cpu_memory
    
    print(f"✓ Total GPU memory available for models: {total_available / 1024**3:.2f}GB")
    print(f"✓ CPU offload memory: {cpu_memory}")
    
    return max_memory

def monitor_and_clear_cache(threshold=0.90):
    """
    Monitor GPU memory and clear cache if usage exceeds threshold.
    Call this periodically during long-running operations.
    
    Args:
        threshold: Memory usage fraction (0.0-1.0) that triggers cache clearing
    """
    if not torch.cuda.is_available():
        return
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i)
        total = props.total_memory
        usage = allocated / total
        
        if usage > threshold:
            print(f"⚠ GPU {i} usage at {usage*100:.1f}%, clearing cache...")
            torch.cuda.empty_cache()
            gc.collect()
