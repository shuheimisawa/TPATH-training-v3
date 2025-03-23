import torch
import torch_directml  # Add DirectML import

# Flag to keep track of initialization
initialized = False
dml_device = None

def get_dml_device(device_id=0):
    """Get DirectML device for AMD GPU"""
    global dml_device, initialized
    if not initialized:
        try:
            dml_device = torch_directml.device(device_id)
            initialized = True
            return dml_device
        except Exception as e:
            print(f"Warning: Failed to initialize DirectML device: {e}")
            return torch.device('cpu')
    return dml_device

def to_device(tensor_or_module, device):
    """Move tensor or module to the specified device"""
    if hasattr(tensor_or_module, 'to'):
        return tensor_or_module.to(device)
    return tensor_or_module

def empty_cache():
    """Clear GPU memory cache (compatible interface with CUDA)"""
    # DirectML doesn't have a direct equivalent to torch.cuda.empty_cache()
    # This is a placeholder function for compatibility
    pass

def is_available():
    """Check if DirectML device is available"""
    try:
        get_dml_device()
        return True
    except Exception as e:
        print(f"Warning: DirectML not available: {e}")
        return False