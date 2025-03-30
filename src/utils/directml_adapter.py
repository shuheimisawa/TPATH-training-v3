import torch
import sys
import importlib.util

# Check if torch_directml is available
_has_directml = importlib.util.find_spec("torch_directml") is not None

# Flag to keep track of initialization
initialized = False
dml_device = None

def get_dml_device(device_id=0):
    """Get DirectML device for AMD GPU"""
    global dml_device, initialized, _has_directml
    
    if not _has_directml:
        print("DirectML not found, falling back to CPU")
        return torch.device('cpu')
    
    if not initialized:
        try:
            import torch_directml
            dml_device = torch_directml.device(device_id)
            initialized = True
            print(f"Successfully initialized DirectML device: {device_id}")
            return dml_device
        except Exception as e:
            print(f"Warning: Failed to initialize DirectML device: {e}")
            print("Falling back to CPU device")
            return torch.device('cpu')
    return dml_device

def to_device(tensor_or_module, device):
    """Move tensor or module to the specified device"""
    try:
        if hasattr(tensor_or_module, 'to'):
            return tensor_or_module.to(device)
        return tensor_or_module
    except Exception as e:
        print(f"Error moving to device: {e}")
        # If we fail to move to the requested device, try CPU as fallback
        if str(device) != 'cpu':
            print("Falling back to CPU")
            try:
                if hasattr(tensor_or_module, 'to'):
                    return tensor_or_module.to('cpu')
            except:
                pass
        return tensor_or_module

def empty_cache():
    """Clear GPU memory cache (compatible interface with CUDA)"""
    try:
        # DirectML doesn't have a direct equivalent to torch.cuda.empty_cache()
        # This is a placeholder function for compatibility
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()  # Run garbage collector to help free memory
    except Exception as e:
        print(f"Warning: Error clearing cache: {e}")

def is_available():
    """Check if DirectML device is available"""
    global _has_directml
    if not _has_directml:
        return False
        
    try:
        device = get_dml_device()
        # Test if device works by creating a small tensor
        test_tensor = torch.zeros(1, device=device)
        del test_tensor  # Clean up test tensor
        return True
    except Exception as e:
        print(f"Warning: DirectML not available: {e}")
        return False

def get_device_info():
    """Get information about the DirectML device"""
    global _has_directml
    if not _has_directml:
        return {"status": "DirectML not installed"}
        
    try:
        import torch_directml
        device = get_dml_device()
        
        # Get basic device information
        info = {
            "status": "available",
            "device_id": str(device),
            "directml_version": torch_directml.__version__ if hasattr(torch_directml, "__version__") else "unknown"
        }
        
        # Try to get more detailed information if available
        try:
            # This might not work on all versions of torch_directml
            info["name"] = torch_directml.device_name(device)
        except:
            pass
            
        return info
    except Exception as e:
        return {"status": f"Error: {str(e)}"}