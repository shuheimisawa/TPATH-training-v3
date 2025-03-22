import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Callable, Any, List, Optional


def setup_distributed(rank: int, world_size: int) -> None:
    """Set up distributed training.
    
    Args:
        rank: Rank of the current process
        world_size: Number of processes
    """
    # Use NCCL if available, else use Gloo
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    dist.destroy_process_group()


def run_distributed(
    fn: Callable,
    world_size: int,
    *args: Any,
    **kwargs: Any
) -> None:
    """Run a function in distributed mode.
    
    Args:
        fn: Function to run
        world_size: Number of processes
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    """
    mp.spawn(
        _distributed_worker,
        args=(fn, world_size, args, kwargs),
        nprocs=world_size,
        join=True
    )


def _distributed_worker(
    rank: int,
    fn: Callable,
    world_size: int,
    args: tuple,
    kwargs: dict
) -> None:
    """Worker function for distributed training.
    
    Args:
        rank: Rank of the current process
        fn: Function to run
        world_size: Number of processes
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
    """
    # Set up distributed environment
    setup_distributed(rank, world_size)
    
    # Get device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Add rank and device to kwargs
    kwargs['rank'] = rank
    kwargs['device'] = device
    
    # Run function
    fn(*args, **kwargs)
    
    # Clean up
    cleanup_distributed()


def is_main_process() -> bool:
    """Check if this is the main process in distributed training.
    
    Returns:
        True if this is the main process or not in distributed mode
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size() -> int:
    """Get the number of processes in distributed training.
    
    Returns:
        Number of processes or 1 if not in distributed mode
    """
    return dist.get_world_size() if dist.is_initialized() else 1


def all_gather(data: torch.Tensor) -> List[torch.Tensor]:
    """Gather data from all processes.
    
    Args:
        data: Tensor to gather
        
    Returns:
        List of gathered tensors
    """
    if not dist.is_initialized():
        return [data]
    
    world_size = dist.get_world_size()
    gathered_data = [torch.zeros_like(data) for _ in range(world_size)]
    dist.all_gather(gathered_data, data)
    
    return gathered_data