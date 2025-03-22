# src/training/optimization.py
import torch
import torch.optim as optim
from typing import Dict, List, Any, Callable


def create_optimizer(model_parameters, config: Dict) -> torch.optim.Optimizer:
    """Create an optimizer based on configuration.
    
    Args:
        model_parameters: Model parameters to optimize
        config: Optimizer configuration
        
    Returns:
        Optimizer instance
    """
    optimizer_type = config.get('type', 'sgd').lower()
    lr = config.get('lr', 0.01)
    weight_decay = config.get('weight_decay', 0.0001)
    
    if optimizer_type == 'sgd':
        momentum = config.get('momentum', 0.9)
        return optim.SGD(
            model_parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adam':
        return optim.Adam(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        return optim.AdamW(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_lr_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> Any:
    """Create a learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer instance
        config: LR scheduler configuration
        
    Returns:
        Learning rate scheduler instance
    """
    scheduler_type = config.get('type', 'step').lower()
    
    if scheduler_type == 'step':
        step_size = config.get('step_size', 10)
        gamma = config.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_type == 'multistep':
        milestones = config.get('milestones', [30, 60, 90])
        gamma = config.get('gamma', 0.1)
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
    elif scheduler_type == 'cosine':
        t_max = config.get('t_max', 100)
        eta_min = config.get('eta_min', 0)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min
        )
    elif scheduler_type == 'plateau':
        mode = config.get('mode', 'min')
        factor = config.get('factor', 0.1)
        patience = config.get('patience', 10)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience
        )
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


