# Addition to src/utils/stain_normalization.py

from sklearn.decomposition import NMF
import numpy as np
import cv2
from typing import Dict, Optional, Union, Tuple, Any
import warnings
import os


class VahadaneNormalizer:
    """
    Stain normalization using Vahadane's method.
    
    Reference:
    Vahadane, A., et al. (2016). Structure-Preserving Color Normalization and
    Sparse Stain Separation for Histological Images. IEEE TMI, 35(8), 1962-1971.
    """
    
    def __init__(self, 
                 target_stains: Optional[np.ndarray] = None,
                 n_stains: int = 2,
                 lambda1: float = 0.1,
                 lambda2: float = 0.1):
        """
        Initialize Vahadane stain normalizer.
        
        Args:
            target_stains: Target stain matrix
            n_stains: Number of stains to separate (typically 2 for H&E)
            lambda1: Sparsity regularization parameter for stain separation
            lambda2: Sparsity regularization parameter for concentrations
        """
        self.target_stains = target_stains
        self.n_stains = n_stains
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.target_concentrations = None
    
    def _convert_rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB to optical density (OD)."""
        # Add a small value to avoid log(0)
        eps = 1e-6
        mask = (image == 0)
        image[mask] = eps
        
        # Standard RGB to OD conversion
        return -np.log10(image / 255.0)
    
    def _convert_od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """Convert optical density (OD) to RGB."""
        # OD to RGB conversion
        rgb = np.clip(255.0 * np.exp(-od * np.log(10)), 0, 255).astype(np.uint8)
        return rgb
    
    def _get_stain_matrix(self, image: np.ndarray) -> np.ndarray:
        """Extract stain matrix using Vahadane's method with sparse NMF."""
        # Convert to optical density
        od = self._convert_rgb_to_od(image.copy())
        
        # Reshape to pixel-feature format
        h, w, c = od.shape
        od = od.reshape((-1, c))
        
        # Filter out background pixels
        tissue_mask = np.sum(od, axis=1) > 0.15
        if np.sum(tissue_mask) == 0:
            warnings.warn("No valid pixels found for stain extraction")
            # Return default H&E stain matrix if no valid pixels
            return np.array([[0.5626, 0.2159], 
                             [0.7201, 0.8012], 
                             [0.4062, 0.5581]])
        
        od = od[tissue_mask]
        
        # Sparse NMF to extract stain matrix
        # Using scikit-learn's NMF with L1 regularization for sparsity
        model = NMF(n_components=self.n_stains, 
                   init='random', 
                   random_state=0,
                   solver='cd',
                   max_iter=1000,
                   alpha=self.lambda1,  # L1/L2 regularization parameter
                   l1_ratio=1.0)  # Use L1 regularization only
        
        try:
            W = model.fit_transform(od)  # Concentrations
            H = model.components_  # Stain matrix (transposed)
            
            # Normalize columns of H
            H = H / np.linalg.norm(H, axis=0)[np.newaxis, :]
            
            # Order H by first column (hematoxylin is usually first)
            idx = np.argsort(H[0])[::-1]
            H = H[:, idx]
            
            return H
            
        except Exception as e:
            warnings.warn(f"Error in stain matrix extraction: {e}")
            # Return default H&E stain matrix if extraction fails
            return np.array([[0.5626, 0.2159], 
                             [0.7201, 0.8012], 
                             [0.4062, 0.5581]])
    
    def _get_concentrations(self, image: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        """Extract stain concentrations using sparse NMF."""
        # Convert to optical density
        od = self._convert_rgb_to_od(image.copy())
        
        # Reshape to pixel-feature format
        h, w, c = od.shape
        od = od.reshape((-1, c))
        
        # Create sparse NMF model with fixed stain matrix
        model = NMF(n_components=self.n_stains,
                   init='custom',
                   random_state=0,
                   solver='cd',
                   max_iter=1000,
                   alpha=self.lambda2,  # L1/L2 regularization parameter
                   l1_ratio=1.0)  # Use L1 regularization only
        
        # Set components (H) and only solve for W
        model.components_ = stain_matrix.T
        W = model.fit_transform(od)
        
        # Reshape W back to image dimensions
        concentrations = W.reshape((h, w, self.n_stains))
        
        return concentrations
    
    def fit(self, target_image: np.ndarray) -> None:
        """
        Fit the normalizer to a target image.
        
        Args:
            target_image: Target image to extract stain parameters from
        """
        # Get stain matrix from target image
        self.target_stains = self._get_stain_matrix(target_image)
        
        # Get stain concentrations from target image
        target_concentrations = self._get_concentrations(target_image, self.target_stains)
        
        # Save concentration statistics
        self.target_concentrations = {
            'mean': np.mean(target_concentrations, axis=(0, 1)),
            'std': np.std(target_concentrations, axis=(0, 1))
        }
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Transform an image to match the target stain.
        
        Args:
            image: Input image to normalize
            
        Returns:
            Normalized image
        """
        if self.target_stains is None:
            warnings.warn("No target stains set. Using default H&E stains.")
            self.target_stains = np.array([[0.5626, 0.2159], 
                                           [0.7201, 0.8012], 
                                           [0.4062, 0.5581]])
        
        # Extract stain matrix from input image
        stain_matrix = self._get_stain_matrix(image)
        
        # Extract stain concentrations from input image
        concentrations = self._get_concentrations(image, stain_matrix)
        
        # Reshape for reconstruction
        h, w, _ = image.shape
        
        # Normalize concentrations (optional - can help with extreme variations)
        if self.target_concentrations is not None:
            source_mean = np.mean(concentrations, axis=(0, 1))
            source_std = np.std(concentrations, axis=(0, 1))
            target_mean = self.target_concentrations['mean']
            target_std = self.target_concentrations['std']
            
            # Transform concentrations to match target statistics
            norm_concentrations = ((concentrations - source_mean) / (source_std + 1e-8)) * target_std + target_mean
            
            # Ensure non-negative concentrations
            norm_concentrations = np.maximum(norm_concentrations, 0)
        else:
            norm_concentrations = concentrations
        
        # Reshape concentrations for matrix multiplication
        norm_concentrations_reshaped = norm_concentrations.reshape(-1, self.n_stains)
        
        # Reconstruct image with target stains
        od = np.dot(norm_concentrations_reshaped, self.target_stains.T)
        od = od.reshape(h, w, 3)
        
        # Convert back to RGB
        return self._convert_od_to_rgb(od)
    
    def save(self, path: str) -> None:
        """
        Save normalizer parameters to file.
        
        Args:
            path: Path to save the parameters
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save parameters
        np.savez(
            path,
            target_stains=self.target_stains,
            target_concentrations_mean=self.target_concentrations['mean'] if self.target_concentrations else np.array([1.0, 1.0]),
            target_concentrations_std=self.target_concentrations['std'] if self.target_concentrations else np.array([1.0, 1.0]),
            n_stains=self.n_stains,
            lambda1=self.lambda1,
            lambda2=self.lambda2
        )
    
    def load(self, path: str) -> None:
        """
        Load normalizer parameters from file.
        
        Args:
            path: Path to the parameters file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parameters file not found: {path}")
        
        # Load parameters
        data = np.load(path)
        
        self.target_stains = data['target_stains']
        self.target_concentrations = {
            'mean': data['target_concentrations_mean'],
            'std': data['target_concentrations_std']
        }
        self.n_stains = data['n_stains'].item()
        self.lambda1 = data['lambda1'].item()
        self.lambda2 = data['lambda2'].item()