# src/utils/stain_normalization.py
import os
import numpy as np
import cv2
from typing import Dict, Optional, Union, Tuple, Any
import warnings


class MacenkoNormalizer:
    """
    Stain normalization using Macenko's method.
    
    Reference:
    Macenko, M., et al. (2009). A method for normalizing histology slides for
    quantitative analysis. IEEE ISBI, 1107-1110.
    """
    
    def __init__(self, 
                 target_stains: Optional[np.ndarray] = None,
                 target_concentrations: Optional[np.ndarray] = None,
                 stain_vector_1: Optional[np.ndarray] = None,
                 stain_vector_2: Optional[np.ndarray] = None,
                 alpha: float = 1.0,
                 beta: float = 0.15):
        """
        Initialize Macenko stain normalizer.
        
        Args:
            target_stains: Target stain matrix
            target_concentrations: Target stain concentrations
            stain_vector_1: Target stain vector 1 (usually hematoxylin)
            stain_vector_2: Target stain vector 2 (usually eosin)
            alpha: Percentile for stain extraction
            beta: Regularization parameter
        """
        self.target_stains = target_stains
        self.target_concentrations = target_concentrations
        self.stain_vector_1 = stain_vector_1
        self.stain_vector_2 = stain_vector_2
        self.alpha = alpha
        self.beta = beta
        
        # Default target stain vectors (H&E standard)
        if stain_vector_1 is None:
            self.stain_vector_1 = np.array([0.5626, 0.7201, 0.4062])  # H
        if stain_vector_2 is None:
            self.stain_vector_2 = np.array([0.2159, 0.8012, 0.5581])  # E
        
        if self.target_stains is None and (stain_vector_1 is not None or stain_vector_2 is not None):
            self.target_stains = np.vstack((self.stain_vector_1, self.stain_vector_2)).T
    
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
        # OD to RGB conversion (inverse of _convert_rgb_to_od)
        rgb = np.clip(255.0 * np.exp(-od * np.log(10)), 0, 255).astype(np.uint8)
        return rgb
    
    def _get_stain_matrix(self, image: np.ndarray) -> np.ndarray:
        """Extract stain matrix using Macenko's method."""
        # Convert to optical density
        od = self._convert_rgb_to_od(image.copy())
        
        # Reshape to one column per pixel
        od = od.reshape((-1, 3)).T
        
        # Filter out background pixels (white)
        mask = (od.sum(axis=0) > self.beta)
        if mask.sum() == 0:
            warnings.warn("No valid pixels found for stain extraction")
            return np.vstack((self.stain_vector_1, self.stain_vector_2)).T
        
        od = od[:, mask]
        
        # Compute eigenvectors
        try:
            _, eigvecs = np.linalg.eigh(np.cov(od))
            eigvecs = eigvecs[:, [2, 1]]  # Use the two largest eigenvectors
            
            # Project data onto eigenvectors
            proj = np.dot(eigvecs.T, od)
            
            # Find angle of each point in projected space
            phi = np.arctan2(proj[1], proj[0])
            
            # Find extremes (percentiles) of angles
            minPhi = np.percentile(phi, self.alpha)
            maxPhi = np.percentile(phi, 100 - self.alpha)
            
            # Get corresponding vectors
            v1 = np.dot(eigvecs, np.array([np.cos(minPhi), np.sin(minPhi)]))
            v2 = np.dot(eigvecs, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
            
            # Ensure first vector is hematoxylin (H) and second is eosin (E)
            if v1[0] < v2[0]:
                stain_vectors = np.array([v1, v2])
            else:
                stain_vectors = np.array([v2, v1])
                
            # Normalize the stain vectors
            stain_vectors /= np.linalg.norm(stain_vectors, axis=1)[:, np.newaxis]
            
            return stain_vectors.T
        except Exception as e:
            warnings.warn(f"Error in stain matrix extraction: {e}")
            return np.vstack((self.stain_vector_1, self.stain_vector_2)).T
    
    def _get_concentrations(self, image: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        """Extract stain concentrations."""
        # Convert to optical density
        od = self._convert_rgb_to_od(image.copy())
        
        # Reshape to one column per pixel
        od_reshaped = od.reshape((-1, 3)).T
        
        # Solve for concentrations using least squares
        concentrations = np.linalg.lstsq(stain_matrix, od_reshaped, rcond=None)[0]
        
        # Reshape back to image-like shape
        return concentrations.T.reshape((*image.shape[0:2], 2))
    
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
            self.target_stains = np.vstack((self.stain_vector_1, self.stain_vector_2)).T
        
        if self.target_concentrations is None:
            # If no target concentrations, just use identity transformation
            target_concentrations_mean = np.array([1.0, 1.0])
            target_concentrations_std = np.array([1.0, 1.0])
        else:
            target_concentrations_mean = self.target_concentrations['mean']
            target_concentrations_std = self.target_concentrations['std']
        
        # Extract stain matrix from input image
        stain_matrix = self._get_stain_matrix(image)
        
        # Extract stain concentrations from input image
        concentrations = self._get_concentrations(image, stain_matrix)
        
        # Normalize stain concentrations
        source_concentrations_mean = np.mean(concentrations, axis=(0, 1))
        source_concentrations_std = np.std(concentrations, axis=(0, 1))
        
        # Transform concentrations to match target statistics
        norm_concentrations = ((concentrations - source_concentrations_mean) / 
                               (source_concentrations_std + 1e-8)) * target_concentrations_std + target_concentrations_mean
        
        # Ensure non-negative concentrations
        norm_concentrations = np.maximum(norm_concentrations, 0)
        
        # Reshape for matrix multiplication
        norm_concentrations_reshaped = norm_concentrations.reshape(-1, 2).T
        
        # Convert back to optical density space
        od = np.dot(self.target_stains, norm_concentrations_reshaped)
        
        # Reshape back to image dimensions
        od = od.T.reshape((*image.shape[0:2], 3))
        
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
            stain_vector_1=self.stain_vector_1,
            stain_vector_2=self.stain_vector_2,
            alpha=self.alpha,
            beta=self.beta
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
        self.stain_vector_1 = data['stain_vector_1']
        self.stain_vector_2 = data['stain_vector_2']
        self.alpha = data['alpha'].item()
        self.beta = data['beta'].item()


class ReinmetNormalizer:
    """Stain normalization using Reinhard's method (color transfer)."""
    
    def __init__(self, target_means: Optional[np.ndarray] = None, 
                target_stds: Optional[np.ndarray] = None):
        """
        Initialize Reinhard normalizer.
        
        Args:
            target_means: Target means in LAB color space
            target_stds: Target standard deviations in LAB color space
        """
        self.target_means = target_means
        self.target_stds = target_stds
    
    def fit(self, target_image: np.ndarray) -> None:
        """
        Fit the normalizer to a target image.
        
        Args:
            target_image: Target image to extract color statistics from
        """
        # Convert to LAB color space
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB)
        
        # Calculate mean and std
        self.target_means = np.mean(target_lab, axis=(0, 1))
        self.target_stds = np.std(target_lab, axis=(0, 1))
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Transform an image to match the target color statistics.
        
        Args:
            image: Input image to normalize
            
        Returns:
            Normalized image
        """
        if self.target_means is None or self.target_stds is None:
            warnings.warn("Target statistics not set. Cannot normalize.")
            return image
        
        # Convert to LAB color space
        source_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Calculate source statistics
        source_means = np.mean(source_lab, axis=(0, 1))
        source_stds = np.std(source_lab, axis=(0, 1))
        
        # Normalize
        normalized_lab = ((source_lab - source_means) / (source_stds + 1e-8)) * self.target_stds + self.target_means
        
        # Ensure L channel is in valid range [0, 255]
        normalized_lab[:, :, 0] = np.clip(normalized_lab[:, :, 0], 0, 255)
        
        # Convert back to RGB
        normalized_rgb = cv2.cvtColor(normalized_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return normalized_rgb
    
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
            target_means=self.target_means,
            target_stds=self.target_stds
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
        
        self.target_means = data['target_means']
        self.target_stds = data['target_stds']


class StainNormalizationTransform:
    """Transform for applying stain normalization in the data pipeline."""
    
    def __init__(self, method: str = 'macenko', target_image_path: Optional[str] = None, 
                params_path: Optional[str] = None):
        """
        Initialize stain normalization transform.
        
        Args:
            method: Normalization method ('macenko' or 'reinhard')
            target_image_path: Path to target reference image
            params_path: Path to saved normalization parameters
        """
        self.method = method.lower()
        
        # Create normalizer based on method
        if self.method == 'macenko':
            self.normalizer = MacenkoNormalizer()
        elif self.method == 'reinhard':
            self.normalizer = ReinmetNormalizer()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        # Load parameters from file if provided
        if params_path is not None and os.path.exists(params_path):
            self.normalizer.load(params_path)
        # Or fit to target image if provided
        elif target_image_path is not None and os.path.exists(target_image_path):
            target_image = cv2.cvtColor(
                cv2.imread(target_image_path), 
                cv2.COLOR_BGR2RGB
            )
            self.normalizer.fit(target_image)
        else:
            warnings.warn("No reference image or parameters provided. Stain normalization may not work effectively.")
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply stain normalization to an image.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        try:
            # Ensure image is RGB
            if len(image.shape) != 3 or image.shape[2] != 3:
                warnings.warn(f"Expected RGB image, got shape {image.shape}. Skipping normalization.")
                return image
            
            # Apply normalization
            normalized = self.normalizer.transform(image)
            
            return normalized
        except Exception as e:
            warnings.warn(f"Error in stain normalization: {e}")
            # Return original image if normalization fails
            return image