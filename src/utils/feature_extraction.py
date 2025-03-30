# New file: src/utils/feature_extraction.py

import numpy as np
import cv2
from typing import Dict, List, Tuple, Union, Optional, Any
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
import warnings


class TextureFeatureExtractor:
    """Extract texture features from images."""
    
    def __init__(self, 
                 gabor_frequencies: List[float] = [0.1, 0.25, 0.4],
                 gabor_orientations: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
                 lbp_radius: int = 3,
                 lbp_points: int = 24,
                 glcm_distances: List[int] = [1, 3, 5],
                 glcm_angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """
        Initialize texture feature extractor.
        
        Args:
            gabor_frequencies: Frequencies for Gabor filters
            gabor_orientations: Orientations for Gabor filters
            lbp_radius: Radius for LBP
            lbp_points: Number of points for LBP
            glcm_distances: Distances for GLCM
            glcm_angles: Angles for GLCM
        """
        self.gabor_frequencies = gabor_frequencies
        self.gabor_orientations = gabor_orientations
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.glcm_distances = glcm_distances
        self.glcm_angles = glcm_angles
        
        # Generate Gabor kernels
        self.gabor_kernels = []
        for frequency in gabor_frequencies:
            for theta in gabor_orientations:
                kernel = np.real(gabor_kernel(frequency, theta=theta))
                self.gabor_kernels.append(kernel)
    
    def _compute_gabor_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute Gabor filter features."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image
        
        # Apply each Gabor kernel
        gabor_features = {}
        for i, kernel in enumerate(self.gabor_kernels):
            # Apply filter
            filtered = ndi.convolve(gray, kernel, mode='wrap')
            
            # Extract statistics
            gabor_features[f'gabor_mean_{i}'] = np.mean(filtered)
            gabor_features[f'gabor_std_{i}'] = np.std(filtered)
            gabor_features[f'gabor_energy_{i}'] = np.sum(filtered**2)
        
        return gabor_features
    
    def _compute_lbp_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute Local Binary Pattern features."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image
        
        # Compute LBP
        lbp = local_binary_pattern(gray, self.lbp_points, self.lbp_radius, method='uniform')
        
        # Compute histogram
        n_bins = self.lbp_points + 2
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        
        # Create features
        lbp_features = {f'lbp_hist_{i}': v for i, v in enumerate(hist)}
        
        return lbp_features
    
    def _compute_glcm_features(self, image: np.ndarray) -> Dict[str, float]:
        """Compute Gray-Level Co-occurrence Matrix features."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image
        
        # Scale to 0-255 and convert to uint8
        gray_scaled = (gray * 255).astype(np.uint8)
        
        # Compute GLCM
        glcm_features = {}
        glcm = graycomatrix(gray_scaled, 
                           distances=self.glcm_distances, 
                           angles=self.glcm_angles, 
                           levels=256, 
                           symmetric=True, 
                           normed=True)
        
        # Compute GLCM properties
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            try:
                feature = graycoprops(glcm, prop)
                
                # Add each distance-angle combination
                for i, d in enumerate(self.glcm_distances):
                    for j, a in enumerate(self.glcm_angles):
                        key = f'glcm_{prop}_d{d}_a{j}'
                        glcm_features[key] = feature[i, j]
            except Exception as e:
                warnings.warn(f"Error computing GLCM property {prop}: {e}")
        
        return glcm_features
    
    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract all texture features from an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of texture features
        """
        features = {}
        
        # Extract all feature types
        features.update(self._compute_gabor_features(image))
        features.update(self._compute_lbp_features(image))
        features.update(self._compute_glcm_features(image))
        
        return features


class MorphologicalFeatureExtractor:
    """Extract morphological features from segmented regions."""
    
    def __init__(self):
        """Initialize morphological feature extractor."""
        pass
    
    def extract_features(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Extract morphological features from a binary mask.
        
        Args:
            mask: Binary mask of the region
            
        Returns:
            Dictionary of morphological features
        """
        features = {}
        
        try:
            # Find contours
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'area': 0, 'perimeter': 0, 'circularity': 0, 'solidity': 0}
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Compute basic shape features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # Compute convex hull features
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0:
                solidity = area / hull_area
            else:
                solidity = 0
            
            # Add to features dictionary
            features['area'] = area
            features['perimeter'] = perimeter
            features['circularity'] = circularity
            features['solidity'] = solidity
            
            # Compute moments and Hu moments
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments)
            
            for i, moment in enumerate(hu_moments):
                features[f'hu_moment_{i}'] = moment[0]
            
            # Compute additional shape descriptors
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            features['aspect_ratio'] = aspect_ratio
            features['extent'] = area / (w * h) if w * h > 0 else 0
            
            # Compute equivalent diameter
            features['equiv_diameter'] = np.sqrt(4 * area / np.pi)
            
            return features
            
        except Exception as e:
            warnings.warn(f"Error computing morphological features: {e}")
            return {'area': 0, 'perimeter': 0, 'circularity': 0, 'solidity': 0}


class ColorFeatureExtractor:
    """Extract color features from images."""
    
    def __init__(self, bins: int = 32):
        """
        Initialize color feature extractor.
        
        Args:
            bins: Number of bins for color histograms
        """
        self.bins = bins
    
    def _compute_color_statistics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute color channel statistics."""
        # Ensure image is RGB
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Expected RGB image")
        
        # Compute statistics for each channel
        features = {}
        
        # RGB statistics
        for i, channel in enumerate(['r', 'g', 'b']):
            features[f'{channel}_mean'] = np.mean(image[:, :, i])
            features[f'{channel}_std'] = np.std(image[:, :, i])
            features[f'{channel}_min'] = np.min(image[:, :, i])
            features[f'{channel}_max'] = np.max(image[:, :, i])
        
        # RGB ratios (useful for histological analysis)
        features['r_g_ratio'] = features['r_mean'] / features['g_mean'] if features['g_mean'] > 0 else 0
        features['r_b_ratio'] = features['r_mean'] / features['b_mean'] if features['b_mean'] > 0 else 0
        features['g_b_ratio'] = features['g_mean'] / features['b_mean'] if features['b_mean'] > 0 else 0
        
        return features
    
    def _compute_hsv_features(self, image: np.ndarray) -> Dict[str, float]:
        """Compute HSV color features."""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Compute statistics for each channel
        features = {}
        
        for i, channel in enumerate(['h', 's', 'v']):
            features[f'{channel}_mean'] = np.mean(hsv[:, :, i])
            features[f'{channel}_std'] = np.std(hsv[:, :, i])
        
        # Compute histograms
        h_hist, _ = np.histogram(hsv[:, :, 0], bins=self.bins, range=(0, 180), density=True)
        s_hist, _ = np.histogram(hsv[:, :, 1], bins=self.bins, range=(0, 256), density=True)
        v_hist, _ = np.histogram(hsv[:, :, 2], bins=self.bins, range=(0, 256), density=True)
        
        # Add histograms to features
        for i, hist_val in enumerate(h_hist):
            features[f'h_hist_{i}'] = hist_val
        
        for i, hist_val in enumerate(s_hist):
            features[f's_hist_{i}'] = hist_val
        
        for i, hist_val in enumerate(v_hist):
            features[f'v_hist_{i}'] = hist_val
        
        return features
    
    def _compute_lab_features(self, image: np.ndarray) -> Dict[str, float]:
        """Compute LAB color features."""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Compute statistics for each channel
        features = {}
        
        for i, channel in enumerate(['l', 'a', 'b']):
            features[f'{channel}_mean'] = np.mean(lab[:, :, i])
            features[f'{channel}_std'] = np.std(lab[:, :, i])
        
        return features
    
    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract color features from an image.
        
        Args:
            image: Input RGB image
            
        Returns:
            Dictionary of color features
        """
        features = {}
        
        # Extract all color feature types
        features.update(self._compute_color_statistics(image))
        features.update(self._compute_hsv_features(image))
        features.update(self._compute_lab_features(image))
        
        return features