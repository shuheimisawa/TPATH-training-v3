# New file: scripts/select_reference_image.py

import os
import numpy as np
import cv2
import argparse
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.utils.stain_normalization import VahadaneNormalizer
from src.utils.io import load_image, save_image
from src.utils.logger import get_logger


def extract_color_features(image):
    """Extract color features from an image."""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Compute mean and std of each channel
    features = []
    
    for i in range(3):
        features.append(np.mean(lab[:, :, i]))
        features.append(np.std(lab[:, :, i]))
    
    # Compute histogram features
    for i in range(3):
        hist, _ = np.histogram(lab[:, :, i], bins=10, density=True)
        features.extend(hist)
    
    return np.array(features)


# Continuing scripts/select_reference_image.py

def select_reference_image(images_dir, output_dir, n_clusters=3, sample_size=None):
    """Select the best reference image for stain normalization."""
    logger = get_logger(
        name="select_reference_image",
        log_file=os.path.join(output_dir, "select_reference_image.log")
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find images
    image_paths = []
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        image_paths.extend(glob(os.path.join(images_dir, f"*{ext}")))
    
    logger.info(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        logger.error(f"No images found in {images_dir}")
        return None
    
    # Sample if too many images
    if sample_size and len(image_paths) > sample_size:
        logger.info(f"Sampling {sample_size} images")
        np.random.shuffle(image_paths)
        image_paths = image_paths[:sample_size]
    
    # Extract features
    features = []
    valid_paths = []
    
    for path in tqdm(image_paths, desc="Extracting features"):
        try:
            # Load image
            image = load_image(path)
            
            # Skip if too small
            if image.shape[0] < 64 or image.shape[1] < 64:
                continue
            
            # Extract features
            image_features = extract_color_features(image)
            
            # Add to list
            features.append(image_features)
            valid_paths.append(path)
            
        except Exception as e:
            logger.warning(f"Error processing {path}: {e}")
    
    logger.info(f"Extracted features from {len(features)} valid images")
    
    if len(features) == 0:
        logger.error("No valid images found")
        return None
    
    # Standardize features
    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Find center images of each cluster
    center_images = []
    
    for i in range(n_clusters):
        # Get indices of images in this cluster
        cluster_indices = np.where(clusters == i)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # Get features of images in this cluster
        cluster_features = features_scaled[cluster_indices]
        
        # Get cluster center
        cluster_center = kmeans.cluster_centers_[i]
        
        # Find image closest to center
        distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        
        # Add to list
        center_images.append({
            'path': valid_paths[closest_idx],
            'index': closest_idx,
            'cluster': i,
            'distance': np.min(distances),
            'cluster_size': len(cluster_indices)
        })
    
    # Sort by cluster size (prefer images from larger clusters)
    center_images.sort(key=lambda x: x['cluster_size'], reverse=True)
    
    # Log center images
    logger.info("Cluster center images:")
    for i, img in enumerate(center_images):
        logger.info(f"{i+1}. {img['path']} (Cluster {img['cluster']}, Size {img['cluster_size']})")
    
    # Select reference image (center of largest cluster)
    reference_image = center_images[0]
    reference_path = reference_image['path']
    
    logger.info(f"Selected reference image: {reference_path}")
    
    # Save reference image
    reference_output_path = os.path.join(output_dir, 'reference_image.png')
    reference_img = load_image(reference_path)
    save_image(reference_img, reference_output_path)
    
    # Visualize clusters
    plt.figure(figsize=(10, 8))
    
    # Plot all images
    for i in range(len(valid_paths)):
        plt.scatter(
            features_pca[i, 0], 
            features_pca[i, 1], 
            c=f'C{clusters[i]}', 
            alpha=0.5
        )
    
    # Mark center images
    for img in center_images:
        idx = img['index']
        plt.scatter(
            features_pca[idx, 0], 
            features_pca[idx, 1], 
            marker='*', 
            s=200, 
            edgecolor='black', 
            c=f'C{img["cluster"]}'
        )
    
    # Mark selected reference image
    ref_idx = reference_image['index']
    plt.scatter(
        features_pca[ref_idx, 0], 
        features_pca[ref_idx, 1], 
        marker='X', 
        s=300, 
        edgecolor='black', 
        c='red'
    )
    
    plt.title('Image Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(os.path.join(output_dir, 'clusters.png'), dpi=300)
    
    # Return path to reference image
    return reference_output_path


def normalize_images(images_dir, reference_path, output_dir, normalizer_type='vahadane', sample_size=None):
    """Normalize images using selected reference image."""
    logger = get_logger(
        name="normalize_images",
        log_file=os.path.join(output_dir, "normalize_images.log")
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reference image
    reference_image = load_image(reference_path)
    
    # Create normalizer
    if normalizer_type.lower() == 'vahadane':
        normalizer = VahadaneNormalizer()
    else:
        raise ValueError(f"Unsupported normalizer type: {normalizer_type}")
    
    # Fit normalizer to reference image
    normalizer.fit(reference_image)
    
    # Save normalizer parameters
    normalizer.save(os.path.join(output_dir, f'{normalizer_type}_params.npz'))
    
    # Find images
    image_paths = []
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        image_paths.extend(glob(os.path.join(images_dir, f"*{ext}")))
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Sample if too many images
    if sample_size and len(image_paths) > sample_size:
        logger.info(f"Sampling {sample_size} images")
        np.random.shuffle(image_paths)
        image_paths = image_paths[:sample_size]
    
    # Normalize images
    for path in tqdm(image_paths, desc="Normalizing images"):
        try:
            # Load image
            image = load_image(path)
            
            # Skip if too small
            if image.shape[0] < 64 or image.shape[1] < 64:
                continue
            
            # Normalize image
            normalized = normalizer.transform(image)
            
            # Save normalized image
            filename = os.path.basename(path)
            output_path = os.path.join(output_dir, f"normalized_{filename}")
            save_image(normalized, output_path)
            
        except Exception as e:
            logger.warning(f"Error normalizing {path}: {e}")
    
    logger.info("Normalization completed")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Select reference image for stain normalization')
    
    parser.add_argument('--images-dir', type=str, required=True,
                        help='Path to directory with images')
    parser.add_argument('--output-dir', type=str, default='experiments/reference_selection',
                        help='Path to output directory')
    parser.add_argument('--n-clusters', type=int, default=3,
                        help='Number of clusters')
    parser.add_argument('--sample-size', type=int, default=100,
                        help='Maximum number of images to process')
    parser.add_argument('--normalizer', type=str, default='vahadane',
                        choices=['vahadane', 'macenko', 'reinhard'],
                        help='Stain normalization method')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize images after selecting reference')
    parser.add_argument('--reference-path', type=str,
                        help='Path to existing reference image (skips selection)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select or use reference image
    if args.reference_path:
        reference_path = args.reference_path
        print(f"Using provided reference image: {reference_path}")
    else:
        reference_path = select_reference_image(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            n_clusters=args.n_clusters,
            sample_size=args.sample_size
        )
        
        if reference_path is None:
            print("Failed to select reference image")
            return
    
    # Normalize images if requested
    if args.normalize:
        normalize_images(
            images_dir=args.images_dir,
            reference_path=reference_path,
            output_dir=os.path.join(args.output_dir, 'normalized'),
            normalizer_type=args.normalizer,
            sample_size=args.sample_size
        )


if __name__ == '__main__':
    main()