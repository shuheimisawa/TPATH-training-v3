# New file: scripts/analyze_glomeruli.py

import os
import numpy as np
import torch
import cv2
import argparse
import json
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from src.utils.feature_extraction import TextureFeatureExtractor, ColorFeatureExtractor, MorphologicalFeatureExtractor
from src.utils.io import load_image, load_json, save_json
from src.utils.logger import get_logger


def extract_features_from_results(results_dir, output_dir, class_names):
    """Extract features from detection results."""
    logger = get_logger(
        name="analyze_glomeruli",
        log_file=os.path.join(output_dir, "analyze_glomeruli.log")
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find result files
    result_files = glob(os.path.join(results_dir, '**/results.json'), recursive=True)
    logger.info(f"Found {len(result_files)} result files")
    
    # Initialize feature extractors
    texture_extractor = TextureFeatureExtractor()
    color_extractor = ColorFeatureExtractor()
    morphological_extractor = MorphologicalFeatureExtractor()
    
    # Initialize data structures
    all_features = []
    all_labels = []
    all_paths = []
    
    # Process each result file
    for result_file in tqdm(result_files, desc="Processing results"):
        try:
            # Load results
            results = load_json(result_file)
            
            # Get slide path
            slide_path = results['slide_path']
            slide_dir = os.path.dirname(result_file)
            
            # Process each tile result
            for tile_result in results['results']:
                # Check if two-stage pipeline results
                if 'combined' in tile_result:
                    # Get detections
                    boxes = np.array(tile_result['combined']['boxes'])
                    labels = np.array(tile_result['combined']['labels'])
                    class_names_list = tile_result['combined']['class_names']
                    
                    # Skip if no detections
                    if len(boxes) == 0:
                        continue
                    
                    # Load tile image
                    tile_path = os.path.join(slide_dir, 'tiles', f"tile_{tile_result['index']:06d}.png")
                    
                    if not os.path.exists(tile_path):
                        # Try to read from the slide
                        continue
                    
                    tile = load_image(tile_path)
                    
                    # Process each detection
                    for i, (box, label, class_name) in enumerate(zip(boxes, labels, class_names_list)):
                        # Skip background or uncertain
                        if class_name == 'background' or class_name == 'Uncertain':
                            continue
                        
                        # Extract patch
                        x1, y1, x2, y2 = map(int, box)
                        patch = tile[y1:y2, x1:x2]
                        
                        # Skip if patch is too small
                        if patch.shape[0] < 32 or patch.shape[1] < 32:
                            continue
                        
                        # Extract features
                        texture_features = texture_extractor.extract_features(patch)
                        color_features = color_extractor.extract_features(patch)
                        
                        # Create binary mask for morphological features
                        mask = np.zeros((patch.shape[0], patch.shape[1]), dtype=np.uint8)
                        mask = 255  # Simple full mask for now
                        morphological_features = morphological_extractor.extract_features(mask)
                        
                        # Combine features
                        features = {**texture_features, **color_features, **morphological_features}
                        
                        # Add to lists
                        all_features.append(features)
                        all_labels.append(class_names.index(class_name))
                        all_paths.append(f"{slide_path}_{tile_result['index']}_{i}")
                
        except Exception as e:
            logger.warning(f"Error processing {result_file}: {e}")
    
    logger.info(f"Extracted features from {len(all_features)} glomeruli")
    
    # Convert features to array
    feature_keys = sorted(all_features[0].keys())
    feature_array = np.array([[features[k] for k in feature_keys] for features in all_features])
    
    # Convert labels to array
    label_array = np.array(all_labels)
    
    # Save features and labels
    np.save(os.path.join(output_dir, 'features.npy'), feature_array)
    np.save(os.path.join(output_dir, 'labels.npy'), label_array)
    save_json(feature_keys, os.path.join(output_dir, 'feature_keys.json'))
    save_json(all_paths, os.path.join(output_dir, 'paths.json'))
    
    # Standardize features
    scaler = StandardScaler()
    feature_array_scaled = scaler.fit_transform(feature_array)
    
    # Apply t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=0)
    features_tsne = tsne.fit_transform(feature_array_scaled)
    
    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        if i in label_array:
            indices = np.where(label_array == i)[0]
            plt.scatter(
                features_tsne[indices, 0],
                features_tsne[indices, 1],
                label=class_name,
                alpha=0.7
            )
    
    plt.title('t-SNE Visualization of Glomeruli Features')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'tsne.png'), dpi=300)
    
    # Train a simple classifier
    X_train = feature_array_scaled
    y_train = label_array
    
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(X_train, y_train)
    
    # Get feature importances
    feature_importances = classifier.feature_importances_
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    
    # Get top 20 features
    indices = np.argsort(feature_importances)[-20:]
    plt.barh(range(20), feature_importances[indices])
    plt.yticks(range(20), [feature_keys[i] for i in indices])
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importances.png'), dpi=300)
    
    # Make predictions
    y_pred = classifier.predict(X_train)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_train, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    
    # Print classification report
    report = classification_report(y_train, y_pred, target_names=class_names, output_dict=True)
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_train, y_pred, target_names=class_names))
    
    # Save report
    save_json(report, os.path.join(output_dir, 'classification_report.json'))
    
    logger.info("Analysis completed")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze detected glomeruli')
    
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Path to directory with results')
    parser.add_argument('--output-dir', type=str, default='experiments/glomeruli_analysis',
                        help='Path to output directory')
    parser.add_argument('--class-names', type=str, nargs='+',
                        default=['Normal', 'Sclerotic', 'Partially_sclerotic', 'Uncertain'],
                        help='Class names')
    
    args = parser.parse_args()
    
    # Extract features
    extract_features_from_results(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        class_names=args.class_names
    )


if __name__ == '__main__':
    main()