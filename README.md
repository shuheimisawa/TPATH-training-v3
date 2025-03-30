# Glomeruli Detection and Classification

This repository contains code for an advanced pipeline for detecting and classifying glomeruli in whole slide images (WSI) of kidney biopsies, with a particular focus on distinguishing between normal and sclerotic glomeruli.

## Overview

The system uses a two-stage approach:
1. **Detection Stage**: First identifies all glomeruli regions in a slide
2. **Classification Stage**: Then classifies each detected glomerulus as normal, sclerotic, or partially sclerotic

Key features of this implementation:
- Advanced stain normalization using Vahadane method for robust color standardization
- Comprehensive feature extraction (texture, color, morphology)
- Two-stage detection and classification pipeline
- Automated reference image selection for normalization
- Analysis tools for visualizing results and feature importance

## Requirements

- Python >= 3.9
- PyTorch >= 1.11.0
- OpenSlide
- scikit-learn
- scikit-image
- OpenCV
- numpy, matplotlib, pandas
- tqdm

```bash
pip install -r requirements.txt
```

## Project Structure

```
src/
├── config/         # Configuration files
├── data/           # Data loading and preprocessing
├── evaluation/     # Evaluation scripts
├── inference/      # Inference scripts including two-stage pipeline
├── models/         # Model definitions (detection and classification)
└── utils/          # Utility functions, including stain normalization
scripts/            # Executable scripts for different pipeline stages
experiments/        # Results and experiment logs
```

## Workflow

The complete pipeline consists of these steps:

1. **Reference Selection**: Select an optimal reference image for stain normalization
2. **Stain Normalization**: Standardize the color appearance of all slides
3. **Training Detection Model**: Train the Cascade Mask R-CNN for glomeruli detection
4. **Training Classification Model**: Train the specialized classifier for the second stage
5. **Inference**: Run the two-stage pipeline on new slides
6. **Analysis**: Analyze and visualize the results

## Step-by-Step Guide

### 1. Reference Image Selection

First, select an optimal reference image for stain normalization:

```bash
python scripts/select_reference_image.py \
    --images-dir data/training_images \
    --output-dir experiments/reference_selection \
    --n-clusters 3 \
    --sample-size 100
```

This will:
- Analyze staining characteristics of images in the specified directory
- Cluster them based on color features
- Select the most representative image as reference
- Save the reference image to `experiments/reference_selection/reference_image.png`

### 2. Compare Stain Normalization Methods (Optional)

To visually compare different normalization methods:

```bash
python scripts/compare_normalizers.py \
    --image-path path/to/test_image.png \
    --reference-path experiments/reference_selection/reference_image.png \
    --output-dir experiments/normalization_comparison
```

This will generate a side-by-side comparison of original, Macenko, Reinhard, and Vahadane normalization methods.

### 3. Prepare Dataset

Process whole slide images into tiles and apply stain normalization:

```bash
python scripts/prepare_dataset.py \
    --input-dir data/slides \
    --annotations-dir data/annotations \
    --output-dir data/processed \
    --reference-image experiments/reference_selection/reference_image.png \
    --normalization-method vahadane
```

This will:
- Extract tiles from whole slide images
- Apply Vahadane stain normalization
- Split data into train/val/test sets
- Save processed tiles and annotations

### 4. Train Stage 1: Detection Model

Train the Cascade Mask R-CNN for glomeruli detection:

```bash
python scripts/train_detector.py \
    --data-config src/config/training_config.py \
    --output-dir experiments/detection_model
```

### 5. Train Stage 2: Classification Model

Train the specialized classifier for distinguishing between normal and sclerotic glomeruli:

```bash
python scripts/train_classifier.py \
    --train-dir data/processed/train \
    --val-dir data/processed/val \
    --output-dir experiments/classifier \
    --epochs 50 \
    --batch-size 16 \
    --class-names Normal Sclerotic Partially_sclerotic Uncertain
```

### 6. Run Inference on New Slides

Process new slides using the full two-stage pipeline:

```bash
python scripts/process_svs.py \
    --model-path experiments/detection_model/best_model.pth \
    --classifier-path experiments/classifier/best_model.pth \
    --input-dir data/test_slides \
    --output-dir experiments/results \
    --two-stage \
    --normalization-method vahadane \
    --reference-image experiments/reference_selection/reference_image.png \
    --visualize
```

This will:
- Process each slide in the input directory
- Apply stain normalization
- Detect all glomeruli using the first stage model
- Classify each detected glomerulus using the second stage model
- Generate visualizations and JSON results

### 7. Analyze Results

Extract and analyze features from the detected and classified glomeruli:

```bash
python scripts/analyze_glomeruli.py \
    --results-dir experiments/results \
    --output-dir experiments/analysis \
    --class-names Normal Sclerotic Partially_sclerotic Uncertain
```

This will:
- Extract features from all detected glomeruli
- Generate t-SNE visualization showing class separation
- Identify the most important features for classification
- Generate confusion matrix and classification report

## Key Components

### Two-Stage Pipeline

The pipeline is implemented in `src/inference/two_stage_pipeline.py` and consists of:

1. **Stage 1 (Detection)**: Using Cascade Mask R-CNN to detect all glomeruli
2. **Stage 2 (Classification)**: Using a specialized classifier with attention mechanism to distinguish between glomerulus types

### Stain Normalization

Implemented in `src/utils/stain_normalization.py` with three methods:
- **Vahadane**: Using sparse non-negative matrix factorization (recommended)
- **Macenko**: Using singular value decomposition
- **Reinhard**: Using color statistics matching

### Feature Extraction

Implemented in `src/utils/feature_extraction.py` with three main feature types:
- **Texture Features**: Gabor filters, Local Binary Patterns, GLCM
- **Color Features**: RGB, HSV, LAB statistics and histograms
- **Morphological Features**: Shape descriptors, moments, solidity, circularity

### Classification Model

Implemented in `src/models/glomeruli_classifier.py` with:
- CNN backbone with attention mechanism
- Multi-feature fusion layer
- Support for both image features and handcrafted features

## Troubleshooting

- **Memory Issues**: For large slides, try reducing `--tile-size` or increasing `--level`
- **Stain Normalization Errors**: Ensure reference image contains tissue and is representative of staining pattern
- **Classification Performance**: Try adjusting `--confidence-threshold` to balance precision and recall

## References

1. Vahadane, A., et al. (2016). Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images. IEEE TMI, 35(8), 1962-1971.
2. Cai, Z., & Vasconcelos, N. (2018). Cascade R-CNN: Delving Into High Quality Object Detection. CVPR.