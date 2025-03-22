# Contributing to Glomeruli Segmentation

Thank you for considering contributing to the Glomeruli Segmentation project! This document provides guidelines to help you contribute effectively.

## Code Style

- Follow PEP 8 guidelines for Python code
- Use descriptive variable names
- Include docstrings for all functions, classes, and modules
- Add type hints where appropriate

## Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add some feature'`)
7. Push to the branch (`git push origin feature/your-feature-name`)
8. Create a new Pull Request

## Pull Request Process

1. Ensure your code follows the project's coding standards
2. Update the README.md with details of changes if appropriate
3. The PR should work on the main development branches
4. Include appropriate test coverage for your changes

## Testing

- Write unit tests for all new functionality
- Run existing tests to ensure your changes don't break existing functionality
- Aim for high test coverage of your code

## Documentation

- Keep documentation up to date with code changes
- Document all public functions, classes, and modules with clear docstrings
- Include examples in docstrings where helpful

## Issue Reporting

When reporting issues, please include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Screenshots if applicable
- Environment details (OS, Python version, library versions)

Thank you for your contributions!


# README.md
# Glomeruli Instance Segmentation

A deep learning-based approach for glomeruli instance segmentation from multistained renal biopsy pathologic images.

## Overview

This project implements a Cascade Mask R-CNN model for detecting, classifying, and segmenting glomeruli in renal biopsy images. It supports three types of special staining methods (PAS, PASM, and Masson) and can classify glomeruli into three categories:

1. **GN (Glomeruli with Normal Structure)**: Glomeruli with completely open capillary tufts that are structurally normal. This includes normal glomeruli, glomeruli with mild lesions, and glomeruli with thickened glomerular basement membrane without other lesions.

2. **GL (Glomeruli with Other Lesions)**: Glomeruli with lesions other than global sclerosis that have lost any part of their structure. This includes segmental sclerosis, moderate to severe mesangial hypercellularity or expansion, crescents, and apparent endothelial proliferation.

3. **GS (Global Sclerosis)**: Glomeruli with complete obliteration of the entire glomerular tuft and loss of normal structure.

## Installation

### Using Conda

```bash
# Clone the repository
git clone https://github.com/shuheimisawa/glomeruli-segmentation.git
cd glomeruli-segmentation

# Create conda environment
conda env create -f environment.yml

# Activate the environment
conda activate glomeruli-segmentation
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/shuheimisawa/glomeruli-segmentation.git
cd glomeruli-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

The system supports renal biopsy images with PAS, PASM, and Masson trichrome staining. You can prepare your dataset using the provided script:

```bash
python scripts/prepare_dataset.py --input-dir /path/to/raw/data --output-dir data
```

## Training

```bash
python scripts/train.py --config experiments/configs/baseline.yaml
```

For distributed training with multiple GPUs:

```bash
python scripts/train.py --config experiments/configs/improved_v1.yaml --distributed
```

## Evaluation

```bash
python scripts/evaluate.py --model-path experiments/checkpoints/best_model.pth \
                          --data-dir data/test \
                          --visualize
```

## Model Export

Export the trained model for deployment:

```bash
python scripts/export_model.py --model-path experiments/checkpoints/best_model.pth \
                              --output-path experiments/exported_models \
                              --format torchscript
```

## Project Structure

- `src/`: Source code
  - `config/`: Configuration files
  - `data/`: Dataset and data loading utilities
  - `models/`: Model architecture
  - `training/`: Training utilities
  - `evaluation/`: Evaluation metrics and visualization
  - `utils/`: Utility functions
- `scripts/`: Scripts for training, evaluation, etc.
- `experiments/`: Experiment configurations, logs, and results
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `tests/`: Unit tests

