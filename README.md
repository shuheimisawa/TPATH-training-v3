# Glomeruli Training

A deep learning project for glomeruli segmentation in kidney tissue images using U-Net architecture.

## Project Structure

```
glomeruli-training/
├── src/
│   ├── data/
│   │   └── dataset.py         # Dataset loading and preprocessing
│   ├── models/
│   │   └── unet.py           # U-Net model implementation
│   └── training/
│       ├── losses.py         # Segmentation loss functions
│       └── segmentation_trainer.py  # U-Net training loop
├── scripts/
│   ├── preprocess_slides.py   # Slide preprocessing script
│   └── train_unet.py         # U-Net training script
├── data_test/                # Test dataset
│   ├── raw/                  # Raw slides and QuPath annotations
│   └── processed/            # Processed patches and masks
├── data/                     # Main dataset
│   ├── raw/                  # Raw slides and QuPath annotations
│   └── processed/            # Processed patches and masks
├── tests/                    # Test files
├── notebooks/                # Jupyter notebooks
├── requirements.txt          # Python dependencies
└── setup.py                 # Package setup file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

The project expects data in the following format:
- Whole slide images (.svs format)
- QuPath annotations in GeoJSON format

### Preprocessing Parameters

The preprocessing script uses the following optimized parameters:
- Patch size: 512x512 pixels (standard for glomeruli segmentation)
- Magnification level: 0 (highest resolution)
- Minimum tissue fraction: 0.1
- Minimum glomeruli fraction: 0.05
- Maximum glomeruli fraction: 0.9

To preprocess your slides:
```bash
python scripts/preprocess_slides.py \
    --slides_dir data/raw/slides \
    --annotations_dir data/raw/annotations/QuPath \
    --output_dir data/processed \
    --patch_size 512 \
    --level 0 \
    --min_glomeruli_fraction 0.05 \
    --max_glomeruli_fraction 0.9 \
    --visualize
```

This will:
- Extract 512x512 patches centered on glomeruli
- Create corresponding segmentation masks
- Generate visualizations for quality control
- Automatically split data into train/validation/test sets (70%/15%/15%)
- Save all processed data in a format suitable for training

### Output Structure

The processed data directory will contain:
- `patches/`: Extracted image patches
- `masks/`: Corresponding segmentation masks
- `vis/`: Visualizations of patches and masks (if --visualize is used)
- `splits.json`: Train/val/test split information

## Training

To train the U-Net model:
```bash
python scripts/train_unet.py --data_dir data/processed --output_dir experiments
```

## Model Architecture

The project uses a U-Net architecture with the following features:
- Input: RGB images (3 channels, 512x512 pixels)
- Output: Multi-class segmentation masks
- Encoder path: 4 downsampling blocks
- Decoder path: 4 upsampling blocks with skip connections
- Final output: Number of classes (background + 4 glomeruli types)

## Classes

The model segments the following classes:
1. Background (label: 0)
2. Normal glomeruli (label: 1)
3. Sclerotic glomeruli (label: 2)
4. Partially sclerotic glomeruli (label: 3)
5. Uncertain cases (label: 4)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.