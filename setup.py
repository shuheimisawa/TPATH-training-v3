from setuptools import setup, find_packages

setup(
    name="glomeruli-training",
    version="0.1.0",
    description="Glomeruli segmentation training package",
    author="TPATH",
    packages=find_packages(),
    install_requires=[
        "torch>=1.11.0",
        "torchvision>=0.12.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.2.0",
        "albumentations>=1.0.0",
        "pycocotools>=2.0.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.61.0",
        "PyYAML>=6.0",
        "torch-summary>=1.4.0",
        "tensorboard>=2.8.0"
    ],
    python_requires=">=3.9",
)