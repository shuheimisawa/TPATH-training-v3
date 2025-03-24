@echo off
setlocal

:: Activate virtual environment
call fresh_venv\Scripts\activate.bat

:: Set environment variables
set PYTHONPATH=%CD%
set CUDA_VISIBLE_DEVICES=0

:: Create necessary directories
mkdir experiments\checkpoints 2>nul
mkdir experiments\logs 2>nul

:: Run the training script
python scripts/train.py ^
    --config experiments/configs/baseline.yaml ^
    --data-dir data_test/processed ^
    --checkpoint-dir experiments/checkpoints ^
    --log-dir experiments/logs ^
    --seed 42

:: Deactivate virtual environment
deactivate

endlocal 